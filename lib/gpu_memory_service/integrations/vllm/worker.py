# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service Worker subclass for vLLM integration.

This module provides a custom Worker class that properly integrates with
GPU Memory Service for VA-stable weight sharing and unmap/remap functionality.

Usage:
    Set --worker-cls=gpu_memory_service.integrations.vllm.worker:GMSWorker
"""

from __future__ import annotations

import gc
import logging
import sys
from contextlib import nullcontext
from typing import List, Optional

import torch
from gpu_memory_service.client.memory_manager import StaleMemoryLayoutError
from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path, kv_cache_socket_tag
from gpu_memory_service.integrations.common import patch_empty_cache
from gpu_memory_service.integrations.common.utils import GMS_TAGS, get_gms_lock_mode
from gpu_memory_service.integrations.vllm.model_loader import register_gms_loader
from gpu_memory_service.integrations.vllm.patches import (
    apply_shadow_mode_patches,
    patch_memory_snapshot,
)
from gpu_memory_service.integrations.vllm.utils import is_shadow_mode

logger = logging.getLogger(__name__)

# Trigger model loader registration and utility patches on import
register_gms_loader()

# Apply core utility patches (always needed for GMS)
patch_empty_cache()
patch_memory_snapshot()

# Apply shadow mode patches if shadow mode is enabled
apply_shadow_mode_patches()

logger.info("[GMS] Worker module loaded - model loader registered, all patches applied")

# Import Worker after patches are applied
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402


class GMSWorker(Worker):
    """vLLM Worker subclass with GMS integration."""

    def init_device(self) -> None:
        """Initialize device with early GMS connection.

        We set CUDA device and establish GMS connection BEFORE calling super()
        so that MemorySnapshot.measure can query committed bytes.
        """
        from vllm.platforms import current_platform

        # Set CUDA device first (vLLM provides self.local_rank)
        device = self.local_rank
        current_platform.set_device(torch.device(f"cuda:{device}"))

        # Establish weights GMS connection (so MemorySnapshot can query committed bytes).
        # Lock type is determined by model_loader_extra_config, set upstream by
        # configure_gms_lock_mode() in main.py.
        extra = (
            getattr(self.vllm_config.load_config, "model_loader_extra_config", {}) or {}
        )
        mode = get_gms_lock_mode(extra)
        get_or_create_gms_client_memory_manager(
            get_socket_path(device, "weights"),
            device,
            mode=mode,
            tag="weights",
        )
        # Parent will set device again (harmless) and do memory checks
        super().init_device()

        # __class__ swap: preserves object identity so vLLM's internal
        # references see our overrides.
        if is_shadow_mode() and hasattr(self, "model_runner"):
            from gpu_memory_service.integrations.vllm.model_runner import (
                GMSShadowModelRunner,
            )

            self.model_runner.__class__ = GMSShadowModelRunner
            self.model_runner.enter_shadow_init()
            logger.info("[GMS] Injected GMSShadowModelRunner via __class__ swap")

    def determine_available_memory(self) -> int:
        """
        Determine actual available memory for the engine.

        During a failover scenario, this function may be called while there is an active engine colocated on the same device.
        We want our assessment to ignore the kv cache allocation of the active engine if there is one.
        """
        if not is_shadow_mode():
            return super().determine_available_memory()

        torch.cuda.reset_peak_memory_stats()
        self.model_runner.profile_run()
        torch.cuda.synchronize()
        torch_peak = torch.cuda.max_memory_allocated()

        # GMS weights mapped via cuMemMap are invisible to PyTorch's memory
        # stats on RO engines. Add them explicitly. On RW engines, torch_peak
        # already includes weights so skip to avoid double-counting.
        weights_memory = int(getattr(self.model_runner, "model_memory_usage", 0))
        if torch_peak < weights_memory:
            non_kv_cache_memory = torch_peak + weights_memory
        else:
            non_kv_cache_memory = torch_peak

        projected_available = self.requested_memory - non_kv_cache_memory

        msg = (
            "[GMS] Shadow mode: projected available memory "
            "%.2f GiB (requested=%.2f GiB, non_kv=%.2f GiB, "
            "torch_peak=%.2f GiB, weights=%.2f GiB)"
            % (
                projected_available / (1 << 30),
                self.requested_memory / (1 << 30),
                non_kv_cache_memory / (1 << 30),
                torch_peak / (1 << 30),
                weights_memory / (1 << 30),
            )
        )
        logger.info(msg)
        print(msg, flush=True)

        return int(projected_available)

    def initialize_from_config(self, kv_cache_config) -> None:
        """Allocate KV cache through the GMS kv_cache mempool.

        Each engine connects RW to its own per-engine kv_cache GMS server.
        Under intra-pod failover the operator spawns one kv_cache socket per
        engine (kv_cache_0, kv_cache_1, ...) so there is no cross-engine RW
        contention; each engine is a single writer on its own server.

        Shadow engines additionally allocate KV tensors over scratch-aliased
        backing (one small physical chunk mapped at every granule of the
        tensor's VA range) to keep total GPU memory ≈ 1× the active engine's
        budget. At wake, commit_real_backing swaps scratch for real
        per-chunk physical at the same VAs; captured cudagraphs continue
        to work because they reference VAs, not physical pages.

        Non-shadow with enable_sleep_mode: same connect-RW path, regular
        allocations (no scratch trick — there is no sibling engine to
        coexist with).
        """
        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized

        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        device = self.local_rank
        wants_gms_kv = (
            is_shadow_mode() or self.vllm_config.model_config.enable_sleep_mode
        )
        if wants_gms_kv:
            get_or_create_gms_client_memory_manager(
                get_socket_path(device, kv_cache_socket_tag()),
                device,
                mode=RequestedLockType.RW,
                tag="kv_cache",
            )
            with gms_use_mem_pool("kv_cache", torch.device(f"cuda:{device}")):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def load_model(self, *args, **kwargs) -> None:
        """Load model with corrected memory accounting.

        After the parent loads the model, we correct the model_memory_usage
        to reflect the actual bytes imported from GMS (not the delta measured
        by vLLM's memory tracking).
        """
        super().load_model(*args, **kwargs)

        # Correct memory accounting for GMS-imported weights
        try:
            from gpu_memory_service.integrations.vllm.model_loader import (
                get_imported_weights_bytes,
            )

            imported_bytes = int(get_imported_weights_bytes())
            if (
                imported_bytes > 0
                and hasattr(self, "model_runner")
                and self.model_runner is not None
            ):
                old_usage = getattr(self.model_runner, "model_memory_usage", 0)
                self.model_runner.model_memory_usage = imported_bytes
                logger.info(
                    "[GMS] Corrected model_memory_usage: %.2f GiB -> %.2f GiB",
                    old_usage / (1 << 30),
                    imported_bytes / (1 << 30),
                )
        except Exception as e:
            logger.debug("[GMS] Could not correct memory accounting: %s", e)

    def sleep(self, level: int = 1) -> None:
        """
        vLLM sleep implementation with GMS integration.

        NOTE: We do NOT call super().sleep() because it tries to copy GPU buffers to CPU,
              which segfaults on already-unmapped GMS memory.

        Handles two cases for KV cache:
        1. Normal: KV cache was allocated via GMS, unmap + abort
        2. Shadow: KV cache was skipped at startup, manager has no allocations
           (unmap_all_vas is a no-op, abort disconnects)
        """
        free_bytes_before = torch.cuda.mem_get_info()[0]

        # Unmap GMS weights: synchronize + unmap all VAs + disconnect
        weights_manager = get_gms_client_memory_manager("weights")
        assert weights_manager is not None, "GMS weights client is not initialized"
        assert not weights_manager.is_unmapped, "GMS weights are already unmapped"
        weights_manager.unmap_all_vas()
        weights_manager.abort()

        # Unmap GMS KV cache: unmap all VAs + disconnect
        kv_cache_manager = get_gms_client_memory_manager("kv_cache")
        if kv_cache_manager is None:
            logger.info("[GMS] No kv_cache manager, skipping kv_cache sleep")
        elif kv_cache_manager.has_plan_a_scratch:
            # Shadow scratch state: leave scratch backing mapped and the GMS
            # session open. There's no contention to release — each engine
            # has its own kv_cache server. On wake, commit_real_backing
            # swaps scratch for real per-chunk physical at the same VAs.
            logger.info("[GMS] Sleep: kept scratch-aliased KV backing")
        else:
            assert not kv_cache_manager.is_unmapped, "GMS KV cache is already unmapped"
            kv_cache_manager.unmap_all_vas()
            kv_cache_manager.abort()

        gc.collect()
        torch.cuda.empty_cache()

        free_bytes_after, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after - free_bytes_before
        used_bytes = total - free_bytes_after
        logger.info(
            "Sleep freed %.2f GiB, %.2f GiB still in use.",
            freed_bytes / (1 << 30),
            used_bytes / (1 << 30),
        )

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """vLLM wake implementation with GMS integration.

        For kv_cache, handles two cases:
        1. Shadow wake: scratch-aliased backing was installed at init over
           the engine's own kv_cache GMS (already RW); swap to unique
           per-chunk physical at the same VAs via commit_real_backing.
        2. Normal sleep/wake: reconnect + reallocate + remap.
        """
        if (
            hasattr(self.model_runner, "exit_shadow_init")
            and self.model_runner.in_shadow_init
        ):
            self.model_runner.exit_shadow_init()

        if tags is None:
            tags = list(GMS_TAGS)

        if "weights" in tags:
            weights_manager = get_gms_client_memory_manager("weights")
            assert weights_manager is not None, "GMS weights client is not initialized"
            assert weights_manager.is_unmapped, "GMS weights are not unmapped"

            # These errors are fatal and unrecoverable in a worker subprocess:
            # the worker cannot serve requests without weights. sys.exit(1)
            # ensures clean termination so the orchestrator (K8s) can restart.
            try:
                weights_manager.connect(RequestedLockType.RO, timeout_ms=30_000)
                weights_manager.remap_all_vas()
            except TimeoutError:
                logger.error(
                    "Fatal: timed out waiting for GMS RO lock during remap "
                    "(GMS may be down or RW lock held indefinitely)"
                )
                sys.exit(1)
            except StaleMemoryLayoutError as e:
                logger.error(
                    "Fatal: weight layout changed while unmapped, cannot remap: %s", e
                )
                sys.exit(1)
            except ConnectionError as e:
                logger.error("Fatal: cannot connect to GMS during remap: %s", e)
                sys.exit(1)

        if "kv_cache" in tags:
            kv_cache_manager = get_gms_client_memory_manager("kv_cache")
            if kv_cache_manager is not None and kv_cache_manager.has_plan_a_scratch:
                # Shadow wake path. The manager has been connected RW since
                # init (each engine has its own kv_cache server, no
                # contention). Swap scratch-aliased backing for real
                # per-chunk physical at the same VAs. Cudagraphs captured at
                # init continue to work because they reference VAs.
                logger.info("[GMS] wake: committing real KV backing")
                kv_cache_manager.commit_real_backing()
                # NixlConnector registration was deferred during scratch phase
                # (patches.patch_register_kv_caches). Fire it now against the
                # just-mapped real handles.
                self._register_kv_caches_with_nixl()
            elif kv_cache_manager is not None and kv_cache_manager.is_unmapped:
                # Normal sleep/wake: reconnect, reallocate handles, remap VAs.
                kv_cache_manager.connect(RequestedLockType.RW)
                kv_cache_manager.reallocate_all_handles(tag="kv_cache")
                kv_cache_manager.remap_all_vas()

            # Reinitialize FP8 KV scales if needed
            if self.cache_config.cache_dtype.startswith("fp8") and hasattr(
                self.model_runner, "init_fp8_kv_scales"
            ):
                self.model_runner.init_fp8_kv_scales()

    def _register_kv_caches_with_nixl(self) -> None:
        """Fire the NixlConnector KV-cache registration after Plan A swap.

        During scratch phase the patches.patch_register_kv_caches gate intercepts
        register_kv_caches(dict) and stashes the dict on the connector as
        self._plan_a_pending_kv_caches. We replay that here, NOT
        self.model_runner.kv_caches — the latter is the list-of-tensors view
        set by vLLM, and NixlConnector.register_kv_caches does kv_caches.values()
        which requires the dict form.

        Imports from the package root (vllm.distributed.kv_transfer) — the
        kv_connector.v1.base re-exports were unreliable across vLLM versions,
        and a silent except ImportError here would make this a no-op,
        leaving NixlConnectorWorker.kv_topo=None and crashing on the first
        scheduler tick. See DYN-2714 commit 6174e090f5c for history.
        """
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            return
        group = get_kv_transfer_group()
        pending = getattr(group, "_plan_a_pending_kv_caches", None)
        if not pending:
            # Nothing was stashed — either no deferred registration, or a
            # non-NixlConnector connector that didn't hit the patched path.
            return
        group.register_kv_caches(pending)
        # Drop the stash so a second call is a no-op.
        try:
            delattr(group, "_plan_a_pending_kv_caches")
        except AttributeError:
            pass
        logger.info(
            "[Plan A] Registered %d kv_cache tensors with KV transfer group",
            len(pending),
        )

    def _maybe_get_memory_pool_context(self, tag: str):
        """Route tag-scoped runtime allocations to the right allocator.

        Weight tensors are allocated explicitly in the GMS model-loader path,
        not through vLLM's tagged runtime allocator hook. For `weights` we
        therefore only suppress CuMemAllocator here so it does not interfere
        with the loader-managed GMS allocations. `kv_cache` is the tag that
        actually allocates through this hook, so it uses the dedicated GMS
        mempool.
        """
        if tag == "weights":
            logger.debug("[GMS] Skipping CuMemAllocator for weights")
            return nullcontext()
        if tag == "kv_cache":
            return gms_use_mem_pool("kv_cache", torch.device("cuda", self.local_rank))
        return super()._maybe_get_memory_pool_context(tag)
