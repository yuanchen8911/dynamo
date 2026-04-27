# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches GMS server processes per GPU. By default: one for weights, one for
kv_cache. Under intra-pod failover (set GMS_FAILOVER_ENGINE_COUNT=N), spawns
N independent kv_cache servers tagged kv_cache_0..kv_cache_{N-1}, one per
engine container. Each engine connects RW exclusive on its own kv_cache
socket; the weights socket stays shared.

Writes a ready file once all expected UDS sockets are present. Runs until
SIGTERM (pod termination kills it).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from gpu_memory_service.common.cuda_utils import list_devices
from gpu_memory_service.common.utils import get_socket_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_READY_FILE = "gms-ready"


def _kv_cache_tags() -> tuple[str, ...]:
    """Return the kv_cache tag(s) to spawn based on failover engine count.

    Default (no failover): ("kv_cache",) — preserves existing behavior.
    With GMS_FAILOVER_ENGINE_COUNT=N>=2: ("kv_cache_0", ..., "kv_cache_{N-1}")
    so each engine container has its own kv_cache server and RW lock.
    """
    raw = os.environ.get("GMS_FAILOVER_ENGINE_COUNT")
    try:
        count = int(raw) if raw else 1
    except ValueError:
        logger.warning("Invalid GMS_FAILOVER_ENGINE_COUNT=%r; defaulting to 1", raw)
        count = 1
    if count <= 1:
        return ("kv_cache",)
    return tuple(f"kv_cache_{i}" for i in range(count))


def _all_tags() -> tuple[str, ...]:
    return ("weights",) + _kv_cache_tags()


def main() -> None:
    ready_file = Path(os.environ.get("GMS_SOCKET_DIR", "/tmp")) / _READY_FILE
    ready_file.unlink(missing_ok=True)

    devices = list_devices()
    tags = _all_tags()
    logger.info("Spawning GMS servers for tags=%s on devices=%s", tags, devices)
    processes = []
    for device in devices:
        for tag in tags:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gpu_memory_service",
                    "--device",
                    str(device),
                    "--tag",
                    tag,
                ]
            )
            logger.info("Started GMS device=%d tag=%s pid=%d", device, tag, proc.pid)
            processes.append(proc)

    def shutdown() -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def terminate(*_args) -> None:
        shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    ready_written = False
    while True:
        if not ready_written:
            sockets_ready = all(
                os.path.exists(get_socket_path(device, tag))
                for device in devices
                for tag in tags
            )
            if sockets_ready:
                ready_file.write_text("ready", encoding="utf-8")
                ready_written = True

        running = False
        for process in processes:
            exit_code = process.poll()
            if exit_code is None:
                running = True
                continue
            shutdown()
            raise SystemExit(exit_code)

        if not running:
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
