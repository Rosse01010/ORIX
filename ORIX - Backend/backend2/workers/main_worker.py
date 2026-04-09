"""
Orquestador principal de workers.
Levanta los siguientes procesos:
  - 1 proceso por cámara RTSP (camera_worker)
  - 1 proceso GPU para batching + FaceNet (gpu_worker)
  - 1 proceso para DB + Redis Streams (db_worker, usa asyncio)

Maneja señales SIGTERM/SIGINT para cierre limpio.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import signal
import subprocess
import sys
import time
from typing import List

# Asegurar que /app esté en el path al ejecutar como módulo
sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAIN-WORKER] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from app.config import settings
    from workers.camera_worker import run_camera_worker
    from workers.gpu_worker import run_gpu_worker

    rtsp_urls = settings.get_rtsp_list()
    if not rtsp_urls:
        logger.error("Sin URLs RTSP configuradas. Verificar variable RTSP_URLS.")
        sys.exit(1)

    logger.info(f"Iniciando sistema con {len(rtsp_urls)} cámara(s)...")

    # Cola compartida (camera_workers → gpu_worker)
    # maxsize previene desbordamiento si el GPU no da abasto
    face_queue: mp.Queue = mp.Queue(maxsize=500)

    processes: List[mp.Process] = []

    # ── GPU Worker ────────────────────────────────────────────────────────
    gpu_proc = mp.Process(
        target=run_gpu_worker,
        kwargs={
            "face_queue":       face_queue,
            "batch_size":       settings.GPU_BATCH_SIZE,
            "batch_timeout_ms": settings.GPU_BATCH_TIMEOUT_MS,
        },
        name="gpu-worker",
        daemon=False,
    )
    gpu_proc.start()
    processes.append(gpu_proc)
    logger.info(f"GPU Worker iniciado (PID={gpu_proc.pid})")

    # ── Camera Workers ────────────────────────────────────────────────────
    for idx, url in enumerate(rtsp_urls):
        camera_id = f"cam-{idx:02d}"
        proc = mp.Process(
            target=run_camera_worker,
            kwargs={
                "camera_id":  camera_id,
                "rtsp_url":   url,
                "face_queue": face_queue,
                "frame_skip": settings.FRAME_SKIP,
            },
            name=f"camera-{camera_id}",
            daemon=False,
        )
        proc.start()
        processes.append(proc)
        logger.info(f"Camera Worker '{camera_id}' iniciado (PID={proc.pid})")

    # ── DB Worker (asyncio en subprocess separado) ────────────────────────
    db_proc = subprocess.Popen(
        [sys.executable, "-m", "workers.db_worker"],
        cwd="/app",
    )
    logger.info(f"DB Worker iniciado (PID={db_proc.pid})")

    # ── Manejo de señales para cierre limpio ──────────────────────────────
    def shutdown(signum, _frame):
        logger.info(f"Señal {signum} recibida — deteniendo todos los procesos...")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        db_proc.terminate()
        # Esperar terminación ordenada
        for proc in processes:
            proc.join(timeout=10)
        db_proc.wait()
        logger.info("Sistema detenido correctamente")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # ── Monitor de salud ──────────────────────────────────────────────────
    logger.info("Sistema en ejecución. Monitoreando procesos...")
    while True:
        for proc in processes:
            if not proc.is_alive():
                logger.warning(
                    f"⚠ Proceso '{proc.name}' terminó inesperadamente "
                    f"(exitcode={proc.exitcode}). Considerar reinicio."
                )
        if db_proc.poll() is not None:
            logger.warning(
                f"⚠ DB Worker terminó inesperadamente "
                f"(returncode={db_proc.returncode}). Considerar reinicio."
            )
        time.sleep(15)


if __name__ == "__main__":
    # spawn es compatible con CUDA/GPU (fork puede causar deadlocks con CUDA)
    mp.set_start_method("spawn", force=True)
    main()
