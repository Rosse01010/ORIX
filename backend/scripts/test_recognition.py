#!/usr/bin/env python3
"""
test_recognition.py
───────────────────
Smoke test for the ORIX facial recognition pipeline.

Usage:
    python scripts/test_recognition.py [IMAGE_PATH] [--api-url URL]

Examples:
    python scripts/test_recognition.py test_face.jpg
    python scripts/test_recognition.py test_face.jpg --api-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="ORIX recognition smoke test")
    parser.add_argument("image", help="Path to a face image (JPEG/PNG)")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Base URL of the ORIX API")
    args = parser.parse_args()

    base = args.api_url.rstrip("/")

    # 1. Health check
    print("[1/3] Checking API health...")
    r = httpx.get(f"{base}/api/recognition/health", timeout=10)
    r.raise_for_status()
    health = r.json()
    print(f"      Status: {health['status']}")
    print(f"      Model loaded: {health.get('model_loaded', 'N/A')}")
    print(f"      Embedding version: {health.get('embedding_version', 'N/A')}")
    print(f"      Threshold: {health['similarity_threshold']}")

    # 2. Test endpoint
    print(f"\n[2/3] Running recognition test on: {args.image}")
    with open(args.image, "rb") as f:
        t0 = time.monotonic()
        r = httpx.post(
            f"{base}/api/recognition/test",
            files={"file": (args.image, f, "image/jpeg")},
            timeout=30,
        )
        wall_ms = (time.monotonic() - t0) * 1000

    r.raise_for_status()
    result = r.json()

    print(f"      Faces detected: {len(result['faces'])}")
    print(f"      Server latency: {result['latency_ms']} ms")
    print(f"      Wall time:      {wall_ms:.0f} ms")
    print(f"      Model version:  {result['model_version']}")

    for i, (face, match) in enumerate(zip(result["faces"], result["matches"])):
        print(f"\n      Face #{i}:")
        print(f"        BBox:       {face['bbox']}")
        print(f"        Quality:    {face['quality']}")
        print(f"        Yaw/Pitch:  {face['yaw']}° / {face['pitch']}°")
        print(f"        Emb dim:    {face['embedding_dim']}")
        print(f"        Emb time:   {face['embedding_ms']} ms")
        print(f"        Match:      {match['name']} (sim={match['similarity']}, tier={match['tier']})")

    # 3. Recognize endpoint
    print(f"\n[3/3] Running /recognize endpoint...")
    with open(args.image, "rb") as f:
        r = httpx.post(
            f"{base}/api/recognition/recognize",
            files={"file": (args.image, f, "image/jpeg")},
            data={"camera": "test_script"},
            timeout=30,
        )
    r.raise_for_status()
    rec = r.json()
    print(f"      Camera:    {rec['camera']}")
    print(f"      Timestamp: {rec['timestamp']}")
    print(f"      Faces:     {len(rec['bboxes'])}")
    for b in rec["bboxes"]:
        print(f"        - {b['name']} (conf={b['confidence']}, tier={b['confidence_tier']}, angle={b['angle']})")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
