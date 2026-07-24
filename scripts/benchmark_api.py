"""
Phase 10 Benchmark & Load Testing Script.
Measures API throughput (Requests Per Second) and latency percentiles (P50, P95, P99)
under high concurrent HTTP load.
"""

import os
import sys
import time
import concurrent.futures
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app


SAMPLE_PREDICT_PAYLOAD = {
    "company_name": "Maruti",
    "model_name": "Swift",
    "variant_name": "VXi",
    "manufacture_year": 2021,
    "km_driven": 25000,
    "fuel_type": "Petrol",
    "transmission": "Manual",
    "owner_count": 1,
    "city": "Mumbai"
}


def send_single_request(client, endpoint="/predict"):
    start = time.time()
    if endpoint == "/predict":
        response = client.post("/predict", json=SAMPLE_PREDICT_PAYLOAD)
    else:
        response = client.get("/health")
    duration_ms = (time.time() - start) * 1000.0
    return response.status_code, duration_ms


def run_benchmark(num_requests: int = 100, max_workers: int = 10):
    print("=" * 80)
    print("PHASE 10: FASTAPI HIGH-CONCURRENCY PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Total Requests: {num_requests} | Concurrent Workers: {max_workers}")

    client = TestClient(app)
    latencies = []
    success_count = 0
    failure_count = 0

    start_total = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_single_request, client, "/predict") for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            status_code, latency = future.result()
            if status_code == 200:
                success_count += 1
                latencies.append(latency)
            else:
                failure_count += 1

    total_duration_sec = time.time() - start_total
    rps = num_requests / total_duration_sec

    p50 = np.percentile(latencies, 50) if latencies else 0
    p95 = np.percentile(latencies, 95) if latencies else 0
    p99 = np.percentile(latencies, 99) if latencies else 0
    mean_lat = np.mean(latencies) if latencies else 0

    print("\n" + "=" * 80)
    print("BENCHMARK PERFORMANCE METRICS SUMMARY")
    print("=" * 80)
    print(f"  |-- Total Execution Time : {total_duration_sec:.2f} seconds")
    print(f"  |-- Throughput (RPS)    : {rps:.2f} requests/sec")
    print(f"  |-- Successful Requests : {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
    print(f"  |-- Failed Requests     : {failure_count}")
    print(f"  |-- Mean Latency        : {mean_lat:.2f} ms")
    print(f"  |-- P50 Median Latency  : {p50:.2f} ms")
    print(f"  |-- P95 Latency         : {p95:.2f} ms")
    print(f"  \\-- P99 Tail Latency    : {p99:.2f} ms")
    print("=" * 80)

    return {
        "rps": round(rps, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "success_rate": round(success_count / num_requests * 100, 2)
    }


if __name__ == "__main__":
    run_benchmark(num_requests=50, max_workers=5)
