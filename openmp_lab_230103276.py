# SECTION 0: LOCAL OPENMP ENVIRONMENT & DETERMINISTIC SEED HARNESS

import os
import math
# Force native OpenMP runtime (libgomp / iomp)
os.environ["NUMBA_THREADING_LAYER"] = "omp"

import time
import hashlib
import json
import numpy as np
import numba as nb

from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id

STUDENT_NAME = "Kurman Akbota Aidynkyzy"
STUDENT_ID = 230103276

assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, \
    "Set a valid numeric STUDENT_ID!"

assert (os.cpu_count() or 1) >= 4, \
    "Local multi-core CPU (>=4 logical cores) required; no Colab!"


# Trigger OpenMP layer initialization and verify backend

@njit(parallel=True)
def _omp_probe():
    acc = 0

    for i in prange(64):
        acc += get_thread_id() + 1

    return acc


_ = _omp_probe()

assert nb.threading_layer() == "omp", \
    f"Expected 'omp' layer, got {nb.threading_layer()}"

rng = np.random.default_rng(STUDENT_ID)


# Student-specific OpenMP parameters

N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024

BINS_T1 = 48 + (STUDENT_ID % 17)

ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000

PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8

ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16

SKEW_MOD = 37 + (STUDENT_ID % 19)

GRID_H = 384 + (STUDENT_ID % 4) * 32

GRID_W = 512 + (STUDENT_ID % 5) * 32

OMEGA_SOR = np.float64(
    1.72 + (STUDENT_ID % 11) * 0.015
)


# Generate deterministic data

data_t1 = rng.integers(
    0, BINS_T1, size=N_T1, dtype=np.int32
)

weights_t1 = rng.uniform(
    0.25, 2.50, size=N_T1
).astype(np.float64)


# Print environment information

print(
    f"Backend: {nb.threading_layer()} | "
    f"Logical Cores: {os.cpu_count()} | "
    f"Student: {STUDENT_ID}"
)
# TASK 1: RACE CONDITION

@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)

    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]

    return hist, w_sums


print("\n=== TASK 1: RACE CONDITION ===")

set_num_threads(1)

hist_ref, wsums_ref = buggy_hist_omp(
    data_t1, weights_t1, BINS_T1
)

print("Expected count:", N_T1)
print("Single-thread count:", np.sum(hist_ref))

set_num_threads(8)

for run in range(10):
    hist, wsums = buggy_hist_omp(
        data_t1, weights_t1, BINS_T1
    )

    lost = N_T1 - np.sum(hist)
    max_diff = np.max(np.abs(hist_ref - hist))
    weight_diff = np.max(np.abs(wsums_ref - wsums))

    print(
        f"Run {run + 1}: "
        f"Lost counts = {lost}, "
        f"Max discrepancy = {max_diff}, "
        f"Weight discrepancy = {weight_diff:.6f}"
    )


# TODO: Implement safe_hist_omp(indices, weights, num_bins).
# Allocate thread-private histograms and worker counters inside that function.
# Verify counts and reproducible weighted sums for 1, 2, 4 and 8 threads.
@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    n_threads = get_num_threads()

    private_hist = np.zeros(
        (n_threads, num_bins), dtype=np.int64
    )

    private_weights = np.zeros(
        (n_threads, num_bins), dtype=np.float64
    )

    worker_hits = np.zeros(
        n_threads, dtype=np.int64
    )
    for i in prange(indices.shape[0]):
        tid = get_thread_id()
        b = indices[i]

        private_hist[tid, b] += 1
        private_weights[tid, b] += weights[i]
        worker_hits[tid] += 1

    hist = np.sum(private_hist, axis=0)
    w_sums = np.sum(private_weights, axis=0)

    return hist, w_sums, worker_hits
print("\n=== TASK 1: SAFE HISTOGRAM ===")

for threads in [1, 2, 4, 8]:
    set_num_threads(threads)

    hist_t1, wsums_t1, hits = safe_hist_omp(
        data_t1, weights_t1, BINS_T1
    )

    print(
        f"Threads: {threads} | "
        f"Total: {np.sum(hist_t1)} | "
        f"Lost: {N_T1 - np.sum(hist_t1)}"
    )

    assert np.array_equal(hist_t1, hist_ref) 
    
    print(
        "Weight difference:",
        np.max(np.abs(wsums_t1 - wsums_ref))
    )

    print(
        "Bit-identical weights:",
        np.array_equal(wsums_t1, wsums_ref)
    )

    print("Worker hits:", hits)

    assert np.sum(hits) == N_T1
    
# TASK 2: FALSE SHARING

@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)

    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)

        for step in range(iters):
            x = (
                x * np.int64(6364136223846793005)
                + np.int64(1442695040888963407)
            ) & np.int64(0x7FFFFFFFFFFFFFFF)

            state[tid] ^= (x >> 17)

    return state


print("\n=== TASK 2: FALSE SHARING ===")

for threads in [1, 2, 4, 8]:
    set_num_threads(threads)

    # Warm-up
    unpadded_state_omp(100, STUDENT_ID)

    start = time.perf_counter()

    state = unpadded_state_omp(
        ITERS_T2, STUDENT_ID
    )

    elapsed = time.perf_counter() - start

    print(f"Threads: {threads} | Time: {elapsed:.6f} s")
# TASK 3: LOAD BALANCING

print("\n=== TASK 3: LOAD BALANCING ===")

work_units = np.array([
    40 + ((r * 7919 + STUDENT_ID) % SKEW_MOD)**2
    for r in range(ROWS_T3)
], dtype=np.int64)

print("Total rows:", ROWS_T3)
print("Total work units:", np.sum(work_units))
print("Minimum workload:", np.min(work_units))
print("Maximum workload:", np.max(work_units))
# TASK 4: RED-BLACK SOR

print("\n=== TASK 4: RED-BLACK SOR ===")

grid = np.zeros(
    (GRID_H, GRID_W),
    dtype=np.float64
)

# Boundary conditions

grid[0, :] = 100.0
grid[-1, :] = -50.0

grid[:, 0] = 25.0 * np.sin(
    np.linspace(0, np.pi, GRID_H)
)

grid[:, -1] = 75.0 * np.cos(
    np.linspace(0, np.pi / 2, GRID_H)
)

print("Grid height:", GRID_H)
print("Grid width:", GRID_W)
print("Omega:", OMEGA_SOR)
print("Iterations:", 600)
# TODO: Implement padded_state_omp and compare with the baseline.
# TODO: Complete Task 3, Task 4, diagnoses and final verification.
