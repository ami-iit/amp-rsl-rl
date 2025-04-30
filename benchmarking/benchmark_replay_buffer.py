#!/usr/bin/env python3
# benchmark_replay_buffer.py
# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
import torch
from amp_rsl_rl.storage.replay_buffer import ReplayBuffer

# =============================================
# CONFIGURATION
# =============================================
device_str = "cuda" if torch.cuda.is_available() else "cpu"
obs_dim = 60  # dimension of each state vector
buffer_size = 200_000  # capacity of the circular buffer
insert_batch = 4096  # how many transitions per insert() call
num_inserts = 50  # how many insert() calls to benchmark
mini_batch_size = 1024  # size of each sampled mini-batch
num_mini_batches = 20  # how many mini-batches to sample


def main():
    device = torch.device(device_str)
    print(f"\n[ReplayBuffer Benchmark] Device: {device}\n")

    # 1) Initialize buffer
    buf = ReplayBuffer(obs_dim, buffer_size, device)

    # 2) Prepare dummy data
    dummy_states = torch.randn(insert_batch, obs_dim, device=device)
    dummy_next = torch.randn(insert_batch, obs_dim, device=device)

    # Warm up (GPU kernels, caches, etc.)
    for _ in range(5):
        buf.insert(dummy_states, dummy_next)
        for _ in buf.feed_forward_generator(1, mini_batch_size):
            pass

    # 3) Benchmark insert()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_inserts):
        buf.insert(dummy_states, dummy_next)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_inserted = insert_batch * num_inserts
    insert_rate = total_inserted / (t1 - t0)
    print(
        f"[insert] Inserted {total_inserted} samples in {(t1 - t0)*1e3:.1f} ms → "
        f"{insert_rate:,.0f} samples/s"
    )

    # Ensure there's enough data to sample
    assert len(buf) >= mini_batch_size * num_mini_batches, (
        f"Need at least {mini_batch_size * num_mini_batches} samples, "
        f"but buffer has only {len(buf)}"
    )

    # 4) Benchmark sampling
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    sampled = 0
    for states, next_states in buf.feed_forward_generator(
        num_mini_batches, mini_batch_size
    ):
        sampled += states.size(0)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    sample_rate = sampled / (t3 - t2)
    print(
        f"[sample]  Sampled {sampled} samples in {(t3 - t2)*1e3:.1f} ms → "
        f"{sample_rate:,.0f} samples/s"
    )

    # 5) Combined insert + sample
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    ops = 0
    for _ in range(num_inserts):
        buf.insert(dummy_states, dummy_next)
        for states, _ in buf.feed_forward_generator(1, mini_batch_size):
            ops += states.size(0)
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    combined_rate = (total_inserted + ops) / (t5 - t4)
    print(
        f"[combined] insert+sample of {total_inserted + ops} ops in {(t5 - t4)*1e3:.1f} ms → "
        f"{combined_rate:,.0f} ops/s\n"
    )


if __name__ == "__main__":
    main()
