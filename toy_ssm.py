import math
import time
import random
import torch
import torch.nn as nn

# ----------------------------------------
# Global Settings
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test over several sequence lengths (feel free to adjust)
SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384]
# Number of repeated queries (for "final" or "random" modes) to simulate multi-turn inference on the same prompt
NUM_QUERIES = 10

# Choose query mode from: "final", "random", "sequential"
QUERY_MODE = "random"  # change as desired

INPUT_DIM = 256
HIDDEN_DIM = 512

PRINT_GPU_MEM = True

# ----------------------------------------
# Model: A Simple 1-Layer SSM
# ----------------------------------------
class ToySSM(nn.Module):
    """
    A simple one-layer SSM-like module.
    The update rule is: h <- A * h + B * x_t + bias.
    Parameter shapes:
      A: (hidden_dim, hidden_dim)
      B: (hidden_dim, input_dim)
      bias: (hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, h, x_t):
        return self.A @ h + self.B @ x_t + self.bias


# ----------------------------------------
# Naive Forward Pass (Fixed)
# ----------------------------------------
def forward_naive(ssm, x_seq):  # FIXED: removed 'query_indices'
    """
    Naively process the entire input sequence x_seq from t=0 to len(x_seq)-1,
    returning the final hidden state.
    """
    h = torch.zeros(ssm.hidden_dim, device=x_seq.device)
    for t in range(x_seq.size(0)):
        h = ssm(h, x_seq[t])
    return h


# ----------------------------------------
# Hierarchical Caching for Inference
# ----------------------------------------
class HierarchicalSSMCache:
    """
    Builds a checkpoint cache for a given sequence by computing
    and storing hidden states every block_size steps (approx sqrt(N)).
    """
    def __init__(self, ssm, x_seq, block_size=None):
        self.ssm = ssm
        self.x_seq = x_seq
        self.N = x_seq.size(0)
        if block_size is None:
            block_size = int(math.sqrt(self.N)) if self.N > 0 else 1
        self.block_size = max(1, block_size)
        self.checkpoints = {}
        self.build_checkpoints()

    def build_checkpoints(self):
        """
        Build checkpoints at multiples of block_size.
        E.g., for N=16384 and block_size=128, store 0,128,256,…,16384.
        """
        h = torch.zeros(self.ssm.hidden_dim, device=self.x_seq.device)
        self.checkpoints[0] = h.clone()
        for cp in range(self.block_size, self.N + 1, self.block_size):
            h_temp = self.checkpoints[cp - self.block_size].clone()
            for j in range(cp - self.block_size, cp):
                h_temp = self.ssm(h_temp, self.x_seq[j])
            self.checkpoints[cp] = h_temp.clone()
        if self.N not in self.checkpoints:
            last_cp = max(self.checkpoints.keys())
            h_temp = self.checkpoints[last_cp].clone()
            for j in range(last_cp, self.N):
                h_temp = self.ssm(h_temp, self.x_seq[j])
            self.checkpoints[self.N] = h_temp.clone()

    def get_hidden_state(self, i):
        """
        Return the hidden state after processing x_seq[:i].
        If i is a checkpoint, return it directly.
        Otherwise, compute from the previous checkpoint to i.
        """
        if i in self.checkpoints:
            return self.checkpoints[i]
        j = (i // self.block_size) * self.block_size
        h_temp = self.checkpoints[j].clone()
        for idx in range(j, i):
            h_temp = self.ssm(h_temp, self.x_seq[idx])
        return h_temp


def forward_cached(ssm, x_seq, query_indices, cache=None):
    """
    Use hierarchical caching to answer queries at indices in query_indices.
    If no cache is provided, build it first (measure build time).
    Returns a dictionary of results and timing info.
    """
    if cache is None:
        reset_peak_memory()
        start_build = time.time()
        cache = HierarchicalSSMCache(ssm, x_seq)
        build_time = time.time() - start_build
    else:
        build_time = 0.0

    results = {}
    query_times = {}
    for i in query_indices:
        reset_peak_memory()
        start_query = time.time()
        results[i] = cache.get_hidden_state(i)
        query_times[i] = time.time() - start_query
    total_query_time = sum(query_times.values())
    return results, build_time, total_query_time, query_times


# ----------------------------------------
# Utilities for Timing and GPU Memory Usage
# ----------------------------------------
def reset_peak_memory():
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def peak_memory_mb():
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0

def print_gpu_mem(prefix=""):
    if PRINT_GPU_MEM and device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e6
        reserved = torch.cuda.memory_reserved(device) / 1e6
        print(f"{prefix} allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")


# ----------------------------------------
# Repeated Inference Test
# ----------------------------------------
def run_repeated_queries_test(seq_len, input_dim, hidden_dim, num_queries, query_mode="final"):
    """
    For a given sequence length, build the model and random input sequence,
    then run repeated inference queries in one of three modes:
      - "final": All queries at index = seq_len.
      - "random": Each query is a random index in [1, seq_len].
      - "sequential": Query indices from 1..seq_len (or evenly-spaced if num_queries < seq_len).
    Compare:
      1) Naive method: run a forward pass each time from scratch.
      2) Caching method: build partial checkpoints and do partial recompute.
    """
    # Build model and input sequence
    ssm = ToySSM(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    x_seq = torch.randn(seq_len, input_dim, device=device)
    
    # Determine query indices based on mode
    if query_mode == "final":
        query_indices = [seq_len] * num_queries
    elif query_mode == "random":
        query_indices = [random.randint(1, seq_len) for _ in range(num_queries)]
    elif query_mode == "sequential":
        if num_queries >= seq_len:
            query_indices = list(range(1, seq_len + 1))
        else:
            step = seq_len // num_queries
            query_indices = [i for i in range(step, seq_len + 1, step)]
    else:
        raise ValueError("Invalid query_mode. Choose from 'final', 'random', or 'sequential'.")

    # ---------- Naive repeated queries ----------
    naive_times = {}
    for i in query_indices:
        reset_peak_memory()
        start = time.time()
        _ = forward_naive(ssm, x_seq[:i])  # FIXED: no 'query_indices' argument
        naive_times[i] = time.time() - start
    total_naive = sum(naive_times.values())
    avg_naive = total_naive / len(naive_times)

    # ---------- Caching repeated queries ----------
    reset_peak_memory()
    start_build = time.time()
    cache = HierarchicalSSMCache(ssm, x_seq)
    build_time = time.time() - start_build
    
    # Use the same query_indices for forward_cached
    _, _, total_caching_query, query_times_detail = forward_cached(ssm, x_seq, query_indices, cache)
    avg_caching_query = total_caching_query / len(query_indices)
    total_caching = build_time + total_caching_query

    # Verify correctness on final hidden state
    h_naive_final = forward_naive(ssm, x_seq)  # entire sequence
    h_cached_final = cache.get_hidden_state(seq_len)
    diff = float((h_naive_final - h_cached_final).abs().sum().item())

    stats = {
        'seq_len': seq_len,
        'query_mode': query_mode,
        'num_queries': len(query_indices),
        'naive_total_time_sec': total_naive,
        'naive_avg_time_sec': avg_naive,
        'naive_query_times': naive_times,
        'caching_build_time_sec': build_time,
        'caching_total_query_time_sec': total_caching_query,
        'caching_avg_query_time_sec': avg_caching_query,
        'caching_total_time_sec': total_caching,
        'final_state_diff': diff,
        'block_size': cache.block_size,
        'num_checkpoints': len(cache.checkpoints),
        'query_indices': query_indices,
        'detailed_cached_query_times': query_times_detail
    }
    return stats


# ----------------------------------------
# Main
# ----------------------------------------
def main():
    print(f"Device: {device}")
    print(f"Input Dim = {INPUT_DIM}, Hidden Dim = {HIDDEN_DIM}")
    print(f"Repeated query test with {NUM_QUERIES} queries per sequence using query mode: {QUERY_MODE}\n")
    if device.type == 'cuda':
        print_gpu_mem("Initial GPU mem usage:")
    
    modes = ["final", "random", "sequential"]
    overall_results = {mode: [] for mode in modes}
    
    for mode in modes:
        print("\n========== Query Mode:", mode, "==========\n")
        for length in SEQ_LENGTHS:
            print("=" * 40)
            print(f"Testing Sequence Length = {length}")
            stats = run_repeated_queries_test(length, INPUT_DIM, HIDDEN_DIM, NUM_QUERIES, query_mode=mode)
            overall_results[mode].append(stats)
            
            print_gpu_mem("After run_repeated_queries_test:")
            print(f"Query Mode: {mode}")
            print(f"[NAIVE] Total Time={stats['naive_total_time_sec']:.4f}s, Avg Time={stats['naive_avg_time_sec']:.4f}s")
            print(f"[CACHE] Checkpoint Build Time={stats['caching_build_time_sec']:.4f}s, "
                  f"Total Query Time={stats['caching_total_query_time_sec']:.4f}s, "
                  f"Avg Query Time={stats['caching_avg_query_time_sec']:.6f}s")
            print(f"[CACHE] Total Cached Time (Build + Queries)={stats['caching_total_time_sec']:.4f}s")
            print(f"Block size (checkpoints every): {stats['block_size']}, "
                  f"Num Checkpoints stored: {stats['num_checkpoints']}")
            print(f"Query Indices: {stats['query_indices']}")
            print(f"Detailed Cached Query Times: {stats['detailed_cached_query_times']}")
            print(f"Final state difference (should be 0): {stats['final_state_diff']:.6f}")
            print("=" * 40)
    
    print("\n=== Overall Repeated Inference Comparison Summary ===")
    for mode in modes:
        print(f"\n--- Query Mode: {mode} ---")
        for s in overall_results[mode]:
            print(f"Length={s['seq_len']} | "
                  f"Naive: Total={s['naive_total_time_sec']:.4f}s, Avg={s['naive_avg_time_sec']:.4f}s | "
                  f"Cache: Total={s['caching_total_time_sec']:.4f}s, Avg Query={s['caching_avg_query_time_sec']:.6f}s | "
                  f"Final Diff={s['final_state_diff']:.6f}")
    
    print("\nDone. This extended benchmark tests a variety of query scenarios," 
          " highlighting where hierarchical caching can offer advantages.")
    

if __name__ == "__main__":
    main()
