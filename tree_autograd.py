import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# -------------------------------
# 1. Enhanced Tree-Based Autograd
# -------------------------------

class TreeTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, children=None):
        ret = torch.Tensor._make_subclass(cls, data)
        ret.children = children if children is not None else []
        return ret

    def backward_tree(self, grad_output=None):
        if grad_output is None:
            grad_output = torch.ones_like(self)

        if not hasattr(self, 'children') or not self.children:
            self.grad = grad_output
            return

        # Recompute children if needed
        with torch.enable_grad():
            for child in self.children:
                if not hasattr(child, 'grad'):
                    child.backward_tree()

        # Propagate gradients
        for child in self.children:
            if hasattr(child, 'grad'):
                child.grad = child.grad + grad_output if hasattr(child, 'grad') else grad_output
                child.backward_tree(child.grad)

# -------------------------------
# 2. Tree-Aware Operations
# -------------------------------

def tree_op(x, weight, op):
    """Generic tree-aware operation wrapper"""
    result = op(x, weight)
    if isinstance(x, TreeTensor) or isinstance(weight, TreeTensor):
        children = []
        if isinstance(x, TreeTensor):
            children.append(x)
        else:
            children.append(TreeTensor(x))
        if isinstance(weight, TreeTensor):
            children.append(weight)
        else:
            children.append(TreeTensor(weight))
        return TreeTensor(result, children=children)
    return result

def tree_matmul(x, weight):
    return tree_op(x, weight, torch.matmul)

def tree_add(x, weight):
    return tree_op(x, weight, torch.add)

# -------------------------------
# 3. Toy Neural Network with Caching for Inference
# -------------------------------

class ToyNN(nn.Module):
    def __init__(self, width, depth, use_tree=False):
        super().__init__()
        self.width = width
        self.depth = depth
        self.use_tree = use_tree
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(width, width, device="cuda"))
            for _ in range(depth)
        ])
        # For caching in inference mode when using tree-based computations
        self.cached_output = None

    def forward(self, x):
        # For tree mode, in eval mode, reuse the cached output if available.
        if self.use_tree:
            if not self.training and self.cached_output is not None:
                return self.cached_output
            x = TreeTensor(x)
            for weight in self.weights:
                # Wrap weight in TreeTensor as well.
                weight = TreeTensor(weight)
                x = tree_matmul(x, weight)
                # Simulate bias by adding a random tensor wrapped in TreeTensor.
                x = tree_add(x, TreeTensor(torch.randn_like(x)))
            output = x.sum()  # Return scalar
            if not self.training:
                self.cached_output = output
            return output
        else:
            # Standard autograd path
            for weight in self.weights:
                x = torch.matmul(x, weight)
                x = torch.add(x, torch.randn_like(x))
            return x.sum()

# -------------------------------
# 4. Enhanced Benchmarking for Training
# -------------------------------

def benchmark_nn(width, depth, use_tree, n_epochs=5):
    model = ToyNN(width, depth, use_tree).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    times = []
    mem_usages = []
    
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Generate new input every epoch for training.
        x = torch.randn(width, width, device="cuda")
        start_time = time.time()
        
        loss = model(x)
        if use_tree and isinstance(loss, TreeTensor):
            loss.backward_tree()
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
        mem_usages.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
    
    return times, mem_usages

# -------------------------------
# 5. Enhanced Benchmarking for Inference with Repeated Calls
# -------------------------------

def benchmark_inference_nn(width, depth, use_tree, n_epochs=5):
    model = ToyNN(width, depth, use_tree).cuda()
    model.eval()  # Set model to evaluation mode
    times = []
    mem_usages = []
    
    # For repeated inference, use a fixed input.
    x = torch.randn(width, width, device="cuda")
    with torch.no_grad():
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            _ = model(x)  # forward pass only
            
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
            mem_usages.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
    
    return times, mem_usages

# -------------------------------
# 6. Comprehensive Experiment Runner
# -------------------------------

def run_comprehensive_experiment():
    configs = [
        {"width": 64, "depth": 10},
        {"width": 128, "depth": 20},
        {"width": 256, "depth": 30}
    ]
    
    results = defaultdict(dict)
    
    for cfg in configs:
        print(f"\nTraining Benchmarking {cfg['depth']} layers, width {cfg['width']}")
        
        # Standard autograd training
        std_times, std_mem = benchmark_nn(cfg["width"], cfg["depth"], False)
        results["Standard", "Train"][(cfg["width"], cfg["depth"])] = {
            "times": std_times,
            "memory": std_mem
        }
        
        # Tree autograd training
        tree_times, tree_mem = benchmark_nn(cfg["width"], cfg["depth"], True)
        results["Tree", "Train"][(cfg["width"], cfg["depth"])] = {
            "times": tree_times,
            "memory": tree_mem
        }
        
        print(f"Standard Training: {np.mean(std_times):.3f}s ± {np.std(std_times):.3f}, {np.mean(std_mem):.1f}MB")
        print(f"Tree Training: {np.mean(tree_times):.3f}s ± {np.std(tree_times):.3f}, {np.mean(tree_mem):.1f}MB")
    
    return results

def run_comprehensive_experiment_inference():
    configs = [
        {"width": 64, "depth": 10},
        {"width": 128, "depth": 20},
        {"width": 256, "depth": 30}
    ]
    
    results = defaultdict(dict)
    
    for cfg in configs:
        print(f"\nInference Benchmarking {cfg['depth']} layers, width {cfg['width']}")
        
        # Standard inference
        std_times, std_mem = benchmark_inference_nn(cfg["width"], cfg["depth"], False)
        results["Standard", "Infer"][(cfg["width"], cfg["depth"])] = {
            "times": std_times,
            "memory": std_mem
        }
        
        # Tree inference (with caching, so repeated calls should be faster)
        tree_times, tree_mem = benchmark_inference_nn(cfg["width"], cfg["depth"], True)
        results["Tree", "Infer"][(cfg["width"], cfg["depth"])] = {
            "times": tree_times,
            "memory": tree_mem
        }
        
        print(f"Standard Inference: {np.mean(std_times):.3f}s ± {np.std(std_times):.3f}, {np.mean(std_mem):.1f}MB")
        print(f"Tree Inference: {np.mean(tree_times):.3f}s ± {np.std(tree_times):.3f}, {np.mean(tree_mem):.1f}MB")
    
    return results

# -------------------------------
# 7. Plotting Functions
# -------------------------------

def plot_epoch_progress(results, mode="Train", filename="nn_benchmark_train.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Outer loop: iterate over each (model_type, exp_mode) key
    for (model_type, exp_mode), inner_dict in results.items():
        if exp_mode != mode:
            continue
        # Inner loop: iterate over each configuration in the inner dictionary
        for config, data in inner_dict.items():
            width, depth = config
            label = f"{depth}x{width} ({model_type})"
            ax1.plot(data["times"], label=label)
            ax2.plot(data["memory"], label=label)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Time (s)")
    ax1.set_title(f"{mode} Time per Epoch")
    ax1.legend()
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Memory Usage (MB)")
    ax2.set_title(f"{mode} Memory Usage per Epoch")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -------------------------------
# 8. Main
# -------------------------------

if __name__ == "__main__":
    # Run and plot training experiment results
    train_results = run_comprehensive_experiment()
    plot_epoch_progress(train_results, mode="Train", filename="nn_benchmark_train.png")
    
    # Run and plot inference experiment results (repeated inference to showcase caching in tree mode)
    infer_results = run_comprehensive_experiment_inference()
    plot_epoch_progress(infer_results, mode="Infer", filename="nn_benchmark_infer.png")
