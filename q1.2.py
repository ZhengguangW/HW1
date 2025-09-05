import torch
import torch.nn as nn
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm

torch.manual_seed(42)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]

def profile_attention(seq_len, embed_dim=64, num_heads=4):
    B = 1  # batch size
    device = "cpu"
    x = torch.randn(B, seq_len, embed_dim).to(device)
    model = SelfAttention(embed_dim, num_heads).to(device)

    # FLOPs
    with torch.no_grad():
        try:
            flops = FlopCountAnalysis(model, x).total()
        except Exception as e:
            print(f"FLOP estimation failed at L={seq_len}: {e}")
            flops = float('nan')

    # Wall-clock
    start_time = time.time()
    out = model(x)
    elapsed = time.time() - start_time

    return flops, elapsed

def run_experiment(trials=5):
    seq_lengths = [10, 100, 1000, 10000]
    results = {"seq": [], "flops": [], "time": [], "flops_err": [], "time_err": []}

    for L in tqdm(seq_lengths, desc="Running on CPU"):
        flops_list, time_list = [], []
        for _ in range(trials):
            try:
                flops, t = profile_attention(L)
                flops_list.append(flops)
                time_list.append(t)
            except RuntimeError as e:
                print(f"Error at L={L}: {e}")
                break

        if flops_list:
            results["seq"].append(L)
            results["flops"].append(np.mean(flops_list))
            results["time"].append(np.mean(time_list))
            results["flops_err"].append(np.std(flops_list)/np.sqrt(trials))
            results["time_err"].append(np.std(time_list)/np.sqrt(trials))

    return results

def plot_results(results):
    for metric in ["flops", "time"]:
        plt.figure(figsize=(8,5))
        sns.lineplot(
            x=results["seq"], y=results[metric],
            label="CPU",
            err_style="band",
            ci=None,
            linestyle='-', marker='o'
        )
        plt.errorbar(
            results["seq"], results[metric],
            yerr=results[f"{metric}_err"], fmt='none', capsize=4, color='gray'
        )
        plt.xscale("log")
        plt.yscale("log" if metric == "flops" else "linear")
        plt.xlabel("Sequence Length (log scale)")
        plt.ylabel({
            "flops": "FLOPs",
            "time": "Wall-clock Time (s)"
        }[metric])
        plt.title(f"Self-Attention {metric.title()} vs Sequence Length (CPU)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cpu_{metric}_vs_length.png")
        plt.show()

if __name__ == "__main__":
    results = run_experiment(trials=5)
    plot_results(results)
