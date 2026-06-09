"""
SPP New Experiments — Spectral Gap and Context Incoherence
================================================================

Two experiments to validate the SPP stability theory:

  Exp A: Spectral Gap σ₁/σ₂ verification  (from existing data)
  Exp B: Context Incoherence η^(l)        (needs model)
"""

import os, json, math, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

SAVE_DIR = "./spp_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def model_slug(name: str) -> str:
    """'Qwen/Qwen2.5-7B' -> 'qwen2.5-7b'"""
    return name.split("/")[-1].lower().replace(" ", "-")

# ============================================================
#  PLOTTING HELPERS
# ============================================================

def _style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

# ============================================================
#  EXP A: SPECTRAL GAP VERIFICATION
# ============================================================

def exp_a_spectral_gap(json_path: str = None):
    """
    Visualize σ₁/σ₂ (rho) and the raw σ₁, σ₂ per layer.
    Shows whether σ₁ is decisively dominant.
    """
    _style()

    if json_path is None:
        json_path = os.path.join(SAVE_DIR, "exp4_fixed_cumulative_gain.json")
    with open(json_path) as f:
        data = json.load(f)

    layers = np.array(data["layers"])
    sigma1 = np.array(data["sigma_1"])
    sigma2 = np.array(data["sigma_2"])
    rho = np.array(data["rho"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: σ₁ vs σ₂
    ax = axes[0]
    ax.plot(layers, sigma1, "o-", color="#d62728", markersize=4, label=r"$\sigma_1$")
    ax.plot(layers, sigma2, "s-", color="#1f77b4", markersize=4, label=r"$\sigma_2$")
    ax.fill_between(layers, sigma2, sigma1, alpha=0.15, color="#d62728")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Singular Value")
    ax.set_title(r"$\sigma_1$ vs $\sigma_2$ per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Spectral gap ratio ρ = σ₁/σ₂
    ax = axes[1]
    ax.bar(layers, rho, color="#2ca02c", alpha=0.7, edgecolor="#2ca02c")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label=r"$\rho=1$ (no gap)")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\rho = \sigma_1 / \sigma_2$")
    ax.set_title("Spectral Gap Ratio per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Fractional energy of top-1
    frac = np.array(data["frac_energy_top1"])
    ax = axes[2]
    ax.plot(layers, frac * 100, "D-", color="#9467bd", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\sigma_1^2 / \sum \sigma_i^2$ (%)")
    ax.set_title("Top-1 Spectral Energy Fraction")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp A: Spectral Gap Verification (Assumption 4.5)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "expA_spectral_gap.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[Exp A] Saved → {out}")

    print(f"\n  Mean ρ = {np.mean(rho):.4f},  Min ρ = {np.min(rho):.4f},  Max ρ = {np.max(rho):.4f}")
    print(f"  Mean frac_top1 = {np.mean(frac)*100:.1f}%")
    plt.close(fig)
    return rho


# ============================================================
#  MODEL & DATA LOADING  (shared by Exp B)
# ============================================================

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HiddenStateCollector:
    def __init__(self, model, layer_prefix):
        self.states = OrderedDict()
        self.hooks = []
        for name, module in model.named_modules():
            if not name.startswith(layer_prefix + "."):
                continue
            suffix = name[len(layer_prefix) + 1:]
            if suffix.isdigit():
                idx = int(suffix)
                self.hooks.append(
                    module.register_forward_hook(self._make_hook(idx))
                )
        print(f"[Collector] Hooks on {len(self.hooks)} layers")

    def _make_hook(self, idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.states[idx] = h.detach()
        return fn

    def clear(self):
        self.states.clear()

    def get_last(self) -> Dict[int, torch.Tensor]:
        return dict(self.states)

    def remove(self):
        for h in self.hooks:
            h.remove()


def infer_layer_prefix(model):
    from collections import Counter
    cands = Counter()
    for name, _ in model.named_modules():
        parts = name.split(".")
        for i in range(len(parts)):
            if parts[i].isdigit():
                cands[".".join(parts[:i])] += 1
                break
    if not cands:
        return "model.layers"
    prefix = cands.most_common(1)[0][0]
    direct = set()
    for name, _ in model.named_modules():
        if name.startswith(prefix + "."):
            s = name[len(prefix) + 1:].split(".")[0]
            if s.isdigit():
                direct.add(int(s))
    print(f"  Layer prefix: '{prefix}'  ({len(direct)} layers: {min(direct)}..{max(direct)})")
    return prefix


def collect_per_sample_hidden(
    model, tokenizer, texts: List[str], collector: HiddenStateCollector,
    rep_token: int = -1, max_samples: int = 100, desc: str = "Collecting",
) -> Dict[int, np.ndarray]:
    """Collect hidden states at `rep_token` for each text. Returns {layer: [n, dim]}."""
    layer_vecs = defaultdict(list)
    n = min(max_samples, len(texts))

    for i in tqdm(range(n), desc=desc):
        collector.clear()
        txt = texts[i].strip()
        if not txt:
            continue
        inputs = tokenizer(txt, return_tensors="pt", padding=False,
                           truncation=True, max_length=512)
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        if inputs["input_ids"].numel() == 0:
            continue
        with torch.no_grad():
            model(**inputs)
        for layer_idx, h in collector.get_last().items():
            vec = h[0, rep_token, :].float().cpu().numpy()
            layer_vecs[layer_idx].append(vec)

    return {l: np.stack(v) for l, v in layer_vecs.items()}


# ============================================================
#  EXP B: CONTEXT INCOHERENCE  η^(l)
# ============================================================

def exp_b_context_incoherence(
    pos_states: Dict[int, np.ndarray],
    neg_states: Dict[int, np.ndarray],
):
    """
    Measure context incoherence by comparing:
      Red:  (1/M) Σ_k ||f_k^(l)||            — avg per-sample fluctuation
      Blue: sqrt( ||Γ^(l)||_op )              — collective fluctuation magnitude

    where f_k^(l) = (d_k^(l+1) - d_k^(l)) - mean(d_k^(l+1) - d_k^(l))
    is the per-sample deviation from the population-mean transition.

    Also computes the V3 diffuseness ratio from the paper's Appendix A.6.
    """
    _style()

    layers = sorted(pos_states.keys())
    M = min(pos_states[layers[0]].shape[0], neg_states[layers[0]].shape[0])

    # Build per-sample concept difference vectors
    D = {}  # D[l] = [M, dim]
    for l in layers:
        D[l] = pos_states[l][:M] - neg_states[l][:M]

    # SVD of D^(l) → v_1^(l)
    V1 = {}
    for l in layers:
        _, S, Vt = np.linalg.svd(D[l] - D[l].mean(0), full_matrices=False)
        V1[l] = Vt[0]

    avg_individual = []   # red line
    collective_mag = []   # blue line
    diffuse_ratio = []    # V3 metric
    transition_layers = []

    for i in range(len(layers) - 1):
        l, l_next = layers[i], layers[i + 1]
        transition_layers.append(l)

        # Per-sample transition: Δd_k = d_k^(l+1) - d_k^(l)
        delta = D[l_next] - D[l]       # [M, dim]
        delta_mean = delta.mean(axis=0) # [dim]

        # Per-sample fluctuation (remove population mean)
        F = delta - delta_mean          # [M, dim]

        # Red: average individual fluctuation magnitude
        norms = np.linalg.norm(F, axis=1)   # [M]
        avg_ind = float(norms.mean())
        avg_individual.append(avg_ind)

        # Blue: collective fluctuation = sqrt(operator norm of Γ)
        # Γ = (1/M) Σ f_k f_k^T   →  ||Γ||_op = σ₁²(F) / M
        try:
            from scipy.sparse.linalg import svds
            k = min(5, M - 1, F.shape[1] - 1)
            _, S_F, _ = svds(F.astype(np.float64), k=k)
            sigma1_F = float(S_F.max())
        except Exception:
            _, S_F, _ = np.linalg.svd(F, full_matrices=False)
            sigma1_F = float(S_F[0])

        gamma_op = sigma1_F ** 2 / M
        collective_mag.append(float(np.sqrt(gamma_op)))

        # V3: diffuseness ratio
        # Project out v_1 component from residual
        v1 = V1[l]
        R = D[l_next] - D[l]   # = delta
        proj = R @ v1[:, None] @ v1[None, :]  # [M, dim]
        R_perp = R - proj
        frob = np.linalg.norm(R_perp, "fro")
        try:
            _, S_perp, _ = svds(R_perp.astype(np.float64), k=1)
            s1_perp = float(S_perp[0])
        except:
            _, S_perp, _ = np.linalg.svd(R_perp, full_matrices=False)
            s1_perp = float(S_perp[0])
        denom = frob / np.sqrt(min(M, F.shape[1]))
        ratio = s1_perp / denom if denom > 1e-10 else 0
        diffuse_ratio.append(float(ratio))

    # ---- Plot ----
    t = np.array(transition_layers)
    avg_ind = np.array(avg_individual)
    coll = np.array(collective_mag)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(t, avg_ind, "o-", color="#d62728", markersize=4, linewidth=1.5,
            label=r"$\frac{1}{M}\sum_k \|f_k^{(l)}\|$ (avg individual)")
    ax.plot(t, coll, "s-", color="#1f77b4", markersize=4, linewidth=1.5,
            label=r"$\sqrt{\|\Gamma^{(l)}\|_{op}}$ (collective)")
    ax.fill_between(t, coll, avg_ind, alpha=0.12, color="#d62728")
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel("Perturbation Magnitude")
    ax.set_title("Context Incoherence: Individual vs Collective Noise")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, diffuse_ratio, "D-", color="#ff7f0e", markersize=4, linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6,
               label="Ratio = 1 (perfectly diffuse)")
    ax.set_xlabel("Layer $l$")
    ax.set_ylabel(r"$\sigma_1(R_\perp) \,/\, (\|R_\perp\|_F / \sqrt{\min(M,d)})$")
    ax.set_title("V3 Diffuseness Ratio (≈1 → incoherent)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(r"Exp B: Context Incoherence $\eta^{(l)}$ (Assumption 4.7)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "expB_context_incoherence.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[Exp B] Saved → {out}")

    ratio_mean_coll = float(np.mean(avg_ind / (coll + 1e-10)))
    print(f"\n  Mean(individual/collective) = {ratio_mean_coll:.2f}x")
    print(f"  Mean diffuseness ratio = {np.mean(diffuse_ratio):.4f}")
    plt.close(fig)

    results = {
        "transition_layers": [int(x) for x in transition_layers],
        "avg_individual_norm": [float(x) for x in avg_individual],
        "collective_sqrt_gamma": [float(x) for x in collective_mag],
        "diffuseness_ratio_V3": [float(x) for x in diffuse_ratio],
    }
    with open(os.path.join(SAVE_DIR, "expB_context_incoherence.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
#  EXP A': ENHANCED SPECTRAL GAP (needs per-sample hidden states)
# ============================================================

def exp_a_enhanced_spectral_gap(
    pos_states: Dict[int, np.ndarray],
    neg_states: Dict[int, np.ndarray],
    n_shuffles: int = 20,
):
    """
    Three-pronged enhanced spectral gap analysis:

      Panel 1: ρ on the *difference matrix* D^(l) vs the combined data matrix.
               D^(l) should have much larger ρ since concept is dominant.

      Panel 2: Cumulative amplification  ∏ρ^(l)  across layers.
               Even ρ≈1.2 per layer compounds to 1.2^L ≫ 1.

      Panel 3: Random-label baseline.  Shuffle pos/neg assignments,
               recompute D^(l), and show ρ collapses → proves gap is
               concept-driven, not a generic property.
    """
    _style()

    layers = sorted(pos_states.keys())
    M = min(pos_states[layers[0]].shape[0], neg_states[layers[0]].shape[0])

    # ---- 1. Compute ρ on difference matrix D^(l) ----
    rho_diff = []     # ρ from D^(l) = pos - neg
    rho_combined = [] # ρ from combined centered data (original Exp A)
    sigma1_diff = []
    sigma2_diff = []
    sigma_all_diff = {}
    concept_alpha = []  # |<concept_dir, v_1(D)>|^2

    for l in layers:
        P = pos_states[l][:M]
        N = neg_states[l][:M]

        # D^(l): concept difference matrix (paper Definition 4.3)
        D = P - N  # [M, dim], each row = d_k = h+_k - h-_k
        _, S_d, Vt_d = np.linalg.svd(D, full_matrices=False)
        rho_d = float(S_d[0] / S_d[1]) if len(S_d) > 1 and S_d[1] > 1e-10 else float('inf')
        rho_diff.append(rho_d)
        sigma1_diff.append(float(S_d[0]))
        sigma2_diff.append(float(S_d[1]) if len(S_d) > 1 else 0)
        sigma_all_diff[l] = S_d[:min(10, len(S_d))].astype(float).tolist()

        # Concept direction: mean(pos) - mean(neg), normalized
        concept_dir = P.mean(0) - N.mean(0)
        cn = np.linalg.norm(concept_dir)
        if cn > 1e-10:
            concept_dir /= cn
        alpha = float(np.dot(concept_dir, Vt_d[0]) ** 2)
        concept_alpha.append(alpha)

        # Combined data matrix (as in original Exp A)
        X = np.concatenate([P, N], axis=0)
        X_c = X - X.mean(0)
        _, S_c, _ = np.linalg.svd(X_c, full_matrices=False)
        rho_c = float(S_c[0] / S_c[1]) if len(S_c) > 1 and S_c[1] > 1e-10 else float('inf')
        rho_combined.append(rho_c)

    rho_diff = np.array(rho_diff)
    rho_combined = np.array(rho_combined)

    # ---- 2. Cumulative amplification ----
    cumul_rho = np.cumprod(rho_diff)

    # ---- 3. Random baseline: shuffle pos/neg labels ----
    rng = np.random.RandomState(42)
    rho_shuffled_all = []  # [n_shuffles, n_layers]

    for s in range(n_shuffles):
        rho_this = []
        for l in layers:
            P = pos_states[l][:M]
            N = neg_states[l][:M]
            # Stack all, then randomly re-assign to "pos" / "neg"
            all_h = np.concatenate([P, N], axis=0)  # [2M, dim]
            idx = rng.permutation(2 * M)
            P_shuf = all_h[idx[:M]]
            N_shuf = all_h[idx[M:]]
            D_shuf = P_shuf - N_shuf
            _, S_s, _ = np.linalg.svd(D_shuf, full_matrices=False)
            r = float(S_s[0] / S_s[1]) if len(S_s) > 1 and S_s[1] > 1e-10 else 1.0
            rho_this.append(r)
        rho_shuffled_all.append(rho_this)

    rho_shuf_mean = np.mean(rho_shuffled_all, axis=0)
    rho_shuf_std = np.std(rho_shuffled_all, axis=0)

    cumul_rho_shuf = np.cumprod(rho_shuf_mean)

    # ---- PLOT ----
    layers_arr = np.array(layers)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) ρ comparison: D^(l) vs combined vs shuffled
    ax = axes[0, 0]
    ax.plot(layers_arr, rho_diff, "o-", color="#d62728", markersize=5,
            linewidth=1.8, label=r"$\rho_{D}$ (difference matrix $D^{(l)}$)")
    ax.plot(layers_arr, rho_combined, "s-", color="#1f77b4", markersize=4,
            linewidth=1.2, alpha=0.7, label=r"$\rho_{data}$ (combined data)")
    ax.fill_between(layers_arr,
                     rho_shuf_mean - rho_shuf_std,
                     rho_shuf_mean + rho_shuf_std,
                     alpha=0.2, color="gray")
    ax.plot(layers_arr, rho_shuf_mean, "x--", color="gray", markersize=4,
            linewidth=1, label=r"$\rho_{shuffled}$ (random labels)")
    ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\rho = \sigma_1 / \sigma_2$")
    ax.set_title(r"Spectral Gap: $D^{(l)}$ vs Combined vs Shuffled")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (0,1) Concept alignment: does v_1(D) = concept direction?
    ax = axes[0, 1]
    ax.plot(layers_arr, concept_alpha, "D-", color="#9467bd", markersize=5,
            linewidth=1.8)
    ax.fill_between(layers_arr, 0, concept_alpha, alpha=0.15, color="#9467bd")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\alpha = |\langle \bar{\lambda}_W,\, v_1^{(l)} \rangle|^2$")
    ax.set_title(r"Concept Alignment: is $v_1$ the concept direction?")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (1,0) Cumulative amplification ∏ρ
    ax = axes[1, 0]
    ax.semilogy(layers_arr, cumul_rho, "D-", color="#d62728", markersize=5,
                linewidth=2, label=r"Concept: $\prod \rho_{D}^{(l)}$")
    ax.semilogy(layers_arr, cumul_rho_shuf, "x--", color="gray", markersize=4,
                linewidth=1.2, label=r"Shuffled: $\prod \rho_{shuffled}^{(l)}$")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\prod_{l'=0}^{l} \rho^{(l')}$ (log scale)")
    ax.set_title(r"Cumulative Amplification $\prod \rho$  (power iteration effect)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate final value
    final_val = cumul_rho[-1]
    ax.annotate(f"{final_val:.0f}×",
                xy=(layers_arr[-1], final_val),
                xytext=(-40, 10), textcoords="offset points",
                fontsize=11, fontweight="bold", color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

    # (1,1) Top singular value spectrum at selected layers
    ax = axes[1, 1]
    selected = [0, 7, 14, 21, layers[-1]]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(selected)))
    for idx, l in enumerate(selected):
        sv = np.array(sigma_all_diff[l])
        sv_norm = sv / sv[0]  # normalize to σ₁=1
        ax.plot(range(1, len(sv_norm) + 1), sv_norm, "o-", color=colors[idx],
                markersize=4, linewidth=1.3, label=f"Layer {l}")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Singular Value Index $k$")
    ax.set_ylabel(r"$\sigma_k / \sigma_1$")
    ax.set_title(r"Normalized Spectrum of $D^{(l)}$ at Selected Layers")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle(r"Exp A$'$: Enhanced Spectral Gap — Difference Matrix, Cumulative, Baseline",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "expA_enhanced_spectral_gap.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[Exp A'] Saved → {out}")

    # Summary
    print(f"\n  ρ on D^(l):        mean={np.mean(rho_diff):.3f},  range=[{np.min(rho_diff):.3f}, {np.max(rho_diff):.3f}]")
    print(f"  ρ on combined:     mean={np.mean(rho_combined):.3f}")
    print(f"  ρ shuffled:        mean={np.mean(rho_shuf_mean):.3f}")
    print(f"  Cumulative ∏ρ_D:   {cumul_rho[-1]:.1f}×  (after {len(layers)} layers)")
    print(f"  Cumulative ∏ρ_shuf:{cumul_rho_shuf[-1]:.1f}×")
    print(f"  Amplification advantage: {cumul_rho[-1]/cumul_rho_shuf[-1]:.1f}×")

    plt.close(fig)

    results = {
        "layers": [int(l) for l in layers],
        "rho_diff_matrix": [float(x) for x in rho_diff],
        "rho_combined_data": [float(x) for x in rho_combined],
        "rho_shuffled_mean": [float(x) for x in rho_shuf_mean],
        "rho_shuffled_std": [float(x) for x in rho_shuf_std],
        "concept_alpha": [float(x) for x in concept_alpha],
        "sigma1_diff": [float(x) for x in sigma1_diff],
        "sigma2_diff": [float(x) for x in sigma2_diff],
        "cumul_rho_diff": [float(x) for x in cumul_rho],
        "cumul_rho_shuffled": [float(x) for x in cumul_rho_shuf],
        "sigma_spectrum_selected": {str(l): sigma_all_diff[l] for l in selected},
    }
    with open(os.path.join(SAVE_DIR, "expA_enhanced_spectral_gap.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================
#  MAIN
# ============================================================

def main():
    global SAVE_DIR

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--only-plots", action="store_true",
                        help="Only run Exp A from existing JSON (no GPU)")
    args = parser.parse_args()

    slug = model_slug(args.model)
    SAVE_DIR = f"./spp_results/{slug}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  SPP Experiments — {args.model}")
    print(f"  Output dir: {SAVE_DIR}")
    print("=" * 60)

    if args.only_plots:
        exp_a_spectral_gap()
        return

    # ---- Load model ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        device_map="auto", trust_remote_code=True)
    model.eval()
    prefix = infer_layer_prefix(model)
    collector = HiddenStateCollector(model, prefix)

    # ---- Load honesty dataset ----
    import json as _json
    with open("file/coco_honesty_dataset.json") as f:
        raw = _json.load(f)

    pos_texts, neg_texts = [], []
    for item in raw["data"]:
        pos_t = item.get("honest", "")
        neg_t = item.get("untruthful", "")
        if pos_t and neg_t:
            pos_texts.append(pos_t.replace("USER: <image> ", "").strip())
            neg_texts.append(neg_t.replace("USER: <image> ", "").strip())
    print(f"  Dataset: {len(pos_texts)} pairs")

    # ---- Collect hidden states ----
    print("\n[1/2] Collecting hidden states...")
    pos_hs = collect_per_sample_hidden(
        model, tokenizer, pos_texts, collector,
        max_samples=args.max_samples, desc="pos")
    neg_hs = collect_per_sample_hidden(
        model, tokenizer, neg_texts, collector,
        max_samples=args.max_samples, desc="neg")

    # ---- Exp A': Enhanced Spectral Gap ----
    print("\n[2/2] Exp A': Enhanced Spectral Gap (Difference Matrix)")
    exp_a_enhanced_spectral_gap(pos_hs, neg_hs, n_shuffles=20)

    # ---- Exp B: Context Incoherence ----
    print("\n[2/2] Exp B: Context Incoherence")
    exp_b_context_incoherence(pos_hs, neg_hs)

    collector.remove()
    print("\n" + "=" * 60)
    print(f"  ALL DONE — {args.model}")
    print(f"  Results: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()