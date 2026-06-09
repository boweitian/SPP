"""
Multi-Model SPP Experiment Plots
Generates publication-quality combined figures across 4 LLMs.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

MODELS = {
    "qwen2.5-7b":              ("Qwen 2.5-7B",  "#2563EB", "-",  "o"),
    "llama-3.1-8b-instruct":   ("Llama 3.1-8B",  "#DC2626", "--", "s"),
    "mistral-7b-instruct-v0.3":("Mistral-7B",    "#059669", "-.", "^"),
    "wizard-vicuna-30b-uncensored":           ("Wizard-Vicuna-30B",    "#D97706", ":",  "D"),
}
BASE = "./spp_results"
OUT  = os.path.join(BASE, "multi_model")
os.makedirs(OUT, exist_ok=True)

def load(slug, name):
    with open(os.path.join(BASE, slug, name)) as f:
        return json.load(f)

def norm_layers(layers):
    """Normalize layer indices to [0, 1] for cross-model comparison."""
    arr = np.array(layers, dtype=float)
    return arr / arr.max() if arr.max() > 0 else arr

# ─────────────────────────── Figure 1: Spectral Dominance ───────────────────
def fig1_spectral_dominance():
    sns.set_theme(style="whitegrid", font_scale=1.8)
    plt.rcParams.update({
        "font.family": ["Ubuntu", "DejaVu Sans"],
        "font.weight": "medium",
        "axes.titleweight": "bold",
        "axes.labelweight": "medium",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    model_handles = []
    shuf_handles  = []
    bar_names, bar_means, bar_stds, bar_colors = [], [], [], []

    for slug, (label, color, ls, marker) in MODELS.items():
        d = load(slug, "expA_enhanced_spectral_gap.json")
        x = norm_layers(d["layers"])

        # (a) Spectral Gap — drop last data point
        x_a    = x[:-1]
        rho_a  = np.array(d["rho_diff_matrix"])[:-1]
        shuf_a = np.array(d["rho_shuffled_mean"])[:-1]
        mk_a   = dict(marker=marker, markersize=4, markevery=max(1, len(x_a)//10))
        axes[0].plot(x_a, rho_a,  color=color, ls=ls,  lw=2.0, **mk_a)
        axes[0].plot(x_a, shuf_a, color=color, ls=":", lw=1.2, alpha=0.65)

        # (b) Concept-Spectral Alignment
        mk = dict(marker=marker, markersize=4, markevery=max(1, len(x)//10))
        axes[1].plot(x, d["concept_alpha"], color=color, ls=ls, lw=2.0, **mk)

        # (d) Bar data: mean ρ per model
        rho_all = np.array(d["rho_diff_matrix"])
        bar_names.append(label)
        bar_means.append(rho_all.mean())
        bar_stds.append(rho_all.std())
        bar_colors.append(color)

        model_handles.append(Line2D([0], [0], color=color, ls=ls, lw=2.0,
                                    marker=marker, markersize=5, label=label))
        shuf_handles.append(Line2D([0], [0], color=color, ls=":", lw=1.5,
                                   alpha=0.85, label=f"{label} (shuffled)"))

    # (d) Mean Spectral Gap bar chart
    bars = axes[2].bar(bar_names, bar_means, yerr=bar_stds, capsize=5,
                       color=bar_colors, alpha=0.78, edgecolor="black", lw=0.7,
                       error_kw=dict(elinewidth=1.2, ecolor="#444444"))
    for bar, m in zip(bars, bar_means):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"{m:.2f}", ha="center", va="bottom", fontsize=17, fontweight="bold")
    axes[2].set_ylabel("Mean ρ (σ₁/σ₂)", fontsize=18)
    axes[2].set_title("(c) Mean Spectral Gap", fontsize=19, fontweight="bold", pad=12)
    axes[2].tick_params(axis="x", labelsize=15, rotation=30)
    plt.setp(axes[2].get_xticklabels(), ha="right", rotation_mode="anchor")

    axes[0].set_ylabel(r"$\rho = \sigma_1/\sigma_2$", fontsize=18)
    axes[0].set_title("(a) Spectral Gap of D(l)", fontsize=19, fontweight="bold", pad=12)
    axes[0].set_xlabel("Normalized Layer Depth", fontsize=17)

    axes[1].set_ylabel(r"$|\cos(v_1^{(l)},\,v_1^{(l+1)})|$", fontsize=18)
    axes[1].set_title("(b) Concept-Spectral Alignment", fontsize=19, fontweight="bold", pad=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Normalized Layer Depth", fontsize=17)

    axes[2].set_ylabel(r"Mean $\rho$", fontsize=18)

    for ax in axes:
        ax.tick_params(labelsize=15)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")

    # Two-row shared legend below subplots (a) and (b) only; bar chart has its own x-labels
    all_handles = model_handles + shuf_handles
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=len(MODELS),
        fontsize=18,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#bbbbbb",
        bbox_to_anchor=(0.5, -0.1),
        handlelength=2.6,
        columnspacing=1.8,
        handletextpad=0.7,
    )

    fig.tight_layout()
    path = os.path.join(OUT, "fig1_spectral_dominance.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    sns.reset_defaults()
    print(f"  Saved {path}")


# ─────────────────── Figure 2: Context Incoherence ──────────────────────────
def fig2_context_incoherence():
    sns.set_theme(style="whitegrid", font_scale=1.8)
    plt.rcParams.update({
        "font.family": ["Ubuntu", "DejaVu Sans"],
        "font.weight": "medium",
        "axes.titleweight": "bold",
        "axes.labelweight": "medium",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    model_handles = []

    for slug, (label, color, ls, marker) in MODELS.items():
        d = load(slug, "expB_context_incoherence.json")
        x = norm_layers(d["transition_layers"])
        ind = np.array(d["avg_individual_norm"])
        col = np.array(d["collective_sqrt_gamma"])
        mk = dict(marker=marker, markersize=4, markevery=max(1, len(x)//10))

        axes[0].plot(x, ind, color=color, ls=ls, lw=2.0, label=label, **mk)
        axes[1].plot(x, col, color=color, ls=ls, lw=2.0, label=label, **mk)

        model_handles.append(Line2D([0], [0], color=color, ls=ls, lw=2.0,
                                    marker=marker, markersize=5, label=label))

    # (c) Bar: mean individual/collective ratio
    names, ratios, bar_colors = [], [], []
    for slug, (label, color, _, _) in MODELS.items():
        d = load(slug, "expB_context_incoherence.json")
        ind = np.array(d["avg_individual_norm"])
        col = np.array(d["collective_sqrt_gamma"])
        ratio = (ind / np.maximum(col, 1e-10)).mean()
        names.append(label)
        ratios.append(ratio)
        bar_colors.append(color)

    bars = axes[2].bar(names, ratios, color=bar_colors, alpha=0.78,
                       edgecolor="black", lw=0.7)
    for bar, r in zip(bars, ratios):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f"{r:.1f}×", ha="center", va="bottom", fontsize=17, fontweight="bold")
    axes[2].axhline(1.0, color="grey", ls="--", lw=0.8)
    axes[2].set_ylim(0, 3.5)
    axes[2].set_ylabel("Ratio (Individual / Collective)", fontsize=18)
    axes[2].set_title("(c) Destructive Interference Ratio", fontsize=19, fontweight="bold", pad=12)
    axes[2].tick_params(axis="x", labelsize=15, rotation=30)
    plt.setp(axes[2].get_xticklabels(), ha="right", rotation_mode="anchor")

    axes[0].set_ylabel(r"$(1/M)\,\sum_k \|f_k\|_2$", fontsize=18)
    axes[0].set_title("(a) Individual Perturbation Norm", fontsize=19, fontweight="bold", pad=12)
    axes[0].set_xlabel("Normalized Layer Depth", fontsize=17)
    axes[0].legend(handles=model_handles, fontsize=18, frameon=True,
                   fancybox=True, framealpha=0.92, edgecolor="#bbbbbb",
                   handlelength=2.6, handletextpad=0.7)

    axes[1].set_ylabel(r"$\sqrt{\|\Gamma^{(l)}\|_{\mathrm{op}}}$", fontsize=18)
    axes[1].set_title("(b) Collective Perturbation Norm", fontsize=19, fontweight="bold", pad=12)
    axes[1].set_xlabel("Normalized Layer Depth", fontsize=17)
    axes[1].legend(handles=model_handles, fontsize=18, frameon=True,
                   fancybox=True, framealpha=0.92, edgecolor="#bbbbbb",
                   handlelength=2.6, handletextpad=0.7)

    for ax in axes:
        ax.tick_params(labelsize=15)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("black")

    fig.tight_layout()
    path = os.path.join(OUT, "fig2_context_incoherence.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    sns.reset_defaults()
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Model SPP Plots")
    print("=" * 60)
    fig1_spectral_dominance()
    fig2_context_incoherence()
    print("\n  All figures saved to:", OUT)