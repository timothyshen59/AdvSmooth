import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# canonical orderings so plots are consistent run to run
CONFIG_ORDER = ["standard", "AT", "RS", "AT+RS"]
ATTACK_ORDER = ["none", "pgd_l2", "eot"]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def config_label(row):
    rs = str(row["randomized_smoothing"]).lower() == "true"
    at = str(row["pgd_l2_defense"]).lower() == "true"
    if rs and at:
        return "AT+RS"
    if at:
        return "AT"
    if rs:
        return "RS"
    return "standard"


def load(csv):
    df = pd.read_csv(csv)
    df["config"] = df.apply(config_label, axis=1)
    return df


def _ordered(values, order):
    """Keep only present values, in canonical order, then append any extras."""
    present = [v for v in order if v in values]
    extras = [v for v in values if v not in order]
    return present + extras


def confidence_heatmap(df, out_dir):
    piv = df.pivot_table(index="config", columns="eval_attack",
                         values="mean_confidence", aggfunc="mean")
    piv = piv.reindex(index=_ordered(piv.index, CONFIG_ORDER),
                      columns=_ordered(piv.columns, ATTACK_ORDER))
    mat = piv.values.astype(float)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, label="mean confidence (true label)")
    plt.xticks(range(len(piv.columns)), piv.columns)
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xlabel("eval attack")
    plt.title("Mean confidence: config x attack")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                         color="white" if mat[i, j] < 0.5 else "black", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "confidence_heatmap.png")
    plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)


def per_class_confidence_heatmap(df, out_dir):
    conf_cols = sorted([c for c in df.columns if c.startswith("conf_class_")],
                       key=lambda c: int(c.split("_")[-1]))
    df = df.copy()
    df["row"] = df["config"] + " | " + df["eval_attack"]
    # order rows by config then attack
    df["ci"] = df["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)}).fillna(99)
    df["ai"] = df["eval_attack"].map({a: i for i, a in enumerate(ATTACK_ORDER)}).fillna(99)
    df = df.sort_values(["ci", "ai"])

    mat = df[conf_cols].values.astype(float)
    labels = df["row"].tolist()

    # Map column indices to CIFAR-10 class names
    class_indices = [int(c.split("_")[-1]) for c in conf_cols]
    class_labels = [
        f"{idx}: {CIFAR10_CLASSES[idx]}" if idx < len(CIFAR10_CLASSES) else str(idx)
        for idx in class_indices
    ]

    n_rows, n_cols = mat.shape
    fig_w = max(10, 1.35 * n_cols + 3)
    fig_h = 0.62 * n_rows + 2.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="confidence (true label)")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(class_labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_xlabel("class")
    ax.set_title("Per-class confidence: config | attack")

    # Annotate each cell with the confidence value, centered
    for i in range(mat.shape[0]):
        for j in range(n_cols):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=7, fontweight="normal")

    plt.tight_layout()
    path = os.path.join(out_dir, "per_class_confidence_heatmap.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print("wrote", path)


def accuracy_lines(df, out_dir):
    piv = df.pivot_table(index="config", columns="eval_attack",
                         values="accuracy", aggfunc="mean")
    piv = piv.reindex(index=_ordered(piv.index, CONFIG_ORDER),
                      columns=_ordered(piv.columns, ATTACK_ORDER))

    plt.figure(figsize=(7, 4.5))
    x = range(len(piv.index))
    for attack in piv.columns:
        plt.plot(x, piv[attack].values, marker="o", label=f"attack={attack}")
    plt.xticks(list(x), piv.index)
    plt.xlabel("config")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.title("Accuracy across configs, per attack")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_lines.png")
    plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs.csv")
    ap.add_argument("--out-dir", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load(args.csv)
    confidence_heatmap(df, args.out_dir)
    per_class_confidence_heatmap(df, args.out_dir)
    accuracy_lines(df, args.out_dir)


if __name__ == "__main__":
    main()