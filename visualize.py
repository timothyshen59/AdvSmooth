import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    present = [v for v in order if v in values]
    extras = [v for v in values if v not in order]
    return present + extras


def _subset(df, dataset=None, epsilon=None):
    """Filter to a single dataset and/or epsilon so plots don't mix incomparable rows."""
    out = df
    if dataset is not None:
        out = out[out["dataset"] == dataset]
    if epsilon is not None:
        out = out[np.isclose(out["epsilon"].astype(float), float(epsilon))]
    return out


def _class_labels(conf_cols, dataset):
    idxs = [int(c.split("_")[-1]) for c in conf_cols]
    if dataset == "cifar10":
        return [f"{i}: {CIFAR10_CLASSES[i]}" if i < len(CIFAR10_CLASSES) else str(i) for i in idxs]
    return [str(i) for i in idxs]


def confidence_heatmap(df, out_dir, dataset, epsilon):
    d = _subset(df, dataset=dataset, epsilon=epsilon)
    if d.empty:
        print(f"[confidence_heatmap] no rows for dataset={dataset} epsilon={epsilon}")
        return
    piv = d.pivot_table(index="config", columns="eval_attack",
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
    plt.title(f"Mean confidence: config x attack ({dataset}, eps={epsilon:g})")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                         color="white" if mat[i, j] < 0.5 else "black", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"confidence_heatmap_{dataset}_eps{epsilon:g}.png")
    plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)


def per_class_confidence_heatmap(df, out_dir, dataset, epsilon):
    d = _subset(df, dataset=dataset, epsilon=epsilon).copy()
    if d.empty:
        print(f"[per_class_confidence_heatmap] no rows for dataset={dataset} epsilon={epsilon}")
        return
    conf_cols = sorted([c for c in d.columns if c.startswith("conf_class_")],
                       key=lambda c: int(c.split("_")[-1]))
    d["row"] = d["config"] + " | " + d["eval_attack"]
    d["ci"] = d["config"].map({c: i for i, c in enumerate(CONFIG_ORDER)}).fillna(99)
    d["ai"] = d["eval_attack"].map({a: i for i, a in enumerate(ATTACK_ORDER)}).fillna(99)
    d = d.sort_values(["ci", "ai"])

    mat = d[conf_cols].values.astype(float)
    labels = d["row"].tolist()
    class_labels = _class_labels(conf_cols, dataset)

    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(max(10, 1.35 * n_cols + 3), 0.62 * n_rows + 2.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="confidence (true label)")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(class_labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_xlabel("class")
    ax.set_title(f"Per-class confidence: config | attack ({dataset}, eps={epsilon:g})")
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.5 else "black", fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, f"per_class_confidence_heatmap_{dataset}_eps{epsilon:g}.png")
    plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)


def accuracy_lines(df, out_dir, dataset, epsilon):
    d = _subset(df, dataset=dataset, epsilon=epsilon)
    if d.empty:
        return
    piv = d.pivot_table(index="config", columns="eval_attack", values="accuracy", aggfunc="mean")
    piv = piv.reindex(index=_ordered(piv.index, CONFIG_ORDER),
                      columns=_ordered(piv.columns, ATTACK_ORDER))
    plt.figure(figsize=(7, 4.5))
    x = range(len(piv.index))
    for attack in piv.columns:
        plt.plot(x, piv[attack].values, marker="o", label=f"attack={attack}")
    plt.xticks(list(x), piv.index)
    plt.xlabel("config"); plt.ylabel("accuracy"); plt.ylim(0, 1)
    plt.title(f"Accuracy across configs ({dataset}, eps={epsilon:g})")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(out_dir, f"accuracy_lines_{dataset}_eps{epsilon:g}.png")
    plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)


def accuracy_vs_epsilon(df, out_dir, dataset):
    """x = epsilon, one line per (config, attack): color = config, linestyle = attack."""
    d = _subset(df, dataset=dataset)
    if d.empty:
        return
    configs = _ordered(list(d["config"].unique()), CONFIG_ORDER)
    attacks = _ordered(list(d["eval_attack"].unique()), ATTACK_ORDER)
    color_map = {c: col for c, col in zip(CONFIG_ORDER, plt.cm.tab10.colors)}
    style_map = {"none": "-", "pgd_l2": "--", "eot": ":"}

    plt.figure(figsize=(8, 5))
    for cfg in configs:
        for atk in attacks:
            sub = d[(d["config"] == cfg) & (d["eval_attack"] == atk)]
            if sub.empty:
                continue
            sub = sub.groupby("epsilon", as_index=False)["accuracy"].mean().sort_values("epsilon")
            plt.plot(sub["epsilon"], sub["accuracy"], marker="o",
                     color=color_map.get(cfg, "gray"),
                     linestyle=style_map.get(atk, "-"),
                     label=f"{cfg} / {atk}")
    plt.xlabel("epsilon (L2 attack budget)")
    plt.ylabel("accuracy"); plt.ylim(0, 1)
    plt.title(f"Accuracy vs epsilon ({dataset})")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(out_dir, f"accuracy_vs_epsilon_{dataset}.png")
    plt.savefig(path, dpi=130); plt.close()
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs.csv")
    ap.add_argument("--out-dir", default="figs")
    ap.add_argument("--dataset", default="mnist", help="dataset to plot (heatmaps need a single one)")
    ap.add_argument("--epsilon", type=float, default=1.0, help="epsilon for the heatmaps / accuracy_lines")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load(args.csv)
    confidence_heatmap(df, args.out_dir, args.dataset, args.epsilon)
    per_class_confidence_heatmap(df, args.out_dir, args.dataset, args.epsilon)
    accuracy_lines(df, args.out_dir, args.dataset, args.epsilon)
    accuracy_vs_epsilon(df, args.out_dir, args.dataset)   # spans all epsilons


if __name__ == "__main__":
    main()