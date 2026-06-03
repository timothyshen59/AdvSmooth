"""
Append one run per row to a CSV with a fixed, flat schema so you can query it
(by seed / attack / defense) and plot easily with pandas.

Columns:
  seed, dataset, epochs,
  randomized_smoothing, pgd_l2_defense   (defense flags: "true"/"false")
  adv_attack_type                        ("pgd_l2" / "eot" / "none")
  eval_attack                            ("none" / "pgd_l2" / "eot"; "none" = clean run)
  epsilon, sigma,
  accuracy, mean_confidence,
  cert_smoothed_accuracy, cert_abstention_rate, cert_mean_radius,
  cert_radius, cert_certified_accuracy   (blank for non-smoothing runs)
  conf_class_0 ... conf_class_{C-1}      (mean P(true label) per class)
"""
import csv
import os

_BASE = ["seed", "dataset", "epochs",
         "randomized_smoothing", "pgd_l2_defense", "adv_attack_type",
         "eval_attack", "epsilon", "sigma",
         "accuracy", "mean_confidence",
         "cert_smoothed_accuracy", "cert_abstention_rate", "cert_mean_radius",
         "cert_radius", "cert_certified_accuracy"]


def csv_fieldnames(num_classes):
    return _BASE + [f"conf_class_{c}" for c in range(num_classes)]


def append_run_csv(path, record, num_classes):
    """Append `record` as one row; writes the header if the file is new/empty.
    Missing keys are written blank, so clean runs and certified runs share a schema."""
    if not path:
        return
    fieldnames = csv_fieldnames(num_classes)
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    row = {k: record.get(k, "") for k in fieldnames}     # unknown keys dropped, missing -> blank
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)