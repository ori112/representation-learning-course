"""Build the Part A report (report/reportA.html -> reportA.pdf) from the summary CSVs.

    uv run python report/build_report_a.py

Tables are filled from summaries/*.csv, so no number in a table is typed by hand. They use the
test-split CSVs once those have rows, and the validation CSVs until then (the caption says which).
The PDF is printed with headless Edge, so the page limit can be checked: the script prints the page count.
"""
import csv
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def read(name):
    """Rows of a summary CSV; [] if the file does not exist yet (test runs not done)."""
    path = os.path.join(ROOT, "summaries", name)
    if not os.path.exists(path):
        return []
    with open(path) as file:
        return list(csv.DictReader(file))


def pct(value, digits=1):
    return f"{float(value):.{digits}f}"


def frozen_table():
    """k-NN and probe on frozen features, both datasets, both layers."""
    test = read("partA_test_frozen_summary.csv")
    if test:
        rows = {(r["dataset"], r["model"], r["layer"]): r for r in test}
        tiny = lambda m, l: rows[("tiny_imagenet", m, l)]
        cifar = lambda m, l: rows[("cifar100", m, l)]
        split = "test"
    else:  # validation numbers: Tiny-ImageNet CSV holds fractions, CIFAR CSV holds percents
        tiny_rows = {(r["model"], r["layer"].replace("conv_pooled", "conv")): r for r in read("partA_full_val_summary.csv")}
        cifar_rows = {(r["model"], r["layer"]): r for r in read("cifar100_frozen_val_summary.csv")}
        tiny = lambda m, l: {k: 100 * float(v) for k, v in tiny_rows[(m, l)].items() if k not in ("model", "layer")}
        cifar = lambda m, l: cifar_rows[(m, l)]
        split = "validation"
    body = ""
    for model, label in (("random", "random init"), ("ae", "vanilla AE"), ("dae", "DAE (ours)")):
        for layer, layer_label in (("bottleneck", "bottleneck 512-d"), ("conv", "pooled conv 1024-d")):
            t, c = tiny(model, layer), cifar(model, layer)
            bold = (model, layer) == ("dae", "conv")
            cells = [label, layer_label, pct(t["knn_top1"]), pct(t["probe_top1_mean"]), pct(t["probe_top5_mean"]),
                     pct(c["knn_top1"]), pct(c["probe_top1_mean"]), pct(c["probe_top5_mean"])]
            body += "<tr" + (' class="best"' if bold else "") + ">" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>"
    return split, body


def finetune_table():
    """Fine-tuning at 1/10/100% labels, mean +- std over 3 seeds, top-1."""
    test = read("partA_test_finetune_summary.csv")
    if test:
        data = {(r["dataset"], r["start"], int(r["label_pct"])): r for r in test}
        get = lambda d, s, p: (data[(d, s, p)]["test_top1_mean"], data[(d, s, p)]["test_top1_std"])
        split = "test"
    else:
        tiny = {(r["start"], int(r["label_pct"])): r for r in read("partA_finetune_3seed_summary.csv")}
        cifar = {(r["start"], int(r["label_pct"])): r for r in read("cifar100_finetune_3seed_summary.csv")}
        get = lambda d, s, p: ((tiny if d == "tiny_imagenet" else cifar)[(s, p)]["val_top1_mean"],
                               (tiny if d == "tiny_imagenet" else cifar)[(s, p)]["val_top1_std"])
        split = "validation"
    body = ""
    for start, label in (("scratch", "supervised from scratch"), ("ae", "vanilla AE init"), ("dae", "DAE init (ours)")):
        cells = [label]
        for dataset in ("tiny_imagenet", "cifar100"):
            for fraction in (1, 10, 100):
                mean, std = get(dataset, start, fraction)
                cells.append(f"{float(mean):.1f} ± {float(std):.1f}")
        body += "<tr" + (' class="best"' if start == "dae" else "") + ">" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>"
    return split, body


def search_table():
    rows = [
        ("corruption", "Gaussian σ ∈ [0.1, 0.5] · patch mask (8×8) drop rate ∈ [0.25, 0.75] · salt-and-pepper ∈ [0.05, 0.30]", "mask 0.30"),
        ("latent size d", "{64, 128, 256, 512}", "512"),
        ("learning rate", "{3e-4, 1e-3, 3e-3}", "1e-3"),
        ("weight decay", "{0, 1e-4}", "0"),
    ]
    return "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)


def main():
    with open(os.path.join(ROOT, "report", "reportA.template.html"), encoding="utf-8") as file:
        html = file.read()
    frozen_split, frozen_body = frozen_table()
    ft_split, ft_body = finetune_table()
    for token, value in {"FROZEN_ROWS": frozen_body, "FROZEN_SPLIT": frozen_split, "FT_ROWS": ft_body,
                         "FT_SPLIT": ft_split, "SEARCH_ROWS": search_table()}.items():
        html = html.replace("{{" + token + "}}", value)
    html_path = os.path.join(ROOT, "report", "reportA.html")
    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html)
    pdf_path = os.path.join(ROOT, "report", "reportA.pdf")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
                    "file:///" + html_path.replace("\\", "/")], check=True, capture_output=True)
    with open(pdf_path, "rb") as file:
        pages = len(re.findall(rb"/Type\s*/Page[^s]", file.read()))
    print(f"wrote {pdf_path}: {pages} pages")


if __name__ == "__main__":
    main()
