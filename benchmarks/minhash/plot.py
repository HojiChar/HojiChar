"""Render chart-only PNG and SVG outputs using Matplotlib.

Install matplotlib, then run: python plot.py
The bar chart uses the measured English 2,000-character workload.
The line chart uses the original English document-length measurements.
All inputs are medians of five rounds with three excerpts per condition.
See the full benchmark bundle for methodology and implementation details.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = {
    "hojichar_018": ("HojiChar 0.18.0", "#007F68", "o", "-"),
    "hojichar_0173": ("HojiChar 0.17.3", "#276BA0", "s", "-"),
    "datatrove": ("Hugging Face Datatrove 0.10.0", "#8C4FAD", "D", "--"),
    "datasketch": ("datasketch 1.6.5", "#CC6B22", "^", "--"),
}


def save(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(ROOT / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def throughput():
    rows = json.loads((ROOT / "throughput_data.json").read_text())
    values = [next(r["documents_per_second"] for r in rows if r["method"] == m) for m in METHODS]
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    bars = ax.barh(
        range(len(METHODS)), values, height=0.58, color=[v[1] for v in METHODS.values()]
    )
    ax.set_yticks(range(len(METHODS)), [v[0] for v in METHODS.values()])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(0, max(values) * 1.16)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Documents / second", labelpad=12)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.1f}",
            va="center",
            fontsize=12,
        )
    save(fig, "throughput_2000_chars")


def latency():
    rows = json.loads((ROOT / "latency_data.json").read_text())
    fig, ax = plt.subplots(figsize=(9.5, 6.5), layout="constrained")
    for method, (label, color, marker, linestyle) in METHODS.items():
        data = sorted([r for r in rows if r["method"] == method], key=lambda r: r["characters"])
        ax.plot(
            [r["characters"] for r in data],
            [r["median_ms"] for r in data],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.4,
            markersize=6,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([100, 300, 1000, 3000, 10000], labels=["100", "300", "1,000", "3,000", "10,000"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Document length (characters)", labelpad=12)
    ax.set_ylabel("Time / document (ms)", labelpad=12)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    save(fig, "latency_by_length")


def datatrove_speedup():
    rows = json.loads((ROOT / "throughput_data.json").read_text())
    rates = {row["method"]: row["documents_per_second"] for row in rows}
    values = [rates[method] / rates["datatrove"] for method in METHODS]
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    bars = ax.barh(
        range(len(METHODS)), values, height=0.58, color=[value[1] for value in METHODS.values()]
    )
    ax.set_yticks(range(len(METHODS)), [value[0] for value in METHODS.values()])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("Speed relative to Hugging Face Datatrove 0.10.0 (1x)", labelpad=12)
    ax.axvline(1, color="#717982", linestyle=":", linewidth=1.2)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}x",
            va="center",
            fontsize=12,
        )
    save(fig, "speedup_vs_datatrove")


def main():
    global plt, ticker
    from importlib import import_module

    matplotlib = import_module("matplotlib")
    matplotlib.use("Agg")
    plt = import_module("matplotlib.pyplot")
    ticker = import_module("matplotlib.ticker")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    throughput()
    latency()
    datatrove_speedup()


if __name__ == "__main__":
    main()
