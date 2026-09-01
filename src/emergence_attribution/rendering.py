"""Dynamic Figure 2--8 renderer over generated run data and node names."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .provenance import sha256_file
from .temporal import load_graph_records


COLOURS = {
    "blue": "#3269A8",
    "red": "#C84A44",
    "green": "#3C8D68",
    "gold": "#C99A2E",
    "grey": "#6D737A",
    "light": "#E8EBEF",
    "dark": "#20262C",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)


def _export(
    figure: plt.Figure,
    output_root: Path,
    stem: str,
    formats: list[str],
    config: dict[str, Any],
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    written = []
    for file_format in formats:
        path = output_root / f"{stem}.{file_format}"
        dpi = (
            int(config.get("tiff_dpi", 600))
            if file_format == "tiff"
            else int(config.get("png_dpi", 300))
        )
        figure.savefig(path, format=file_format, dpi=dpi)
        written.append(path)
    plt.close(figure)
    return written


def _first_baseline_path(run_root: Path, scenario: str) -> Path:
    manifest = json.loads(
        (run_root / "data" / "simulation_manifest.json").read_text(encoding="utf-8")
    )
    records = sorted(
        (
            item
            for item in manifest["task_records"]
            if item["scenario"] == scenario and item["condition"] == "baseline"
        ),
        key=lambda item: int(item["seed"]),
    )
    return run_root / records[0]["raw_path"]


def figure_2(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    with np.load(_first_baseline_path(run_root, "schelling"), allow_pickle=False) as archive:
        grids = archive["state_grid"]
        unhappy = archive["unhappy_count"]
    with np.load(_first_baseline_path(run_root, "deffuant"), allow_pickle=False) as archive:
        opinions = archive["state_opinion"]
        extreme = archive["extreme_agent_count"]
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#F4F4F2", COLOURS["blue"], COLOURS["gold"]])
    axes[0, 0].imshow(grids[0] + 1, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[0, 0].set_title("Schelling: initial state")
    axes[0, 1].imshow(grids[-1] + 1, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[0, 1].set_title("Schelling: final state")
    axes[0, 2].plot(unhappy, color=COLOURS["blue"], lw=1.5)
    axes[0, 2].set(title="Unsatisfied agents", xlabel="Time", ylabel="Count")
    bins = np.linspace(-1, 1, 25)
    axes[1, 0].hist(opinions[0], bins=bins, color=COLOURS["grey"], alpha=0.9)
    axes[1, 0].set(title="Deffuant: initial opinions", xlabel="Opinion", ylabel="Agents")
    axes[1, 1].hist(opinions[-1], bins=bins, color=COLOURS["red"], alpha=0.9)
    axes[1, 1].set(title="Deffuant: final opinions", xlabel="Opinion", ylabel="Agents")
    axes[1, 2].plot(extreme, color=COLOURS["red"], lw=1.5)
    axes[1, 2].set(title="Extreme-opinion agents", xlabel="Time", ylabel="Count")
    for letter, ax in zip("abcdef", axes.ravel()):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_2_simulation_dynamics", formats, config)


def _graph_positions(representation: dict[str, Any]) -> dict[str, tuple[float, float]]:
    scale_x = {"micro": 0.0, "meso": 1.0, "macro": 2.0}
    positions = {}
    for scale in ("micro", "meso", "macro"):
        nodes = [item for item in representation["indicators"] if item["scale"] == scale]
        for index, node in enumerate(sorted(nodes, key=lambda item: (item["branch_id"], item["id"]))):
            positions[node["id"]] = (scale_x[scale], -index)
    return positions


def figure_3(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    results = pd.read_csv(run_root / "analysis" / "main_results.csv")
    graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    alignments = json.loads(
        (run_root / "analysis" / "indicator_alignment.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (run_root / "analysis" / "graph_evaluation.json").read_text(encoding="utf-8")
    )
    scenarios = sorted(results["scenario"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        representation = json.loads(
            (run_root / "representation" / f"{scenario}_representation.json").read_text(encoding="utf-8")
        )
        graph = graphs[(scenario, "full_method")]
        mapping = alignments[scenario]["mapping"]
        correct = {tuple(item) for item in evaluation[f"{scenario}:full_method"]["correct_edges"]}
        positions = _graph_positions(representation)
        network = nx.DiGraph()
        network.add_nodes_from(positions)
        network.add_edges_from((edge.source, edge.target) for edge in graph)
        node_colours = [
            {"micro": "#D9E7F5", "meso": "#F2E5C2", "macro": "#DCEBDD"}[
                next(item["scale"] for item in representation["indicators"] if item["id"] == node)
            ]
            for node in network.nodes
        ]
        nx.draw_networkx_nodes(
            network, positions, ax=axes[row_index, 0], node_size=80, node_color=node_colours, edgecolors=COLOURS["dark"], linewidths=0.4
        )
        for edge in graph:
            aligned_pair = (mapping.get(edge.source), mapping.get(edge.target))
            is_correct = aligned_pair in correct
            nx.draw_networkx_edges(
                network,
                positions,
                edgelist=[(edge.source, edge.target)],
                ax=axes[row_index, 0],
                edge_color=COLOURS["green"] if is_correct else COLOURS["red"],
                width=1.1 if is_correct else 1.4,
                alpha=0.90 if is_correct else 0.88,
                arrowsize=6,
            )
        details = evaluation[f"{scenario}:full_method"]
        axes[row_index, 0].text(
            0.02,
            0.02,
            f"Missed reference relations: {len(details['missed_edges'])}\nUnmatched added relations: {len(details['unmatched_added_edges'])}",
            transform=axes[row_index, 0].transAxes,
            fontsize=6.5,
            va="bottom",
        )
        axes[row_index, 0].set_title(f"{scenario.title()}: retained graph")
        axes[row_index, 0].axis("off")
        subset = results[results["scenario"] == scenario]
        x = np.arange(len(subset))
        axes[row_index, 1].bar(x - 0.18, subset["edge_f1"], width=0.36, color=COLOURS["blue"], label="Edge F1")
        shd_scaled = subset["shd"] / max(float(subset["shd"].max()), 1.0)
        axes[row_index, 1].bar(x + 0.18, shd_scaled, width=0.36, color=COLOURS["red"], label="SHD (scaled)")
        axes[row_index, 1].set_xticks(x, subset["method"].str.replace("_", " "), rotation=25, ha="right")
        axes[row_index, 1].set_ylim(0, 1.05)
        axes[row_index, 1].set_title(f"{scenario.title()}: structural evaluation")
        axes[row_index, 1].legend(frameon=False, ncol=2)
    for letter, ax in zip("abcd", axes.ravel()):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_3_graph_recovery", formats, config)


def figure_4(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    data = pd.read_csv(run_root / "analysis" / "data_efficiency_repeated_subsampling.csv")
    scenarios = sorted(data["scenario"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        subset = data[data["scenario"] == scenario]
        for method, colour in (("full_method", COLOURS["blue"]), ("trajectory_vote", COLOURS["red"])):
            method_data = subset[subset["method"] == method]
            grouped = method_data.groupby("trajectory_count")
            x = np.asarray(sorted(grouped.groups))
            for column_index, metric in enumerate(("edge_f1", "stability")):
                median = grouped[metric].median().reindex(x).to_numpy()
                low = grouped[metric].quantile(0.25).reindex(x).to_numpy()
                high = grouped[metric].quantile(0.75).reindex(x).to_numpy()
                axes[row_index, column_index].plot(x, median, marker="o", ms=3, color=colour, label=method.replace("_", " "))
                axes[row_index, column_index].fill_between(x, low, high, color=colour, alpha=0.16)
                axes[row_index, column_index].set(xlabel="Independent trajectories", ylabel=metric.replace("_", " ").title(), title=f"{scenario.title()}: {metric.replace('_', ' ')}")
                axes[row_index, column_index].set_ylim(0, 1.05)
    for ax in axes.ravel():
        ax.legend(frameon=False)
    for letter, ax in zip("abcd", axes.ravel()):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_4_data_efficiency", formats, config)


def figure_5(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    curves = pd.read_parquet(run_root / "analysis" / "effect_curves.parquet")
    timing = pd.read_csv(run_root / "analysis" / "path_timing_summary.csv")
    selection = json.loads(
        (run_root / "analysis" / "representative_path_selection.json").read_text(encoding="utf-8")
    )
    scenarios = sorted(selection.get("scenarios", {}))
    figure, axes = plt.subplots(max(len(scenarios), 1), 2, figsize=(7.2, 2.4 * max(len(scenarios), 1)), squeeze=False, constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        path_id = selection["scenarios"][scenario].get("path_id")
        if not path_id:
            axes[row_index, 0].text(0.5, 0.5, "No complete ordered path", ha="center", va="center")
            axes[row_index, 1].axis("off")
            continue
        path_rows = timing[(timing["scenario"] == scenario) & (timing["path_id"] == path_id)]
        first = path_rows.iloc[0]
        node_order = [first["source"], first["meso"], first["macro"]]
        for node, colour in zip(node_order, (COLOURS["blue"], COLOURS["gold"], COLOURS["green"])):
            selected_curve = curves[
                (curves["scenario"] == scenario)
                & (curves["parameter"] == first["parameter"])
                & (curves["direction"] == first["direction"])
                & (curves["node_id"] == node)
            ].sort_values("time")
            axes[row_index, 0].plot(selected_curve["time"], selected_curve["mean"], color=colour, label=node)
            axes[row_index, 0].fill_between(selected_curve["time"], selected_curve["ci_low"], selected_curve["ci_high"], color=colour, alpha=0.14)
        axes[row_index, 0].axhline(0, color=COLOURS["dark"], lw=0.6)
        axes[row_index, 0].set(title=f"{scenario.title()}: mean standardised paired response", xlabel="Time", ylabel="Mean effect")
        axes[row_index, 0].legend(frameon=False, fontsize=6)
        ordered = path_rows.sort_values("scale", key=lambda values: values.map({"micro": 0, "meso": 1, "macro": 2}))
        axes[row_index, 1].errorbar(
            np.arange(3),
            ordered["onset_time"],
            yerr=np.vstack([ordered["onset_time"] - ordered["onset_ci_low"], ordered["onset_ci_high"] - ordered["onset_time"]]),
            fmt="o-",
            color=COLOURS["blue"],
            capsize=3,
        )
        axes[row_index, 1].set_xticks(np.arange(3), ["Micro", "Meso", "Macro"])
        axes[row_index, 1].set(title="Response onset and 95% interval", ylabel="Onset time")
    for letter, ax in zip("abcdefghijklmnopqrstuvwxyz", axes.ravel()):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_5_intervention_timing", formats, config)


def figure_6(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    effects = pd.read_parquet(run_root / "analysis" / "paired_effects.parquet")
    scenarios = sorted(effects["scenario"].unique())
    figure, axes = plt.subplots(1, len(scenarios), figsize=(7.2, 3.8), squeeze=False, constrained_layout=True)
    for index, scenario in enumerate(scenarios):
        subset = effects[effects["scenario"] == scenario].copy()
        subset["condition"] = subset["parameter"] + ":" + subset["direction"]
        matrix = subset.pivot(index="node_id", columns="condition", values="cumulative_effect")
        maximum = max(float(np.nanmax(np.abs(matrix.to_numpy()))), 1e-12)
        image = axes[0, index].imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        axes[0, index].set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        axes[0, index].set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=5)
        axes[0, index].set_title(f"{scenario.title()}: full-range effects")
        figure.colorbar(image, ax=axes[0, index], fraction=0.046, pad=0.03, label="Standardised cumulative effect")
        _panel(axes[0, index], chr(ord("a") + index))
    return _export(figure, output_root, "figure_6_effect_matrix", formats, config)


def figure_7(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    timing = pd.read_csv(run_root / "analysis" / "path_timing_summary.csv")
    classes = pd.read_csv(run_root / "analysis" / "intervention_classifications.csv")
    scenarios = sorted(classes["scenario"].unique())
    figure, axes = plt.subplots(2, len(scenarios), figsize=(7.2, 5.0), squeeze=False, constrained_layout=True)
    order = ["supported", "directionally_contradicted", "no_stable_downstream_effect", "manipulation_failure", "inconclusive"]
    for index, scenario in enumerate(scenarios):
        subset = classes[classes["scenario"] == scenario]
        counts = subset["primary_class"].value_counts().reindex(order, fill_value=0)
        axes[0, index].barh(np.arange(len(order)), counts, color=[COLOURS["green"], COLOURS["red"], COLOURS["gold"], COLOURS["grey"], "#A9AEB4"])
        axes[0, index].set_yticks(np.arange(len(order)), [value.replace("_", " ") for value in order])
        axes[0, index].set(title=f"{scenario.title()}: intervention evidence", xlabel="Relation-condition records")
        path_subset = timing[timing["scenario"] == scenario]
        for scale, colour in (("micro", COLOURS["blue"]), ("meso", COLOURS["gold"]), ("macro", COLOURS["green"])):
            values = path_subset[path_subset["scale"] == scale]["onset_time"]
            values = values[values >= 0]
            axes[1, index].scatter(np.full(len(values), {"micro": 0, "meso": 1, "macro": 2}[scale]), values, s=9, alpha=0.55, color=colour)
        axes[1, index].set_xticks([0, 1, 2], ["Micro", "Meso", "Macro"])
        axes[1, index].set(title="Observed propagation onset", ylabel="Onset time")
    for letter, ax in zip("abcd", axes.ravel()):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_7_multiscale_propagation", formats, config)


def figure_8(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    robustness = pd.read_csv(run_root / "analysis" / "observation_robustness.csv")
    scalability = pd.read_csv(run_root / "analysis" / "causal_scalability.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for scenario, colour in zip(sorted(robustness["scenario"].unique()), (COLOURS["blue"], COLOURS["red"])):
        subset = robustness[(robustness["scenario"] == scenario) & (robustness["factor"] == "observation_noise")]
        grouped = subset.groupby("noise_level")["edge_f1"]
        axes[0, 0].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
        subset = robustness[(robustness["scenario"] == scenario) & (robustness["factor"] == "missing_values")]
        grouped = subset.groupby("missing_fraction")["edge_f1"]
        axes[0, 1].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
        subset = robustness[(robustness["scenario"] == scenario) & (robustness["factor"] == "support_threshold")]
        grouped = subset.groupby("support_threshold")["retained_edge_count"]
        axes[1, 0].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
        subset = scalability[scalability["scenario"] == scenario]
        grouped = subset.groupby("candidate_indicator_count")["runtime_seconds"]
        axes[1, 1].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
    axes[0, 0].set(title="Observation noise", xlabel="Noise level", ylabel="Edge F1")
    axes[0, 1].set(title="Missing observations", xlabel="Missing fraction", ylabel="Edge F1")
    axes[1, 0].set(title="Support-threshold sensitivity", xlabel="Support threshold", ylabel="Retained edges")
    axes[1, 1].set(title="Candidate-space scaling", xlabel="Candidate indicators", ylabel="Runtime (s)")
    for ax in axes.ravel():
        ax.legend(frameon=False)
    for letter, ax in zip("abcd", axes.ravel()):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_8_robustness_efficiency", formats, config)


FIGURE_FUNCTIONS: list[Callable[[Path, Path, list[str], dict[str, Any]], list[Path]]] = [
    figure_2,
    figure_3,
    figure_4,
    figure_5,
    figure_6,
    figure_7,
    figure_8,
]


def render_all_figures(
    run_root: Path,
    plot_repo: Path | None = None,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    _style()
    snapshot = json.loads(
        (run_root / "config" / "experiment_config.snapshot.json").read_text(encoding="utf-8")
    )
    render_config = dict(snapshot["render"])
    if plot_repo is not None:
        style_path = plot_repo / "config" / "figure_config.json"
        if style_path.is_file():
            external = json.loads(style_path.read_text(encoding="utf-8"))
            render_config.update(external.get("exports", {}))
    selected_formats = [value.lower() for value in (formats or render_config["formats"])]
    invalid = sorted(set(selected_formats) - {"png", "svg", "pdf", "tiff"})
    if invalid:
        raise ValueError(f"unsupported render formats: {invalid}")
    output_root = run_root / "figures"
    outputs = []
    for function in FIGURE_FUNCTIONS:
        outputs.extend(function(run_root, output_root, selected_formats, render_config))
    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "source_run": run_root.name,
        "formats": selected_formats,
        "style_config_source": str(plot_repo / "config" / "figure_config.json") if plot_repo else "internal defaults",
        "scientific_render_contract": {
            "graph_error_edge_alpha": 0.88,
            "effect_curve_centre": "mean",
            "effect_matrix_colour_range": "full_data_range",
            "undisclosed_clipping": False,
        },
        "outputs": {
            path.relative_to(run_root).as_posix(): sha256_file(path) for path in outputs
        },
    }
    manifest_path = output_root / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    bundle_manifest = run_root / "visualization_input" / "render_manifest.json"
    if bundle_manifest.parent.is_dir():
        bundle_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return manifest

