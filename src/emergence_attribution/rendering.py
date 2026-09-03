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
from .controlled import controlled_representation


COLOURS = {
    "blue": "#2B5D7E",
    "red": "#B84A3C",
    "green": "#2F8881",
    "gold": "#C98652",
    "purple": "#786A8F",
    "grey": "#6F777B",
    "light": "#D9DAD7",
    "dark": "#20282E",
    "micro": "#E4EEF3",
    "meso": "#E1EFEC",
    "macro": "#F3E6DC",
}


def _style(config: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": config.get("font_family", ["Arial", "DejaVu Sans"]),
            "font.size": float(config.get("font_size", 7)),
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "figure.facecolor": config.get("paper_background", "#FCFBF8"),
            "axes.facecolor": config.get("paper_background", "#FCFBF8"),
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
        (run_root / "data" / "baseline_simulation_manifest.json").read_text(encoding="utf-8")
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
    figure = plt.figure(figsize=(7.2, 5.15), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.78))
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(4)]
    trace_axes = [figure.add_subplot(grid[2, :2]), figure.add_subplot(grid[2, 2:])]
    cmap = matplotlib.colors.ListedColormap(["#F4F4F2", COLOURS["blue"], COLOURS["gold"]])
    checkpoints_grid = np.linspace(0, len(grids) - 1, 4, dtype=int)
    for index, time_index in enumerate(checkpoints_grid):
        axes[index].imshow(grids[time_index] + 1, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        axes[index].set_title(f"Spatial state, t={time_index}")
        axes[index].set_xticks([])
        axes[index].set_yticks([])
    checkpoints_opinion = np.linspace(0, len(opinions) - 1, 4, dtype=int)
    x = np.arange(opinions.shape[1])
    for offset, time_index in enumerate(checkpoints_opinion, start=4):
        axes[offset].scatter(x, opinions[time_index], s=3, alpha=0.55, color=COLOURS["purple"], linewidths=0)
        axes[offset].axhline(0, color=COLOURS["light"], lw=0.6)
        axes[offset].set(title=f"Opinion state, t={time_index}", ylim=(-1.02, 1.02))
        axes[offset].set_xlabel("Agent")
    trace_axes[0].plot(unhappy, color=COLOURS["blue"], lw=1.25)
    trace_axes[0].set(title="Unsatisfied agents", xlabel="Time", ylabel="Count")
    trace_axes[1].plot(extreme, color=COLOURS["red"], lw=1.25)
    trace_axes[1].set(title="Extreme-opinion agents", xlabel="Time", ylabel="Count")
    for letter, ax in zip("abcdefghij", [*axes, *trace_axes]):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_2_simulation_dynamics", formats, config)


def _graph_positions(representation: dict[str, Any]) -> dict[str, tuple[float, float]]:
    scale_x = {"micro": 0.0, "meso": 1.0, "macro": 2.0}
    positions = {}
    for scale in ("micro", "meso", "macro"):
        nodes = [item for item in representation["indicators"] if item["scale"] == scale]
        for index, node in enumerate(sorted(nodes, key=lambda item: item["id"])):
            positions[node["id"]] = (scale_x[scale], -index)
    return positions


def figure_3(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    results = pd.read_csv(run_root / "analysis" / "controlled_recovery_results.csv")
    graphs = load_graph_records(run_root / "analysis" / "controlled_recovery_graphs.jsonl")
    evaluation = json.loads(
        (run_root / "analysis" / "controlled_recovery_graph_evaluation.json").read_text(encoding="utf-8")
    )
    scenarios = sorted(results["scenario"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.50), constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        representation = controlled_representation(scenario)
        positions = _graph_positions(representation)
        scales = {item["id"]: item["scale"] for item in representation["indicators"]}
        for column_index, method in enumerate(("unrestricted_temporal_search", "full_method")):
            ax = axes[row_index, column_index]
            graph = graphs[(scenario, method)]
            details = evaluation[f"{scenario}:{method}"]
            correct = {tuple(item) for item in details["correct_edges"]}
            network = nx.DiGraph()
            network.add_nodes_from(positions)
            network.add_edges_from((edge.source, edge.target) for edge in graph)
            nx.draw_networkx_nodes(
                network, positions, ax=ax, node_size=72,
                node_color=[COLOURS[scales[node]] for node in network.nodes],
                edgecolors=COLOURS["dark"], linewidths=0.35,
            )
            for edge in graph:
                pair = (edge.source, edge.target)
                is_correct = pair in correct
                nx.draw_networkx_edges(
                    network, positions, edgelist=[pair], ax=ax,
                    edge_color=COLOURS["blue"] if is_correct else COLOURS["gold"],
                    style="solid" if is_correct else "dashed",
                    width=1.05 if is_correct else 1.25,
                    alpha=0.92 if is_correct else float(config["graph_error_edge_alpha"]),
                    arrowsize=6,
                )
            row = results[
                (results["scenario"] == scenario) & (results["method"] == method)
            ].iloc[0]
            ax.text(
                0.02, 0.02,
                f"F1={row.edge_f1:.2f}  SHD={row.shd:.0f}\nmissed={len(details['missed_edges'])}  added={len(details['added_edges'])}",
                transform=ax.transAxes, fontsize=6.2, va="bottom",
            )
            label = "Unrestricted" if column_index == 0 else "Structured + bootstrap"
            ax.set_title(f"{scenario.title()} — {label}")
            ax.axis("off")
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
            for column_index, metric in enumerate(("temporal_qualification_rate", "stability")):
                median = grouped[metric].median().reindex(x).to_numpy()
                low = grouped[f"{metric}_ci_low"].median().reindex(x).to_numpy()
                high = grouped[f"{metric}_ci_high"].median().reindex(x).to_numpy()
                finite_median = np.isfinite(median)
                if np.any(finite_median):
                    axes[row_index, column_index].plot(
                        x[finite_median], median[finite_median], marker="o", ms=3,
                        color=colour, label=method.replace("_", " "),
                    )
                finite = np.isfinite(low) & np.isfinite(high)
                if np.any(finite):
                    axes[row_index, column_index].fill_between(x, low, high, where=finite, color=colour, alpha=0.16)
                axes[row_index, column_index].set(xlabel="Independent trajectories", ylabel=metric.replace("_", " ").title(), title=f"{scenario.title()}: {metric.replace('_', ' ')}")
                axes[row_index, column_index].set_ylim(0, 1.05)
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
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
    scenarios = list(config.get("scenario_order", sorted(selection.get("scenarios", {}))))
    figure, axes = plt.subplots(max(len(scenarios), 1), 2, figsize=(7.2, 2.4 * max(len(scenarios), 1)), squeeze=False, constrained_layout=True)
    for row_index, scenario in enumerate(scenarios):
        path_id = selection.get("scenarios", {}).get(scenario, {}).get("path_id")
        if not path_id:
            axes[row_index, 0].text(0.5, 0.5, "No complete ordered path", ha="center", va="center")
            axes[row_index, 1].text(0.5, 0.5, "No onset interval", ha="center", va="center")
            axes[row_index, 0].axis("off")
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
        onset = ordered["onset_time"].to_numpy(dtype=float)
        low = ordered["onset_ci_low"].to_numpy(dtype=float)
        high = ordered["onset_ci_high"].to_numpy(dtype=float)
        valid = (
            (onset >= 0) & np.isfinite(onset) & np.isfinite(low) & np.isfinite(high)
            & (low <= onset) & (onset <= high)
        )
        if np.any(valid):
            axes[row_index, 1].errorbar(
                np.arange(3)[valid], onset[valid],
                yerr=np.vstack([onset[valid] - low[valid], high[valid] - onset[valid]]),
                fmt="o-", color=COLOURS["blue"], capsize=3,
            )
        if not np.all(valid):
            axes[row_index, 1].text(
                0.98, 0.03, "undetected onset shown as missing",
                transform=axes[row_index, 1].transAxes, ha="right", va="bottom",
                fontsize=5.8, color=COLOURS["grey"],
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
    path_classification = pd.read_csv(
        run_root / "analysis" / "path_intervention_classification.csv"
    )
    eligible_ids = set(
        path_classification[
            path_classification["path_classification"].astype(str) == "supported"
        ]["path_id"].astype(str)
    )
    scenarios = list(config.get("scenario_order", sorted(timing["scenario"].unique())))
    figure, axes = plt.subplots(
        max(len(scenarios), 1), 1, figsize=(7.2, 4.6), squeeze=False,
        constrained_layout=True,
    )
    for index, scenario in enumerate(scenarios):
        ax = axes[index, 0]
        subset = timing[
            (timing["scenario"] == scenario)
            & (timing["path_id"].astype(str).isin(eligible_ids))
        ].copy()
        macro = subset[subset["scale"] == "macro"].copy()
        macro["magnitude"] = macro["cumulative_effect"].abs()
        selected = macro.sort_values(
            ["magnitude", "path_id"], ascending=[False, True]
        )["path_id"].drop_duplicates().head(10)
        subset = subset[subset["path_id"].isin(selected)]
        if subset.empty:
            ax.text(0.5, 0.5, "No complete multiscale path", ha="center", va="center")
            ax.axis("off")
            continue
        matrix = subset.pivot_table(
            index="path_id", columns="scale", values="cumulative_effect", aggfunc="mean"
        ).reindex(columns=["micro", "meso", "macro"])
        matrix = matrix.reindex(selected)
        values = matrix.to_numpy(dtype=float)
        limit = max(float(np.nanmax(np.abs(values))), 1e-12)
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks([0, 1, 2], ["Micro", "Meso", "Macro"])
        labels = []
        for row_number, path_id in enumerate(matrix.index, start=1):
            first = subset[subset["path_id"] == path_id].iloc[0]
            labels.append(f"P{row_number:02d}  {first['parameter']}:{first['direction']}")
        ax.set_yticks(np.arange(len(matrix)), labels, fontsize=5.4)
        ax.set_title(f"{scenario.title()} — strongest validated propagation paths")
        figure.colorbar(
            image, ax=ax, fraction=0.025, pad=0.02,
            label="Mean standardised cumulative effect",
        )
    for letter, ax in zip("ab", axes.ravel()):
        _panel(ax, letter)
    return _export(figure, output_root, "figure_7_multiscale_propagation", formats, config)


def figure_8(run_root: Path, output_root: Path, formats: list[str], config: dict[str, Any]) -> list[Path]:
    robustness = pd.read_csv(run_root / "analysis" / "observation_robustness.csv")
    scalability = pd.read_csv(run_root / "analysis" / "causal_scalability.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for scenario, colour in zip(sorted(robustness["scenario"].unique()), (COLOURS["blue"], COLOURS["red"])):
        subset = robustness[(robustness["scenario"] == scenario) & (robustness["factor"] == "observation_noise")]
        grouped = subset.groupby("noise_level")["temporal_qualification_rate"]
        axes[0, 0].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
        subset = robustness[(robustness["scenario"] == scenario) & (robustness["factor"] == "missing_values")]
        grouped = subset.groupby("missing_fraction")["temporal_qualification_rate"]
        axes[0, 1].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
        subset = robustness[(robustness["scenario"] == scenario) & (robustness["factor"] == "support_threshold")]
        grouped = subset.groupby("support_threshold")["retained_edge_count"]
        axes[1, 0].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
        subset = scalability[scalability["scenario"] == scenario]
        grouped = subset.groupby("candidate_indicator_count")["runtime_seconds"]
        axes[1, 1].plot(grouped.mean().index, grouped.mean(), marker="o", color=colour, label=scenario)
    axes[0, 0].set(title="Observation noise", xlabel="Noise level", ylabel="Qualification rate")
    axes[0, 1].set(title="Missing observations", xlabel="Missing fraction", ylabel="Qualification rate")
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
    snapshot = json.loads(
        (run_root / "config" / "experiment_config.snapshot.json").read_text(encoding="utf-8")
    )
    render_config = dict(snapshot["render"])
    render_config["scenario_order"] = list(snapshot["scenarios"])
    _style(render_config)
    style_reference_hash = None
    if plot_repo is not None:
        style_path = plot_repo / "config" / "figure_config.json"
        if not style_path.is_file():
            raise FileNotFoundError(f"local plotting reference is missing: {style_path}")
        style_reference_hash = sha256_file(style_path)
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
        "style_reference": str(plot_repo / "config" / "figure_config.json") if plot_repo else "migrated internal implementation",
        "style_reference_sha256": style_reference_hash,
        "scientific_parameters_source": "frozen experiment config snapshot",
        "scientific_render_contract": {
            "graph_error_edge_alpha": float(render_config["graph_error_edge_alpha"]),
            "effect_curve_centre": render_config["effect_curve_centre"],
            "effect_matrix_colour_range": render_config["effect_matrix_colour_range"],
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
