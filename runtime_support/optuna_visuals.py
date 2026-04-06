from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json

from runtime_support.optuna_studies import StudyContext, build_study_catalog_payload


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "Optuna visualization requires plotly and kaleido dependencies"
        ) from exc
    return go


def _safe_best_value(summary: Mapping[str, Any]) -> float:
    value = summary.get("best_value")
    if value is None:
        return 0.0
    return float(value)


def _safe_trial_count(summary: Mapping[str, Any], key: str) -> int:
    value = summary.get(key)
    if value is None:
        return 0
    return int(value)


def _write_figure_bundle(fig: Any, stem: Path) -> list[dict[str, str]]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    html_path = stem.with_suffix(".html")
    png_path = stem.with_suffix(".png")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(png_path))
    except FileNotFoundError:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(png_path))
    return [
        {"kind": "html", "path": str(html_path.resolve())},
        {"kind": "png", "path": str(png_path.resolve())},
    ]


def write_study_visualizations(
    *,
    study_summary: Mapping[str, Any],
    visuals_root: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    visuals_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, str]] = []
    go = _plotly()

    summary_fig = go.Figure(
        data=[
            go.Bar(
                x=["best_value", "trial_count", "finished_trial_count"],
                y=[
                    _safe_best_value(study_summary),
                    _safe_trial_count(study_summary, "trial_count"),
                    _safe_trial_count(study_summary, "finished_trial_count"),
                ],
            )
        ]
    )
    summary_fig.update_layout(title=str(study_summary.get("study_name", "optuna-study")))
    artifacts.extend(_write_figure_bundle(summary_fig, visuals_root / "study_summary"))

    state_counts = study_summary.get("state_counts", {})
    labels = list(state_counts) or ["pending"]
    values = [int(state_counts[key]) for key in state_counts] or [0]
    state_fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    state_fig.update_layout(title="Trial states")
    artifacts.extend(_write_figure_bundle(state_fig, visuals_root / "trial_states"))

    inventory = {
        "study_index": study_summary.get("study_index"),
        "study_name": study_summary.get("study_name"),
        "artifacts": artifacts,
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


def write_cross_study_visualizations(
    *,
    run_root: Path,
    study_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    visual_root = run_root / "visualizations"
    visual_root.mkdir(parents=True, exist_ok=True)
    entries = list(study_catalog.get("entries", []))
    go = _plotly()
    leaderboard_rows: list[dict[str, Any]] = []
    for entry in entries:
        summary = entry.get("summary", {}) if isinstance(entry, Mapping) else {}
        leaderboard_rows.append(
            {
                "study_index": int(entry.get("study_index", 0)),
                "study_label": entry.get("study_label"),
                "best_value": _safe_best_value(summary),
                "trial_count": _safe_trial_count(summary, "trial_count"),
                "finished_trial_count": _safe_trial_count(
                    summary, "finished_trial_count"
                ),
                "canonical_projection": bool(entry.get("canonical_projection")),
            }
        )

    leaderboard_path = visual_root / "cross_study_leaderboard.csv"
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard_rows[0]) if leaderboard_rows else [
            "study_index",
            "study_label",
            "best_value",
            "trial_count",
            "finished_trial_count",
            "canonical_projection",
        ])
        writer.writeheader()
        for row in leaderboard_rows:
            writer.writerow(row)

    summary_path = visual_root / "cross_study_summary.json"
    summary_payload = {
        "study_count": study_catalog.get("study_count", 0),
        "canonical_projection_study_index": study_catalog.get(
            "canonical_projection_study_index"
        ),
        "selected_study_index": study_catalog.get("selected_study_index"),
        "rows": leaderboard_rows,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    labels = [row["study_label"] for row in leaderboard_rows] or ["study-01"]
    best_values = [row["best_value"] for row in leaderboard_rows] or [0.0]
    best_fig = go.Figure(data=[go.Bar(x=labels, y=best_values)])
    best_fig.update_layout(title="Best value by study")
    best_artifacts = _write_figure_bundle(best_fig, visual_root / "best_value_by_study")

    finished_values = [row["finished_trial_count"] for row in leaderboard_rows] or [0]
    finished_fig = go.Figure(data=[go.Bar(x=labels, y=finished_values)])
    finished_fig.update_layout(title="Finished trials by study")
    finished_artifacts = _write_figure_bundle(
        finished_fig, visual_root / "finished_trials_by_study"
    )

    dashboard_path = visual_root / "cross_study_dashboard.html"
    lines = [
        "<html><body><h1>Cross-study Optuna dashboard</h1>",
        f"<p>selected_study_index={study_catalog.get('selected_study_index')}</p>",
        f"<p>canonical_projection_study_index={study_catalog.get('canonical_projection_study_index')}</p>",
        "<ul>",
    ]
    for row in leaderboard_rows:
        lines.append(
            f"<li>{row['study_label']}: best_value={row['best_value']}, finished={row['finished_trial_count']}</li>"
        )
    lines.extend(["</ul>", "</body></html>"])
    dashboard_path.write_text("\n".join(lines), encoding="utf-8")

    inventory = {
        "artifacts": [
            {"kind": "json", "path": str(summary_path.resolve())},
            {"kind": "csv", "path": str(leaderboard_path.resolve())},
            *best_artifacts,
            *finished_artifacts,
            {"kind": "html", "path": str(dashboard_path.resolve())},
        ]
    }
    inventory_path = visual_root / "artifact_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return {
        "visualizations_root": str(visual_root.resolve()),
        "artifact_inventory_path": str(inventory_path.resolve()),
    }


def build_study_visualizations(
    context: StudyContext,
    study_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return write_study_visualizations(
        study_summary=study_summary,
        visuals_root=context.visuals_root,
        inventory_path=context.visual_inventory_path,
    )


def build_cross_study_visualizations(
    stage_root: Path,
    contexts: Sequence[StudyContext],
    summary_by_study: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    study_catalog = build_study_catalog_payload(
        stage_root,
        contexts[0].selection if contexts else None,  # type: ignore[arg-type]
        study_summaries=summary_by_study,
    ) if contexts else {"study_count": 0, "entries": []}
    return write_cross_study_visualizations(run_root=stage_root, study_catalog=study_catalog)
