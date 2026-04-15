"""Phase 8 — Visualization.

Produces four primary outputs:
  1. Constraint graph (networkx + pyvis, interactive HTML)
  2. Heatmaps of posterior distributions (matplotlib KDE + credible ellipses)
  3. Real-world map overlay (folium)
  4. Ensemble sample visualization (matplotlib scatter cloud)
  5. (Extended) Cross-novel geographic comparison
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.io import read_json, read_jsonl, data_dir
from src.utils.schemas import (
    ConstraintModel,
    GroundedEntity,
    PosteriorSummary,
    Sample,
)
from src.utils.geo import km_to_latlon

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Constraint Graph
# ---------------------------------------------------------------------------

def make_constraint_graph(
    model: ConstraintModel,
    grounded: List[GroundedEntity],
    out_path: Path,
) -> None:
    """Interactive HTML constraint graph (networkx + pyvis)."""
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError:
        log.warning("networkx or pyvis not available — skipping constraint graph")
        return

    g = nx.Graph()

    # Nodes
    id_to_name = {f.entity_id: f.name for f in model.fixed_entities}
    id_to_name.update({l.entity_id: l.name for l in model.latent_entities})

    fixed_ids = {f.entity_id for f in model.fixed_entities}

    for eid, name in id_to_name.items():
        color = "#4a90d9" if eid in fixed_ids else "#e74c3c"  # blue=real, red=fictional
        g.add_node(eid, label=name, color=color, title=f"{'Real' if eid in fixed_ids else 'Fictional'}: {name}")

    # Edges
    edge_colors = {
        "near": "#27ae60",
        "far": "#e67e22",
        "north_of": "#8e44ad",
        "south_of": "#8e44ad",
        "east_of": "#8e44ad",
        "west_of": "#8e44ad",
        "across": "#2980b9",
        "distance_approx": "#f39c12",
        "co_occurrence": "#bdc3c7",
        "in_region": "#16a085",
        "on_coast": "#1abc9c",
    }

    for c in model.constraints:
        if len(c.entities) < 2:
            continue
        e1, e2 = c.entities[0], c.entities[1]
        if not g.has_node(e1) or not g.has_node(e2):
            continue
        color = edge_colors.get(c.type, "#95a5a6")
        width = max(1, int(c.weight * 8))
        g.add_edge(e1, e2, title=f"{c.type} (w={c.weight:.2f})", color=color, width=width, label=c.type)

    net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(g)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 200
        },
        "solver": "forceAtlas2Based"
      }
    }
    """)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out_path))
    log.info("Constraint graph saved to %s", out_path)


# ---------------------------------------------------------------------------
# 2. Heatmaps
# ---------------------------------------------------------------------------

def make_heatmaps(
    all_samples: List[Sample],
    summaries: List[PosteriorSummary],
    model: ConstraintModel,
    out_dir: Path,
    resolution: int = 100,
) -> None:
    """Per-entity 2-D KDE heatmap with 95% credible ellipse overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from scipy.stats import gaussian_kde
    except ImportError:
        log.warning("matplotlib or scipy not available — skipping heatmaps")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    latent_names = {e.entity_id: e.name for e in model.latent_entities}
    summary_map = {s.entity_id: s for s in summaries}

    for entity_id, name in latent_names.items():
        xs = np.array([s.entities[entity_id].x for s in all_samples if entity_id in s.entities])
        ys = np.array([s.entities[entity_id].y for s in all_samples if entity_id in s.entities])

        if len(xs) < 10:
            log.warning("Too few samples for heatmap: %s", name)
            continue

        fig, ax = plt.subplots(figsize=(8, 7))

        # KDE
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
            x_grid = np.linspace(xs.min() - 5, xs.max() + 5, resolution)
            y_grid = np.linspace(ys.min() - 5, ys.max() + 5, resolution)
            xx, yy = np.meshgrid(x_grid, y_grid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=20, cmap="YlOrRd")
            ax.contour(xx, yy, zz, levels=5, colors="white", linewidths=0.5, alpha=0.5)
        except Exception as exc:
            log.warning("KDE failed for %s: %s", name, exc)
            ax.scatter(xs, ys, s=1, alpha=0.2, color="red")

        # 95% credible ellipse
        if entity_id in summary_map:
            s = summary_map[entity_id]
            cr = s.credible_region_95
            ellipse = mpatches.Ellipse(
                (cr.center_x, cr.center_y),
                width=2 * cr.semi_major,
                height=2 * cr.semi_minor,
                angle=cr.angle_deg,
                edgecolor="white",
                facecolor="none",
                linewidth=2,
                linestyle="--",
                label="95% credible region",
            )
            ax.add_patch(ellipse)
            ax.plot(s.posterior_mean["x"], s.posterior_mean["y"], "w*", markersize=12, label="Posterior mean")

        ax.set_xlabel("x (km east of origin)")
        ax.set_ylabel("y (km north of origin)")
        ax.set_title(f"Posterior Distribution: {name}")
        ax.legend(fontsize=9)
        plt.colorbar(ax.collections[0], ax=ax, label="Density") if ax.collections else None
        plt.tight_layout()

        safe_name = name.replace(" ", "_").replace("/", "_")
        fig.savefig(out_dir / f"{safe_name}.png", dpi=150)
        plt.close(fig)
        log.info("Heatmap saved for '%s'", name)


# ---------------------------------------------------------------------------
# 3. Folium Map Overlay
# ---------------------------------------------------------------------------

def make_map_overlay(
    model: ConstraintModel,
    grounded: List[GroundedEntity],
    summaries: List[PosteriorSummary],
    all_samples: List[Sample],
    out_path: Path,
    cfg: dict,
) -> None:
    """Interactive folium map with real entity pins and fictional entity uncertainty circles."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        log.warning("folium not available — skipping map overlay")
        return

    origin_lat = model.coordinate_system.origin_lat
    origin_lon = model.coordinate_system.origin_lon
    zoom = cfg.get("visualization", {}).get("map_zoom", 10)

    m = folium.Map(location=[origin_lat, origin_lon], zoom_start=zoom, tiles="OpenStreetMap")

    # Plot real entities
    real_group = folium.FeatureGroup(name="Real entities")
    for entity in grounded:
        if entity.type == "real" and entity.latitude is not None:
            folium.Marker(
                location=[entity.latitude, entity.longitude],
                popup=folium.Popup(
                    f"<b>{entity.canonical_name}</b><br>Real entity<br>Confidence: {entity.confidence:.2f}" if entity.confidence else f"<b>{entity.canonical_name}</b>",
                    max_width=200,
                ),
                icon=folium.Icon(color="blue", icon="map-marker"),
                tooltip=entity.canonical_name,
            ).add_to(real_group)
    real_group.add_to(m)

    # Latent entity means + uncertainty circles
    id_to_name = {e.entity_id: e.name for e in model.latent_entities}
    fict_group = folium.FeatureGroup(name="Fictional entities (posterior mean)")
    uncertainty_group = folium.FeatureGroup(name="Fictional entities (uncertainty)")

    for summary in summaries:
        mean_x = summary.posterior_mean["x"]
        mean_y = summary.posterior_mean["y"]
        std_x = summary.posterior_std["x"]
        std_y = summary.posterior_std["y"]

        mean_lat, mean_lon = km_to_latlon(mean_x, mean_y, origin_lat, origin_lon)
        uncertainty_km = math.sqrt(std_x**2 + std_y**2)
        uncertainty_m = uncertainty_km * 1000  # metres for folium Circle

        # Marker at posterior mean
        folium.Marker(
            location=[mean_lat, mean_lon],
            popup=folium.Popup(
                f"<b>{summary.name}</b><br>Fictional (inferred)<br>"
                f"Mean: ({mean_x:.2f}, {mean_y:.2f}) km<br>"
                f"Std: ({std_x:.2f}, {std_y:.2f}) km<br>"
                f"Entropy: {summary.spatial_entropy:.2f}<br>"
                f"Modes: {summary.num_modes}",
                max_width=250,
            ),
            icon=folium.Icon(color="red", icon="question-sign"),
            tooltip=f"{summary.name} (fictional)",
        ).add_to(fict_group)

        # Uncertainty circle
        folium.Circle(
            location=[mean_lat, mean_lon],
            radius=uncertainty_m,
            color="#e74c3c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.1,
            tooltip=f"{summary.name}: ±{uncertainty_km:.1f} km",
        ).add_to(uncertainty_group)

    fict_group.add_to(m)
    uncertainty_group.add_to(m)

    # Heatmap of fictional entity samples
    heat_group = folium.FeatureGroup(name="Posterior heatmap (fictional entities)")
    heat_data = []
    for sample in all_samples[:500]:  # limit for performance
        for eid, pos in sample.entities.items():
            lat, lon = km_to_latlon(pos.x, pos.y, origin_lat, origin_lon)
            heat_data.append([lat, lon])

    if heat_data:
        HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(heat_group)
        heat_group.add_to(m)

    folium.LayerControl().add_to(m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    log.info("Map overlay saved to %s", out_path)


# ---------------------------------------------------------------------------
# 4. Ensemble Sample Visualization
# ---------------------------------------------------------------------------

def make_ensemble_plot(
    all_samples: List[Sample],
    model: ConstraintModel,
    summaries: List[PosteriorSummary],
    out_path: Path,
    n_draw: int = 50,
) -> None:
    """Plot N sampled configurations as a scatter cloud with constraint edges."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        log.warning("matplotlib not available — skipping ensemble plot")
        return

    latent_names = {e.entity_id: e.name for e in model.latent_entities}
    fixed_xy = {f.entity_id: (f.x, f.y) for f in model.fixed_entities}
    fixed_names = {f.entity_id: f.name for f in model.fixed_entities}

    # Sample n_draw evenly spaced configurations
    indices = np.linspace(0, len(all_samples) - 1, min(n_draw, len(all_samples)), dtype=int)
    sampled = [all_samples[i] for i in indices]

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = cm.get_cmap("Blues", n_draw)

    # Draw fixed entities
    for eid, (x, y) in fixed_xy.items():
        name = fixed_names[eid]
        ax.scatter(x, y, s=200, color="#2c3e50", zorder=5, marker="^")
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9, color="#2c3e50", fontweight="bold")

    # Draw each sample
    for k, sample in enumerate(sampled):
        alpha = 0.3
        color = cmap(k / n_draw)

        for eid, pos in sample.entities.items():
            ax.scatter(pos.x, pos.y, s=30, color=color, alpha=alpha, zorder=3)

        # Draw constraint edges
        for c in model.constraints[:20]:  # limit edges for readability
            if len(c.entities) < 2:
                continue
            eid1, eid2 = c.entities[0], c.entities[1]

            def get_pos(e):
                if e in sample.entities:
                    p = sample.entities[e]
                    return p.x, p.y
                if e in fixed_xy:
                    return fixed_xy[e]
                return None

            p1, p2 = get_pos(eid1), get_pos(eid2)
            if p1 and p2:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha * 0.5, linewidth=0.5)

    # Overlay posterior means
    for summary in summaries:
        mx, my = summary.posterior_mean["x"], summary.posterior_mean["y"]
        ax.scatter(mx, my, s=150, color="#e74c3c", zorder=6, marker="*")
        ax.annotate(
            summary.name,
            (mx, my),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            color="#e74c3c",
            fontweight="bold",
        )

    ax.set_xlabel("x (km east of origin)")
    ax.set_ylabel("y (km north of origin)")
    ax.set_title(f"Ensemble of {len(sampled)} Sampled Geographic Configurations\n(red stars = posterior means, triangles = real places)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Ensemble plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 8: Visualization ===")

    vis_dir = Path(cfg.get("visualization", {}).get("output_dir", "visualizations"))
    constraint_graph_path = vis_dir / "constraint_graph.html"
    heatmaps_dir = vis_dir / "heatmaps"
    map_path = vis_dir / "overlay_maps" / "full_map.html"
    ensemble_path = vis_dir / "ensemble_samples" / "ensemble.png"

    if all(p.exists() for p in [constraint_graph_path, map_path, ensemble_path]) and not force:
        log.info("Visualization outputs exist, skipping (use --force to overwrite).")
        return

    # Clean old outputs to prevent stale files from prior runs
    import shutil
    if heatmaps_dir.exists():
        shutil.rmtree(heatmaps_dir)
        log.info("Cleared old heatmaps directory")

    # Load data
    dd = data_dir(cfg)
    model: ConstraintModel = read_json(dd / "constraints.json", model=ConstraintModel)
    grounded: List[GroundedEntity] = read_jsonl(dd / "grounded_entities.jsonl", model=GroundedEntity)

    summaries: List[PosteriorSummary] = read_jsonl(
        dd / "convergence" / "posterior_summaries.jsonl", model=PosteriorSummary
    )

    # Load samples (cap for performance)
    max_samples = cfg.get("visualization", {}).get("ensemble_num_samples", 50) * 20
    all_samples: List[Sample] = []
    for sf in sorted((dd / "samples").glob("*.jsonl")):
        for s in read_jsonl(sf, model=Sample):
            all_samples.append(s)
            if len(all_samples) >= max_samples:
                break
        if len(all_samples) >= max_samples:
            break

    log.info("Loaded %d samples for visualization", len(all_samples))

    # 1. Constraint graph
    make_constraint_graph(model, grounded, constraint_graph_path)

    # 2. Heatmaps
    resolution = cfg.get("visualization", {}).get("heatmap_resolution", 100)
    make_heatmaps(all_samples, summaries, model, heatmaps_dir, resolution=resolution)

    # 3. Map overlay
    make_map_overlay(model, grounded, summaries, all_samples, map_path, cfg)

    # 4. Ensemble
    n_draw = cfg.get("visualization", {}).get("ensemble_num_samples", 50)
    make_ensemble_plot(all_samples, model, summaries, ensemble_path, n_draw=n_draw)

    log.info("Phase 8 complete. Visualizations written to %s/", vis_dir)
