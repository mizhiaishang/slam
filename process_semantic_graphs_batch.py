#!/usr/bin/env python3
"""Batch-process the semantic_graphs folder into a navigation graph."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import build_nav_graph_nx as nav
from nav_graph_export_utils import (
    export_graph_contents,
    export_graph_visualization,
    export_graph_visualization_3d,
)


SOURCE_DIR = Path(r"C:\Users\mizhiaishang\Desktop\semantic_graphs")
CONFIG_PATH = Path(__file__).resolve().parent / nav.DEFAULT_CONFIG_PATH
OUTPUT_DIR = SOURCE_DIR / "generated_outputs"
GRAPH_PATH = OUTPUT_DIR / "semantic_graphs_nav_graph.pkl"
STATS_PATH = OUTPUT_DIR / "semantic_graphs_nav_graph_stats.json"
CONTENT_PATH = OUTPUT_DIR / "semantic_graphs_nav_graph_contents.json"
SVG_PATH = OUTPUT_DIR / "semantic_graphs_nav_graph_visualization.svg"
HTML_PATH = OUTPUT_DIR / "semantic_graphs_nav_graph_visualization.html"
HTML_3D_PATH = OUTPUT_DIR / "semantic_graphs_nav_graph_visualization_3d.html"


def build_batch_args() -> argparse.Namespace:
    """Load build config and pin outputs to the semantic_graphs folder."""

    config_payload = nav.load_config_file(str(CONFIG_PATH))
    section_defaults = dict(nav.DEFAULT_CONFIG["build"])
    section_from_file = config_payload.get("build", {})

    if section_from_file is None:
        section_from_file = {}
    if not isinstance(section_from_file, dict):
        raise ValueError("config section 'build' must be a JSON object")

    merged = dict(section_defaults)
    merged.update(section_from_file)

    args = argparse.Namespace(command="build")
    for key in section_defaults:
        setattr(args, key, merged.get(key))

    args.input_dir = str(SOURCE_DIR)
    args.output = str(GRAPH_PATH)
    args.recursive = True
    args.overwrite = True
    args.append = False
    return args


def extract_step_index(path: Path) -> int:
    """Read the numeric step index from a parent directory like step12."""

    match = re.search(r"step(\d+)$", path.parent.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot infer step index from parent directory: {path.parent}")
    return int(match.group(1))


def discover_step_files(topology_key_field: str) -> Tuple[List[nav.PendingFile], List[str]]:
    """Collect step*/info.json files and sort them by numeric step order."""

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"source directory not found: {SOURCE_DIR}")

    pending_files: List[Tuple[int, nav.PendingFile]] = []
    scan_errors: List[str] = []

    for path in sorted(SOURCE_DIR.glob("step*/info.json")):
        try:
            payload = nav.load_json(path)
            step_index = extract_step_index(path)
            pending_files.append(
                (
                    step_index,
                    nav.PendingFile(
                        path=path,
                        source_path=str(path.resolve()),
                        sort_key=f"{step_index:08d}",
                        timestamp=nav.normalize_text(payload.get("timestamp")),
                        topology_key=nav.topology_key_from_payload(payload, topology_key_field),
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            scan_errors.append(f"skip {path}: {exc}")

    pending_files.sort(key=lambda item: (item[0], item[1].path.as_posix().lower()))
    return [item[1] for item in pending_files], scan_errors


def build_semantic_graphs() -> Dict[str, object]:
    """Process all semantic_graphs steps and export graph artifacts."""

    nav.ensure_networkx()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = build_batch_args()
    output_path = Path(args.output).resolve()

    dynamic_classes = nav.build_dynamic_classes(args.extra_dynamic_classes)
    config = nav.build_config(args, dynamic_classes)

    pending_files, scan_errors = discover_step_files(str(args.topology_key_field))
    if not pending_files:
        raise RuntimeError(f"no step*/info.json files found under {SOURCE_DIR}")

    if output_path.exists():
        output_path.unlink()

    graph = nav.new_nav_graph(config)
    topology_index, waypoint_index, object_index = nav.rebuild_spatial_indexes(graph)
    previous_waypoint = None
    imported_waypoints = 0
    imported_observations = 0
    build_errors: List[str] = []

    for pending in pending_files:
        try:
            payload = nav.load_json(pending.path)
            previous_waypoint = nav.append_payload(
                graph=graph,
                payload=payload,
                source_path=pending.source_path,
                previous_waypoint=previous_waypoint,
                topology_index=topology_index,
                waypoint_index=waypoint_index,
                object_index=object_index,
                sort_key_override=pending.sort_key,
            )
            imported_waypoints += 1
            detections = payload.get("detections")
            if isinstance(detections, list):
                imported_observations += sum(1 for item in detections if isinstance(item, dict))
        except Exception as exc:  # noqa: BLE001
            build_errors.append(f"skip {pending.path}: {exc}")

    nav.write_graph(graph, output_path)
    content_payload = export_graph_contents(graph, CONTENT_PATH, GRAPH_PATH)
    export_graph_visualization(content_payload, SVG_PATH, HTML_PATH)
    export_graph_visualization_3d(content_payload, HTML_3D_PATH)
    inferred_topology_links = 0
    if isinstance(content_payload.get("edges"), dict):
        inferred_edges = content_payload["edges"].get("topology_topology_inferred", [])
        if isinstance(inferred_edges, list):
            inferred_topology_links = len(inferred_edges)

    stats_payload = {
        "source_dir": str(SOURCE_DIR),
        "graph_path": str(output_path),
        "data_files": [str(item.path) for item in pending_files],
        "scan_files": len(pending_files),
        "imported_waypoints": imported_waypoints,
        "imported_observations": imported_observations,
        "scan_errors": scan_errors,
        "build_errors": build_errors,
        "stats": nav.graph_stats(graph),
        "inferred_topology_links": inferred_topology_links,
        "content_path": str(CONTENT_PATH),
        "svg_path": str(SVG_PATH),
        "html_path": str(HTML_PATH),
        "html_3d_path": str(HTML_3D_PATH),
    }
    STATS_PATH.write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "source_dir": str(SOURCE_DIR),
        "graph_path": str(output_path),
        "content_path": str(CONTENT_PATH),
        "stats_path": str(STATS_PATH),
        "svg_path": str(SVG_PATH),
        "html_path": str(HTML_PATH),
        "html_3d_path": str(HTML_3D_PATH),
        "scan_files": len(pending_files),
        "topology_nodes": content_payload["stats"]["topology_nodes"],
        "waypoint_nodes": content_payload["stats"]["waypoint_nodes"],
        "object_nodes": content_payload["stats"]["object_nodes"],
        "inferred_topology_links": inferred_topology_links,
        "scan_errors": len(scan_errors),
        "build_errors": len(build_errors),
    }


def main() -> int:
    result = build_semantic_graphs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
