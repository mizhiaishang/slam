"""Independent runtime navigation script for object-guided waypoint routing.

This script is intentionally separate from the map-building workflow.
It loads an already-built navigation graph, accepts the current position,
finds a path toward the configured target object, and outputs only the
path-point information needed by downstream navigation code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import build_nav_graph_nx as nav
from navigation_visualization_utils import (
    export_navigation_route_map,
    export_navigation_route_map_3d,
)
from nav_graph_export_utils import export_graph_contents


# Change these defaults when switching to another graph or another target object.
GRAPH_PATH = Path(r"C:\Users\mizhiaishang\Desktop\semantic_graphs\generated_outputs\semantic_graphs_nav_graph.pkl")
TARGET_OBJECT_QUERY = "refrigerator"
INCLUDE_DYNAMIC = False


def safe_name_fragment(value: object) -> str:
    """Convert one free-form token into a filename-safe fragment."""

    text = nav.normalize_text(value)
    if not text:
        return "target"
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    cleaned = cleaned.strip("_")
    return cleaned or "target"


def format_coordinate_text(value: object) -> str:
    """Format one coordinate list for human-readable route logs."""

    if not isinstance(value, list) or len(value) < 3:
        return "-"
    return f"[{float(value[0]):.3f}, {float(value[1]):.3f}, {float(value[2]):.3f}]"


def parse_current_position(value: object) -> list[float]:
    """Normalize one current-position in-7put into [x, y, z]."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = nav.parse_vec3(value)
        if parsed is not None:
            return [float(parsed[0]), float(parsed[1]), float(parsed[2])]

    text = nav.normalize_text(value)
    if not text:
        raise ValueError("current position is required, expected x,y,z")

    normalized = text.replace("，", ",").replace(" ", "")
    parts = normalized.split(",")
    if len(parts) != 3:
        raise ValueError("current position must look like x,y,z")

    parsed = nav.parse_vec3(parts)
    if parsed is None:
        raise ValueError("current position must contain valid numeric x,y,z values")
    return [float(parsed[0]), float(parsed[1]), float(parsed[2])]


def compact_goal_summary(selected_target: Optional[dict]) -> Optional[dict]:
    """Keep only the target fields needed by the runtime navigation output."""

    if not isinstance(selected_target, dict):
        return None

    goal_waypoint = selected_target.get("goal_waypoint")
    if not isinstance(goal_waypoint, dict):
        goal_waypoint = None

    return {
        "object_id": selected_target.get("object_id"),
        "class_name": selected_target.get("class_name"),
        "topology_id": selected_target.get("topology_id"),
        "topology_key": selected_target.get("topology_key"),
        "object_centroid": selected_target.get("object_centroid"),
        "goal_waypoint": goal_waypoint,
        "path_distance": selected_target.get("path_distance"),
    }


def build_route_steps(path_points: Sequence[object]) -> list[dict]:
    """Convert path points into clearer step-by-step route descriptions."""

    route_steps = []
    for item in path_points:
        if not isinstance(item, dict):
            continue

        step_index = item.get("path_index")
        waypoint_id = item.get("waypoint_id")
        topology_key = item.get("topology_key")
        position = item.get("position")
        route_steps.append(
            {
                "step_index": step_index,
                "node_name": waypoint_id,
                "topology_name": topology_key,
                "coordinate": position,
                "message": (
                    f"经过节点 {waypoint_id}，"
                    f"所属拓扑为 {topology_key}，"
                    f"坐标为 {format_coordinate_text(position)}"
                ),
            }
        )
    return route_steps


def build_runtime_payload(plan_result: dict, current_position: Sequence[float], object_query: str) -> dict:
    """Convert the full planning result into a navigation-runtime payload."""

    path_points = []
    for waypoint in plan_result.get("path_waypoints", []):
        if not isinstance(waypoint, dict):
            continue
        path_points.append(
            {
                "path_index": waypoint.get("path_index"),
                "waypoint_id": waypoint.get("waypoint_id"),
                "timestamp": waypoint.get("timestamp"),
                "topology_id": waypoint.get("topology_id"),
                "topology_key": waypoint.get("topology_key"),
                "position": waypoint.get("position"),
                "source_path": waypoint.get("source_path"),
            }
        )

    route_steps = build_route_steps(path_points)
    current_waypoint = plan_result.get("current_waypoint")
    selected_target = compact_goal_summary(plan_result.get("selected_target"))

    return {
        "query": {
            "current_position": [float(current_position[0]), float(current_position[1]), float(current_position[2])],
            "target_object_query": object_query,
            "include_dynamic": INCLUDE_DYNAMIC,
        },
        "message": plan_result.get("message"),
        "path_found": bool(plan_result.get("path_found")),
        "current_waypoint": current_waypoint,
        "selected_target": selected_target,
        "route_summary": {
            "current_waypoint_id": None if not isinstance(current_waypoint, dict) else current_waypoint.get("waypoint_id"),
            "current_waypoint_coordinate": None if not isinstance(current_waypoint, dict) else current_waypoint.get("position"),
            "target_object": None if not isinstance(selected_target, dict) else selected_target.get("class_name"),
            "target_waypoint_id": None
            if not isinstance(selected_target, dict) or not isinstance(selected_target.get("goal_waypoint"), dict)
            else selected_target["goal_waypoint"].get("waypoint_id"),
            "target_waypoint_coordinate": None
            if not isinstance(selected_target, dict) or not isinstance(selected_target.get("goal_waypoint"), dict)
            else selected_target["goal_waypoint"].get("position"),
            "path_point_count": len(path_points),
            "path_distance": None if not isinstance(selected_target, dict) else selected_target.get("path_distance"),
        },
        "path_points": path_points,
        "route_steps": route_steps,
        "path_coordinates": plan_result.get("path_coordinates", []),
    }


def build_console_report(payload: dict) -> str:
    """Render a concise human-readable route report."""

    query = payload.get("query", {})
    if not isinstance(query, dict):
        query = {}
    summary = payload.get("route_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    route_steps = payload.get("route_steps", [])
    if not isinstance(route_steps, list):
        route_steps = []

    lines = [
        "导航结果",
        f"- 目标物体: {query.get('target_object_query', '-')}",
        f"- 当前位置: {format_coordinate_text(query.get('current_position'))}",
        f"- 是否找到路径: {'是' if payload.get('path_found') else '否'}",
        f"- 当前所在节点: {summary.get('current_waypoint_id', '-')}",
        f"- 当前节点坐标: {format_coordinate_text(summary.get('current_waypoint_coordinate'))}",
        f"- 目标节点: {summary.get('target_waypoint_id', '-')}",
        f"- 目标节点坐标: {format_coordinate_text(summary.get('target_waypoint_coordinate'))}",
        f"- 路径点数量: {summary.get('path_point_count', 0)}",
        f"- 路径距离: {summary.get('path_distance') if summary.get('path_distance') is not None else '-'}",
        "",
        "路径经过",
    ]

    if route_steps:
        for step in route_steps:
            if not isinstance(step, dict):
                continue
            step_index = step.get("step_index", "-")
            lines.append(f"{step_index}. {step.get('message', '-')}")
    else:
        lines.append("无可用路径点。")

    if payload.get("output_path"):
        lines.extend(
            [
                "",
                f"结果文件: {payload.get('output_path')}",
            ]
        )
    if payload.get("visualization_html_path"):
        lines.append(f"二维路径图: {payload.get('visualization_html_path')}")
    if payload.get("visualization_3d_html_path"):
        lines.append(f"三维路径图: {payload.get('visualization_3d_html_path')}")
    return "\n".join(lines)


def navigate_from_current_position(
    current_position: object,
    object_query: Optional[str] = None,
    graph_path: Optional[Path] = None,
) -> dict:
    """Load the graph and return only the path-point information."""

    nav.ensure_networkx()
    resolved_graph_path = (graph_path or GRAPH_PATH).resolve()
    if not resolved_graph_path.exists():
        raise FileNotFoundError(f"graph not found: {resolved_graph_path}")

    normalized_position = parse_current_position(current_position)
    normalized_object_query = nav.normalize_text(object_query) or TARGET_OBJECT_QUERY

    graph = nav.read_graph(resolved_graph_path)
    plan_result = nav.plan_navigation_to_object(
        graph=graph,
        current_position=normalized_position,
        object_query=normalized_object_query,
        include_dynamic=INCLUDE_DYNAMIC,
    )
    payload = build_runtime_payload(plan_result, normalized_position, normalized_object_query)

    content_path = resolved_graph_path.with_name(f"{resolved_graph_path.stem}_contents.json")
    content_payload = export_graph_contents(graph, content_path, resolved_graph_path)

    target_fragment = safe_name_fragment(normalized_object_query)
    route_svg_path = resolved_graph_path.with_name(f"{resolved_graph_path.stem}_{target_fragment}_route_map.svg")
    route_html_path = resolved_graph_path.with_name(f"{resolved_graph_path.stem}_{target_fragment}_route_map.html")
    route_html_3d_path = resolved_graph_path.with_name(f"{resolved_graph_path.stem}_{target_fragment}_route_map_3d.html")
    export_navigation_route_map(content_payload, payload, route_svg_path, route_html_path)
    export_navigation_route_map_3d(content_payload, payload, route_html_3d_path)

    output_path = resolved_graph_path.with_name(f"{resolved_graph_path.stem}_runtime_path_points.json")
    route_log_path = resolved_graph_path.with_name(f"{resolved_graph_path.stem}_{target_fragment}_route_log.txt")
    payload["output_path"] = str(output_path)
    payload["content_path"] = str(content_path)
    payload["visualization_svg_path"] = str(route_svg_path)
    payload["visualization_html_path"] = str(route_html_path)
    payload["visualization_3d_html_path"] = str(route_html_3d_path)
    payload["route_log_path"] = str(route_log_path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    route_log_path.write_text(build_console_report(payload), encoding="utf-8")
    return payload


def main(current_position: Optional[object] = None, object_query: Optional[str] = None) -> int:
    """Run the standalone navigation query.

    Usage examples:
    - Python: ``main([1.0, 2.0, 0.0], "chair")``
    - Command line: ``py object_navigation_runtime.py 1.0,2.0,0.0 chair``
    - Interactive: ``py object_navigation_runtime.py`` and then type ``1.0,2.0,0.0``
    """

    if current_position is None:
        if len(sys.argv) >= 2:
            current_position = sys.argv[1]
        else:
            current_position = input("请输入当前位置 x,y,z: ").strip()

    if object_query is None and len(sys.argv) >= 3:
        object_query = sys.argv[2]

    payload = navigate_from_current_position(current_position=current_position, object_query=object_query)
    print(build_console_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
