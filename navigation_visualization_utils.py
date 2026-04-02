"""Visualization helpers for runtime navigation routes."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Dict, List

from nav_graph_export_utils import (
    extract_xy,
    extract_xyz,
    format_position,
    nice_step,
    topology_palette,
)


def _normalized_content(content_payload: Dict[str, object]) -> tuple[list, list, list, list]:
    topologies = content_payload.get("topologies", [])
    waypoints = content_payload.get("waypoints", [])
    objects = content_payload.get("objects", [])
    edges = content_payload.get("edges", {})

    if not isinstance(topologies, list):
        topologies = []
    if not isinstance(waypoints, list):
        waypoints = []
    if not isinstance(objects, list):
        objects = []
    if not isinstance(edges, dict):
        edges = {}

    waypoint_edges = edges.get("waypoint_waypoint", [])
    if not isinstance(waypoint_edges, list):
        waypoint_edges = []
    return topologies, waypoints, objects, waypoint_edges


def _normalized_navigation(navigation_payload: Dict[str, object]) -> tuple[dict, dict, dict, list]:
    query_payload = navigation_payload.get("query", {})
    if not isinstance(query_payload, dict):
        query_payload = {}

    selected_target = navigation_payload.get("selected_target", {})
    if not isinstance(selected_target, dict):
        selected_target = {}

    goal_waypoint = selected_target.get("goal_waypoint", {})
    if not isinstance(goal_waypoint, dict):
        goal_waypoint = {}

    path_points = navigation_payload.get("path_points", [])
    if not isinstance(path_points, list):
        path_points = []

    return query_payload, selected_target, goal_waypoint, path_points


def export_navigation_route_map(
    content_payload: Dict[str, object],
    navigation_payload: Dict[str, object],
    svg_path: Path,
    html_path: Path,
) -> None:
    """Export a 2D route overlay map."""

    topologies, waypoints, _objects, waypoint_edges = _normalized_content(content_payload)
    query_payload, selected_target, goal_waypoint, path_points = _normalized_navigation(navigation_payload)

    topology_color_map: Dict[str, str] = {}
    for index, topology in enumerate(sorted(topologies, key=lambda item: str(item.get("topology_id", "")))):
        topology_color_map[str(topology.get("topology_id"))] = topology_palette(index)

    waypoints_by_id = {str(item.get("waypoint_id")): item for item in waypoints}
    route_points = []
    for item in path_points:
        if not isinstance(item, dict):
            continue
        point_xy = extract_xy(item.get("position"))
        if point_xy is not None:
            route_points.append((item, point_xy))

    current_xy = extract_xy(query_payload.get("current_position"))
    goal_waypoint_xy = extract_xy(goal_waypoint.get("position"))
    goal_object_xy = extract_xy(selected_target.get("object_centroid"))

    world_points: List[tuple[float, float]] = []
    for topology in topologies:
        xy = extract_xy(topology.get("centroid"))
        if xy is not None:
            world_points.append(xy)
    for waypoint in waypoints:
        xy = extract_xy(waypoint.get("position"))
        if xy is not None:
            world_points.append(xy)
    if current_xy is not None:
        world_points.append(current_xy)
    if goal_waypoint_xy is not None:
        world_points.append(goal_waypoint_xy)
    if goal_object_xy is not None:
        world_points.append(goal_object_xy)
    if not world_points:
        world_points = [(0.0, 0.0)]

    min_x = min(point[0] for point in world_points)
    max_x = max(point[0] for point in world_points)
    min_y = min(point[1] for point in world_points)
    max_y = max(point[1] for point in world_points)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    pad_x = max(1.0, span_x * 0.08)
    pad_y = max(1.0, span_y * 0.08)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    span_x = max_x - min_x
    span_y = max_y - min_y

    width = 1360
    height = 980
    left = 88
    right = 40
    top = 108
    bottom = 84
    draw_w = width - left - right
    draw_h = height - top - bottom
    scale = min(draw_w / span_x, draw_h / span_y)
    offset_x = left + (draw_w - span_x * scale) / 2.0

    def map_xy(world_x: float, world_y: float) -> tuple[float, float]:
        screen_x = offset_x + (world_x - min_x) * scale
        screen_y = top + draw_h - ((world_y - min_y) * scale + (draw_h - span_y * scale) / 2.0)
        return screen_x, screen_y

    grid_parts: List[str] = []
    label_parts: List[str] = []
    grid_step = nice_step(max(span_x, span_y))

    x_tick = math.floor(min_x / grid_step) * grid_step
    while x_tick <= max_x + 1e-9:
        screen_x, _ = map_xy(x_tick, min_y)
        grid_parts.append(
            f'<line x1="{screen_x:.2f}" y1="{top}" x2="{screen_x:.2f}" y2="{top + draw_h}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        label_parts.append(
            f'<text x="{screen_x:.2f}" y="{top + draw_h + 24}" font-size="12" text-anchor="middle" fill="#64748b">{x_tick:.1f}</text>'
        )
        x_tick += grid_step

    y_tick = math.floor(min_y / grid_step) * grid_step
    while y_tick <= max_y + 1e-9:
        _, screen_y = map_xy(min_x, y_tick)
        grid_parts.append(
            f'<line x1="{left}" y1="{screen_y:.2f}" x2="{left + draw_w}" y2="{screen_y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        label_parts.append(
            f'<text x="{left - 12}" y="{screen_y + 4:.2f}" font-size="12" text-anchor="end" fill="#64748b">{y_tick:.1f}</text>'
        )
        y_tick += grid_step

    base_edge_parts: List[str] = []
    topology_parts: List[str] = []
    waypoint_parts: List[str] = []
    route_parts: List[str] = []
    marker_parts: List[str] = []

    for edge in waypoint_edges:
        if not isinstance(edge, dict):
            continue
        source_waypoint = waypoints_by_id.get(str(edge.get("source_waypoint_id")))
        target_waypoint = waypoints_by_id.get(str(edge.get("target_waypoint_id")))
        if source_waypoint is None or target_waypoint is None:
            continue
        source_xy = extract_xy(source_waypoint.get("position"))
        target_xy = extract_xy(target_waypoint.get("position"))
        if source_xy is None or target_xy is None:
            continue
        sx, sy = map_xy(*source_xy)
        tx, ty = map_xy(*target_xy)
        base_edge_parts.append(
            f'<line x1="{sx:.2f}" y1="{sy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" stroke="#cbd5e1" stroke-opacity="0.75" stroke-width="2"/>'
        )

    route_waypoint_ids = {str(item.get("waypoint_id")) for item in path_points if isinstance(item, dict)}
    for topology in topologies:
        topology_id = str(topology.get("topology_id"))
        centroid_xy = extract_xy(topology.get("centroid"))
        if centroid_xy is None:
            continue
        screen_x, screen_y = map_xy(*centroid_xy)
        fill = topology_color_map.get(topology_id, "#64748b")
        topology_parts.append(
            f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="10" fill="{fill}" fill-opacity="0.12" stroke="{fill}" stroke-opacity="0.55" stroke-width="1.8"/>'
        )
        topology_parts.append(
            f'<text x="{screen_x + 12:.2f}" y="{screen_y - 10:.2f}" font-size="12" font-weight="700" fill="{fill}">{html.escape(str(topology.get("topology_key", topology_id)))}</text>'
        )

    for waypoint in waypoints:
        waypoint_id = str(waypoint.get("waypoint_id"))
        if waypoint_id in route_waypoint_ids:
            continue
        waypoint_xy = extract_xy(waypoint.get("position"))
        if waypoint_xy is None:
            continue
        screen_x, screen_y = map_xy(*waypoint_xy)
        waypoint_parts.append(
            f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="4.2" fill="#94a3b8" stroke="#ffffff" stroke-width="1.2"/>'
        )

    if len(route_points) >= 2:
        route_parts.append(
            '<polyline points="'
            + " ".join(f"{map_xy(*point_xy)[0]:.2f},{map_xy(*point_xy)[1]:.2f}" for _, point_xy in route_points)
            + '" fill="none" stroke="#ea580c" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.96"/>'
        )

    for route_item, route_xy in route_points:
        screen_x, screen_y = map_xy(*route_xy)
        path_index = route_item.get("path_index")
        route_parts.append(
            f'<circle cx="{screen_x:.2f}" cy="{screen_y:.2f}" r="8.5" fill="#ea580c" stroke="#ffffff" stroke-width="2.4"/>'
        )
        route_parts.append(
            f'<text x="{screen_x + 10:.2f}" y="{screen_y - 9:.2f}" font-size="12" font-weight="700" fill="#9a3412">{html.escape(str(path_index))}</text>'
        )

    if current_xy is not None:
        current_screen_x, current_screen_y = map_xy(*current_xy)
        marker_parts.append(f'<circle cx="{current_screen_x:.2f}" cy="{current_screen_y:.2f}" r="11" fill="none" stroke="#2563eb" stroke-width="3"/>')
        marker_parts.append(f'<circle cx="{current_screen_x:.2f}" cy="{current_screen_y:.2f}" r="4.5" fill="#2563eb" stroke="#ffffff" stroke-width="1.5"/>')
        marker_parts.append(f'<text x="{current_screen_x + 14:.2f}" y="{current_screen_y + 4:.2f}" font-size="12" font-weight="700" fill="#1d4ed8">Current</text>')

    if goal_waypoint_xy is not None:
        goal_screen_x, goal_screen_y = map_xy(*goal_waypoint_xy)
        marker_parts.append(f'<circle cx="{goal_screen_x:.2f}" cy="{goal_screen_y:.2f}" r="11" fill="none" stroke="#7c3aed" stroke-width="3"/>')
        marker_parts.append(f'<circle cx="{goal_screen_x:.2f}" cy="{goal_screen_y:.2f}" r="5.2" fill="#7c3aed" stroke="#ffffff" stroke-width="1.5"/>')
        marker_parts.append(f'<text x="{goal_screen_x + 14:.2f}" y="{goal_screen_y + 4:.2f}" font-size="12" font-weight="700" fill="#6d28d9">Goal Waypoint</text>')

    if goal_object_xy is not None:
        object_screen_x, object_screen_y = map_xy(*goal_object_xy)
        marker_parts.append(f'<circle cx="{object_screen_x:.2f}" cy="{object_screen_y:.2f}" r="9" fill="none" stroke="#dc2626" stroke-width="2.8"/>')
        marker_parts.append(f'<line x1="{object_screen_x - 7:.2f}" y1="{object_screen_y:.2f}" x2="{object_screen_x + 7:.2f}" y2="{object_screen_y:.2f}" stroke="#dc2626" stroke-width="2"/>')
        marker_parts.append(f'<line x1="{object_screen_x:.2f}" y1="{object_screen_y - 7:.2f}" x2="{object_screen_x:.2f}" y2="{object_screen_y + 7:.2f}" stroke="#dc2626" stroke-width="2"/>')
        marker_parts.append(f'<text x="{object_screen_x + 14:.2f}" y="{object_screen_y + 18:.2f}" font-size="12" font-weight="700" fill="#b91c1c">Target Object</text>')

    path_found = bool(navigation_payload.get("path_found"))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<style>text { font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }</style>'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        f'<rect x="{left}" y="{top}" width="{draw_w}" height="{draw_h}" rx="18" ry="18" fill="#ffffff" stroke="#d7e3f4" stroke-width="1.5"/>'
        '<text x="40" y="34" font-size="28" font-weight="700" fill="#0f172a">Navigation Route Map</text>'
        f'<text x="40" y="58" font-size="14" fill="#475569">Target object: {html.escape(str(query_payload.get("target_object_query", "-")))} | path found: {"yes" if path_found else "no"}</text>'
        '<line x1="40" y1="84" x2="80" y2="84" stroke="#cbd5e1" stroke-width="3"/>'
        '<text x="90" y="89" font-size="13" fill="#334155">Graph waypoint edges</text>'
        '<line x1="280" y1="84" x2="320" y2="84" stroke="#ea580c" stroke-width="5"/>'
        '<text x="330" y="89" font-size="13" fill="#334155">Selected route</text>'
        + "".join(grid_parts)
        + "".join(label_parts)
        + f'<text x="{left + draw_w / 2:.2f}" y="{height - 26}" font-size="13" text-anchor="middle" fill="#475569">world x</text>'
        + f'<text x="22" y="{top + draw_h / 2:.2f}" font-size="13" text-anchor="middle" fill="#475569" transform="rotate(-90 22 {top + draw_h / 2:.2f})">world y</text>'
        + "".join(base_edge_parts)
        + "".join(topology_parts)
        + "".join(waypoint_parts)
        + "".join(route_parts)
        + "".join(marker_parts)
        + '</svg>'
    )
    svg_path.write_text(svg, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Navigation Route Map</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
      color: #0f172a;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid #d7e3f4;
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    p {{
      margin: 6px 0;
      color: #475569;
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 12px;
      font-size: 14px;
      color: #334155;
    }}
    .meta span {{
      background: #edf4ff;
      border-radius: 999px;
      padding: 8px 12px;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 14px;
      background: #fff;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 16px;
      border-radius: 14px;
      overflow: auto;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1>Navigation Route Map</h1>
      <p>该视图在完整地图上叠加了当前导航路径、当前位置、目标 waypoint 和目标物体。</p>
      <div class="meta">
        <span>target object: {html.escape(str(query_payload.get("target_object_query", "-")))}</span>
        <span>path found: {"yes" if path_found else "no"}</span>
        <span>path points: {len(route_points)}</span>
        <span>current waypoint: {html.escape(str(navigation_payload.get("current_waypoint", {}).get("waypoint_id", "-")))}</span>
      </div>
    </div>
    <div class="panel">
      <img src="{svg_path.name}" alt="navigation route map">
    </div>
    <div class="panel">
      <details>
        <summary>Expand navigation JSON</summary>
        <pre>{html.escape(json.dumps(navigation_payload, ensure_ascii=False, indent=2))}</pre>
      </details>
    </div>
  </div>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")


def export_navigation_route_map_3d(
    content_payload: Dict[str, object],
    navigation_payload: Dict[str, object],
    html_path: Path,
) -> None:
    """Export a 3D route overlay view."""

    topologies, waypoints, _objects, waypoint_edges = _normalized_content(content_payload)
    query_payload, selected_target, goal_waypoint, path_points = _normalized_navigation(navigation_payload)

    waypoint_x: List[float] = []
    waypoint_y: List[float] = []
    waypoint_z: List[float] = []
    waypoint_text: List[str] = []
    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    edge_z: List[float | None] = []
    route_x: List[float] = []
    route_y: List[float] = []
    route_z: List[float] = []
    route_text: List[str] = []

    waypoints_by_id = {str(item.get("waypoint_id")): item for item in waypoints}

    for waypoint in waypoints:
        xyz = extract_xyz(waypoint.get("position"))
        if xyz is None:
            continue
        waypoint_x.append(xyz[0])
        waypoint_y.append(xyz[1])
        waypoint_z.append(xyz[2])
        waypoint_text.append(
            "<br>".join(
                [
                    str(waypoint.get("waypoint_id", "-")),
                    f"topology: {waypoint.get('topology_key', '-')}",
                    f"position: {format_position(waypoint.get('position'))}",
                ]
            )
        )

    for edge in waypoint_edges:
        if not isinstance(edge, dict):
            continue
        source_waypoint = waypoints_by_id.get(str(edge.get("source_waypoint_id")))
        target_waypoint = waypoints_by_id.get(str(edge.get("target_waypoint_id")))
        if source_waypoint is None or target_waypoint is None:
            continue
        source_xyz = extract_xyz(source_waypoint.get("position"))
        target_xyz = extract_xyz(target_waypoint.get("position"))
        if source_xyz is None or target_xyz is None:
            continue
        edge_x.extend([source_xyz[0], target_xyz[0], None])
        edge_y.extend([source_xyz[1], target_xyz[1], None])
        edge_z.extend([source_xyz[2], target_xyz[2], None])

    for item in path_points:
        if not isinstance(item, dict):
            continue
        xyz = extract_xyz(item.get("position"))
        if xyz is None:
            continue
        route_x.append(xyz[0])
        route_y.append(xyz[1])
        route_z.append(xyz[2])
        route_text.append(
            "<br>".join(
                [
                    str(item.get("waypoint_id", "-")),
                    f"path_index: {item.get('path_index', '-')}",
                    f"position: {format_position(item.get('position'))}",
                ]
            )
        )

    world_z_values = [*waypoint_z]
    current_xyz = extract_xyz(query_payload.get("current_position"))
    goal_waypoint_xyz = extract_xyz(goal_waypoint.get("position"))
    goal_object_xyz = extract_xyz(selected_target.get("object_centroid"))
    if current_xyz is not None:
        world_z_values.append(current_xyz[2])
    if goal_waypoint_xyz is not None:
        world_z_values.append(goal_waypoint_xyz[2])
    if goal_object_xyz is not None:
        world_z_values.append(goal_object_xyz[2])
    if not world_z_values:
        world_z_values = [0.0]
    topology_layer_z = max(world_z_values) + max(1.5, max(world_z_values) - min(world_z_values) if len(world_z_values) > 1 else 1.5)

    topology_x: List[float] = []
    topology_y: List[float] = []
    topology_z: List[float] = []
    topology_text: List[str] = []
    for topology in topologies:
        xyz = extract_xyz(topology.get("centroid"))
        if xyz is None:
            continue
        topology_x.append(xyz[0])
        topology_y.append(xyz[1])
        topology_z.append(topology_layer_z)
        topology_text.append(
            "<br>".join(
                [
                    str(topology.get("topology_key", topology.get("topology_id", "-"))),
                    f"centroid: {format_position(topology.get('centroid'))}",
                ]
            )
        )

    traces = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "name": "All Waypoint Edges",
            "x": edge_x,
            "y": edge_y,
            "z": edge_z,
            "hoverinfo": "skip",
            "showlegend": True,
            "line": {"color": "rgba(148,163,184,0.48)", "width": 3},
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "Waypoints",
            "x": waypoint_x,
            "y": waypoint_y,
            "z": waypoint_z,
            "hovertext": waypoint_text,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker": {"size": 3, "color": "#94a3b8", "opacity": 0.75, "line": {"color": "#ffffff", "width": 0.8}},
        },
    ]

    if route_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers+text",
                "name": "Selected Route",
                "x": route_x,
                "y": route_y,
                "z": route_z,
                "text": [str(index) for index in range(len(route_x))],
                "textposition": "top center",
                "hovertext": route_text,
                "hovertemplate": "%{hovertext}<extra></extra>",
                "line": {"color": "#ea580c", "width": 8},
                "marker": {"size": 6, "color": "#ea580c", "line": {"color": "#ffffff", "width": 1.2}},
            }
        )

    if current_xyz is not None:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": "Current Position",
                "x": [current_xyz[0]],
                "y": [current_xyz[1]],
                "z": [current_xyz[2]],
                "text": ["Current"],
                "textposition": "top center",
                "hovertemplate": f"Current<br>{format_position(query_payload.get('current_position'))}<extra></extra>",
                "marker": {"size": 8, "color": "#2563eb", "line": {"color": "#ffffff", "width": 1.4}},
            }
        )

    if goal_waypoint_xyz is not None:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": "Goal Waypoint",
                "x": [goal_waypoint_xyz[0]],
                "y": [goal_waypoint_xyz[1]],
                "z": [goal_waypoint_xyz[2]],
                "text": ["Goal Waypoint"],
                "textposition": "top center",
                "hovertemplate": f"{html.escape(str(goal_waypoint.get('waypoint_id', '-')))}<br>{format_position(goal_waypoint.get('position'))}<extra></extra>",
                "marker": {"size": 8, "symbol": "diamond", "color": "#7c3aed", "line": {"color": "#ffffff", "width": 1.4}},
            }
        )

    if goal_object_xyz is not None:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": "Target Object",
                "x": [goal_object_xyz[0]],
                "y": [goal_object_xyz[1]],
                "z": [goal_object_xyz[2]],
                "text": ["Target Object"],
                "textposition": "bottom center",
                "hovertemplate": f"{html.escape(str(selected_target.get('class_name', '-')))}<br>{format_position(selected_target.get('object_centroid'))}<extra></extra>",
                "marker": {"size": 8, "symbol": "x", "color": "#dc2626", "line": {"color": "#ffffff", "width": 1.4}},
            }
        )

    if topology_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": "Topologies",
                "x": topology_x,
                "y": topology_y,
                "z": topology_z,
                "text": [str(item.get("topology_key", item.get("topology_id", "-"))) for item in topologies if extract_xyz(item.get("centroid")) is not None],
                "textposition": "top center",
                "hovertext": topology_text,
                "hovertemplate": "%{hovertext}<extra></extra>",
                "marker": {"size": 5, "symbol": "diamond", "color": "#0f766e", "opacity": 0.65, "line": {"color": "#ffffff", "width": 1.0}},
            }
        )

    layout = {
        "paper_bgcolor": "#f8fafc",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 0, "r": 0, "b": 0, "t": 10},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0, "bgcolor": "rgba(255,255,255,0.72)"},
        "scene": {
            "xaxis": {"title": "x", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef", "zerolinecolor": "#bcccdc"},
            "yaxis": {"title": "y", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef", "zerolinecolor": "#bcccdc"},
            "zaxis": {"title": "z", "backgroundcolor": "#f8fbff", "gridcolor": "#d8e2ef", "zerolinecolor": "#bcccdc"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.55, "y": 1.4, "z": 1.12}},
        },
    }

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Navigation Route 3D View</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
      color: #0f172a;
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid #d7e3f4;
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    p {{
      margin: 6px 0;
      color: #475569;
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 12px;
      font-size: 14px;
      color: #334155;
    }}
    .meta span {{
      background: #edf4ff;
      border-radius: 999px;
      padding: 8px 12px;
    }}
    #plot {{
      width: 100%;
      height: 860px;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 16px;
      border-radius: 14px;
      overflow: auto;
      font-size: 13px;
    }}
    .note {{
      font-size: 13px;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <h1>Navigation Route 3D View</h1>
      <p>该视图突出显示当前导航路径、当前位置、目标 waypoint 和目标物体。</p>
      <div class="meta">
        <span>target object: {html.escape(str(query_payload.get("target_object_query", "-")))}</span>
        <span>path found: {"yes" if navigation_payload.get("path_found") else "no"}</span>
        <span>path points: {len(route_x)}</span>
      </div>
      <p class="note">This 3D page loads Plotly from the official CDN when opened.</p>
    </div>
    <div class="panel">
      <div id="plot"></div>
    </div>
    <div class="panel">
      <details>
        <summary>Expand navigation JSON</summary>
        <pre>{html.escape(json.dumps(navigation_payload, ensure_ascii=False, indent=2))}</pre>
      </details>
    </div>
  </div>
  <script>
    const traces = {json.dumps(traces, ensure_ascii=False)};
    const layout = {json.dumps(layout, ensure_ascii=False)};
    Plotly.newPlot('plot', traces, layout, {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d']
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
