#!/usr/bin/env python3
"""Build a hierarchical navigation graph with NetworkX.

This script maintains a three-layer navigation graph:

1. topology nodes
   A topology node represents a higher-level area or cluster. In the current
   implementation, waypoints are grouped into a topology by a configurable JSON
   field such as ``class_node``.

2. waypoint nodes
   Each input frame/JSON becomes one waypoint node. Waypoints preserve the
   original traversal order and therefore form the main path-search layer.

3. object nodes
   Object nodes represent semantically meaningful things observed from one or
   more nearby waypoints inside the same topology. They are deduplicated by
   ``class_name`` plus spatial proximity.

The graph contains several edge types:
- topology -> waypoint: containment / attachment
- waypoint -> waypoint: temporal traversal order between adjacent steps
- waypoint -> object: mounting / observation relationship

The script supports two ingestion styles:
- batch import from many JSON files in a directory
- direct in-memory ingestion by calling ``append_payload(...)``

All runtime parameters are read from ``nav_graph_config.json``. The command line
only selects the mode: ``build``, ``stats`` or ``nearby``.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


SCHEMA_VERSION = "nx-hierarchy-1"
DEFAULT_DYNAMIC_CLASSES = {
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "bird",
}
DEFAULT_CONFIG_PATH = "nav_graph_config.json"

DEFAULT_CONFIG = {
    "build": {
        "input_dir": ".",
        "output": "nav_graph.pkl",
        "recursive": False,
        "overwrite": False,
        "append": False,
        "topology_key_field": "class_node",
        "merge_distance": 1.0,
        "distance_dims": 2,
        "waypoint_cell_size": 2.0,
        "object_cell_size": 1.0,
        "topology_cell_size": 4.0,
        "min_confidence": 0.0,
        "extra_dynamic_classes": "",
    },
    "stats": {
        "graph": "nav_graph.pkl",
    },
    "nearby": {
        "graph": "nav_graph.pkl",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "radius": 5.0,
        "topology_limit": 5,
        "waypoint_limit": 5,
        "object_limit": 10,
        "include_dynamic": False,
    },
}


@dataclass
class PendingFile:
    """Lightweight metadata for one source file before it is imported.

    The actual JSON payload is read later during the build loop. This keeps the
    scan step cheap and focused on discovery and stable ordering.
    """

    path: Path
    source_path: str
    sort_key: str
    timestamp: str
    topology_key: str


# ---------------------------------------------------------------------------
# Configuration and CLI
# ---------------------------------------------------------------------------

def ensure_networkx() -> None:
    """Fail fast if the runtime dependency is missing."""

    if nx is None:
        raise RuntimeError(
            "networkx is not installed. Please run: py -m pip install networkx"
        )


MODE_CHOICES = {
    "1": "build",
    "2": "stats",
    "3": "nearby",
    "build": "build",
    "stats": "stats",
    "nearby": "nearby",
}


def normalize_mode(mode: object) -> str:
    """Normalize and validate one mode token."""

    command = normalize_text(mode).lower()
    if command in MODE_CHOICES:
        return MODE_CHOICES[command]
    raise ValueError("unsupported mode, expected one of: 1, 2, 3, build, stats, nearby")


def select_mode() -> str:
    """Resolve the runtime mode from argv or from a tiny interactive menu.

    The script no longer exposes any parameter-level CLI. Users only choose one
    of the three supported modes, while every detailed setting still comes from
    ``nav_graph_config.json``.
    """

    if len(sys.argv) >= 2:
        return normalize_mode(sys.argv[1])

    print("请选择运行模式：")
    print("1. build")
    print("2. stats")
    print("3. nearby")
    return normalize_mode(input("请输入模式编号或名称: "))


def resolve_args(mode: Optional[object] = None) -> argparse.Namespace:
    """Build the runtime namespace from an explicit mode or from user selection."""

    command = normalize_mode(mode) if mode is not None else select_mode()
    return apply_config_to_args(argparse.Namespace(command=command))


def load_config_file(path_str: str = DEFAULT_CONFIG_PATH) -> Dict[str, object]:
    """Load the shared JSON config file.

    Returning an empty dict when the file does not exist makes it possible to
    fall back to ``DEFAULT_CONFIG`` during development.
    """

    path = Path(path_str)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a JSON object: {path}")
    return payload


def apply_config_to_args(args: argparse.Namespace) -> argparse.Namespace:
    """Merge the selected mode section from config into the parsed args.

    Example:
    - ``build`` reads ``DEFAULT_CONFIG["build"]`` and then overlays the
      ``build`` section from ``nav_graph_config.json``.
    - ``nearby`` does the same for its own section.
    """

    config_payload = load_config_file(DEFAULT_CONFIG_PATH)
    section_defaults = dict(DEFAULT_CONFIG.get(args.command, {}))
    section_from_file = config_payload.get(args.command, {})

    if section_from_file is None:
        section_from_file = {}
    if not isinstance(section_from_file, dict):
        raise ValueError(f"config section '{args.command}' must be a JSON object")

    merged_section = dict(section_defaults)
    merged_section.update(section_from_file)

    for key in section_defaults:
        setattr(args, key, merged_section.get(key))

    if args.command == "nearby":
        if args.x is None or args.y is None:
            raise ValueError("nearby requires x and y, provide them in nav_graph_config.json")

    return args


# ---------------------------------------------------------------------------
# Generic parsing helpers
# ---------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_timestamp(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    digits = "".join(char for char in text if char.isdigit())
    if digits:
        return digits.zfill(20)
    return text


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_vec3(value: object) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    x = safe_float(value[0])
    y = safe_float(value[1])
    z = safe_float(value[2])
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def parse_vec4(value: object) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not isinstance(value, Sequence):
        return (None, None, None, None)
    parts = [safe_float(item) for item in list(value[:4])]
    while len(parts) < 4:
        parts.append(None)
    return tuple(parts[:4])  # type: ignore[return-value]


def point_distance(a: Sequence[float], b: Sequence[float], dims: int) -> float:
    if dims == 2:
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def grid_key(position: Sequence[float], cell_size: float) -> Tuple[int, int, int]:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    return (
        math.floor(position[0] / cell_size),
        math.floor(position[1] / cell_size),
        math.floor(position[2] / cell_size),
    )


def iter_bucket_keys(center: Sequence[float], radius: float, cell_size: float) -> Iterable[Tuple[int, int, int]]:
    min_corner = (center[0] - radius, center[1] - radius, center[2] - radius)
    max_corner = (center[0] + radius, center[1] + radius, center[2] + radius)
    min_key = grid_key(min_corner, cell_size)
    max_key = grid_key(max_corner, cell_size)
    for bx in range(min_key[0], max_key[0] + 1):
        for by in range(min_key[1], max_key[1] + 1):
            for bz in range(min_key[2], max_key[2] + 1):
                yield (bx, by, bz)


def make_bucket_id(bucket: Tuple[int, int, int]) -> str:
    return f"{bucket[0]},{bucket[1]},{bucket[2]}"


def parse_bucket_id(bucket_text: object) -> Optional[Tuple[int, int, int]]:
    text = normalize_text(bucket_text)
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != 3:
        return None
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def bucket_neighbors(center_key: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (center_key[0] + dx, center_key[1] + dy, center_key[2] + dz)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def topology_key_from_payload(payload: dict, topology_key_field: str) -> str:
    topology_key = normalize_text(payload.get(topology_key_field))
    if topology_key:
        return topology_key
    return "topology_default"


def build_sort_key(payload: dict, path: Path, topology_key_field: str) -> str:
    timestamp_key = normalize_timestamp(payload.get("timestamp"))
    topology_key = topology_key_from_payload(payload, topology_key_field)
    return f"{timestamp_key}|{topology_key}|{path.name}|{str(path.resolve()).lower()}"


def collect_pending_files(
    input_dir: Path,
    recursive: bool,
    topology_key_field: str,
    ignored_paths: Optional[Sequence[Path]] = None,
) -> Tuple[List[PendingFile], List[str]]:
    """Discover candidate JSON files and prepare a stable import order.

    Files are not imported immediately. Instead, we build ``PendingFile``
    records containing only the metadata required to sort and reference them
    later in the build phase.
    """

    pattern = "**/*.json" if recursive else "*.json"
    paths = sorted(input_dir.glob(pattern))
    pending: List[PendingFile] = []
    errors: List[str] = []
    ignored_resolved = {
        str(path.resolve()).lower()
        for path in (ignored_paths or [])
    }

    for path in paths:
        if not path.is_file():
            continue
        if str(path.resolve()).lower() in ignored_resolved:
            continue
        try:
            payload = load_json(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"skip {path}: {exc}")
            continue

        pending.append(
            PendingFile(
                path=path,
                source_path=str(path.resolve()),
                sort_key=build_sort_key(payload, path, topology_key_field),
                timestamp=normalize_text(payload.get("timestamp")),
                topology_key=topology_key_from_payload(payload, topology_key_field),
            )
        )

    pending.sort(key=lambda item: item.sort_key)
    return pending, errors


def print_warnings(messages: Sequence[str]) -> None:
    for message in messages:
        print(message, file=sys.stderr)


def build_dynamic_classes(extra_dynamic_classes: str) -> List[str]:
    """Combine built-in dynamic classes with user-provided extra classes."""

    classes = set(DEFAULT_DYNAMIC_CLASSES)
    for item in extra_dynamic_classes.split(","):
        name = item.strip()
        if name:
            classes.add(name)
    return sorted(classes)


def new_nav_graph(config: Dict[str, object]) -> "nx.Graph":
    """Create an empty graph and initialize graph-level metadata.

    The ``counters`` dictionary is used to generate stable node/observation ids
    without depending on external storage.
    """

    graph = nx.Graph()
    graph.graph.update(
        {
            "graph_kind": "hierarchical_nav_graph",
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "counters": {
                "next_waypoint_seq": 1,
                "next_object_seq": 1,
                "next_observation_seq": 1,
            },
        }
    )
    return graph


def read_graph(path: Path) -> "nx.Graph":
    """Load a previously saved NetworkX graph from disk."""

    with path.open("rb") as handle:
        return pickle.load(handle)


def write_graph(graph: "nx.Graph", path: Path) -> None:
    """Persist the current graph to disk using pickle."""

    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_config(args: argparse.Namespace, dynamic_classes: Sequence[str]) -> Dict[str, object]:
    """Build the normalized runtime config stored inside the graph metadata."""

    return {
        "topology_key_field": args.topology_key_field,
        "merge_distance": args.merge_distance,
        "distance_dims": args.distance_dims,
        "waypoint_cell_size": args.waypoint_cell_size,
        "object_cell_size": args.object_cell_size,
        "topology_cell_size": args.topology_cell_size,
        "min_confidence": args.min_confidence,
        "dynamic_classes": list(dynamic_classes),
    }


def ensure_graph_metadata(
    graph: "nx.Graph",
    expected_config: Dict[str, object],
    append: bool,
) -> Dict[str, object]:
    """Validate that an existing graph is compatible with the requested build.

    Appending with mismatched configuration would silently corrupt semantics,
    for example by mixing different merge thresholds in the same graph.
    """

    if graph.graph.get("graph_kind") != "hierarchical_nav_graph":
        raise ValueError("graph_kind mismatch: not a hierarchical_nav_graph")

    if graph.graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, got {graph.graph.get('schema_version')}"
        )

    current = dict(graph.graph.get("config", {}))
    if not current:
        graph.graph["config"] = expected_config
        return expected_config

    if append:
        for key, value in expected_config.items():
            if current.get(key) != value:
                raise ValueError(
                    f"Existing graph config mismatch for '{key}'. existing={current.get(key)!r}, requested={value!r}"
                )
        return current

    graph.graph["config"] = expected_config
    return expected_config


def next_sequence(graph: "nx.Graph", counter_name: str) -> int:
    """Increment and return a per-graph integer counter."""

    counters = graph.graph.setdefault("counters", {})
    value = int(counters.get(counter_name, 1))
    counters[counter_name] = value + 1
    return value


def new_waypoint_node_id(graph: "nx.Graph") -> str:
    return f"waypoint:{next_sequence(graph, 'next_waypoint_seq')}"


def new_object_node_id(graph: "nx.Graph") -> str:
    return f"object:{next_sequence(graph, 'next_object_seq')}"


def new_observation_id(graph: "nx.Graph") -> str:
    return f"obs:{next_sequence(graph, 'next_observation_seq')}"


def make_topology_node_id(topology_key: str) -> str:
    """Create the stable topology node id from a grouping key."""

    key = topology_key or "topology_default"
    return f"topology:{key}"


def topology_nodes(graph: "nx.Graph") -> Iterable[Tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "topology":
            yield node_id, attrs


def waypoint_nodes(graph: "nx.Graph") -> Iterable[Tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "waypoint":
            yield node_id, attrs


def object_nodes(graph: "nx.Graph") -> Iterable[Tuple[str, dict]]:
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "object":
            yield node_id, attrs


def rebuild_spatial_indexes(
    graph: "nx.Graph",
) -> Tuple[Dict[Tuple[int, int, int], set], Dict[Tuple[int, int, int], set], Dict[Tuple[int, int, int], set]]:
    """Recreate in-memory spatial hash maps from nodes already stored in graph.

    These indexes are intentionally not stored as standalone graph objects.
    Rebuilding them keeps the serialized graph simpler while still enabling
    efficient nearby queries and merge checks at runtime.
    """

    topology_index: Dict[Tuple[int, int, int], set] = {}
    waypoint_index: Dict[Tuple[int, int, int], set] = {}
    object_index: Dict[Tuple[int, int, int], set] = {}

    for node_id, attrs in graph.nodes(data=True):
        bucket = parse_bucket_id(attrs.get("bucket"))
        if bucket is None:
            continue
        node_type = attrs.get("node_type")
        if node_type == "topology":
            topology_index.setdefault(bucket, set()).add(node_id)
        elif node_type == "waypoint":
            waypoint_index.setdefault(bucket, set()).add(node_id)
        elif node_type == "object":
            object_index.setdefault(bucket, set()).add(node_id)

    return topology_index, waypoint_index, object_index


def get_existing_source_paths(graph: "nx.Graph") -> set:
    """Return source files that have already been imported as waypoints."""

    return {
        normalize_text(attrs.get("source_path"))
        for _, attrs in waypoint_nodes(graph)
        if normalize_text(attrs.get("source_path"))
    }


def get_last_waypoint(graph: "nx.Graph") -> Optional[Tuple[str, Tuple[float, float, float], str, str]]:
    """Find the last waypoint according to the persisted sort order.

    This is used by append builds so newly imported waypoints can continue the
    temporal chain from the previous end of the graph.
    """

    candidates: List[Tuple[str, Tuple[float, float, float], str, str]] = []
    for node_id, attrs in waypoint_nodes(graph):
        position = attrs.get("position")
        if not isinstance(position, list) or len(position) < 3:
            continue
        candidates.append(
            (
                node_id,
                (float(position[0]), float(position[1]), float(position[2])),
                normalize_text(attrs.get("sort_key")),
                normalize_text(attrs.get("topology_id")),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[2], item[0]))
    return candidates[-1]


def move_index_bucket(
    node_id: str,
    old_bucket: Tuple[int, int, int],
    new_bucket: Tuple[int, int, int],
    index_map: Dict[Tuple[int, int, int], set],
) -> None:
    """Move one node from its old spatial bucket to a new one."""

    if old_bucket != new_bucket:
        current = index_map.get(old_bucket)
        if current is not None:
            current.discard(node_id)
            if not current:
                index_map.pop(old_bucket, None)
    index_map.setdefault(new_bucket, set()).add(node_id)


def get_or_create_topology(
    graph: "nx.Graph",
    topology_key: str,
    position: Sequence[float],
    timestamp: str,
    topology_index: Dict[Tuple[int, int, int], set],
) -> str:
    """Return an existing topology node or create a new one.

    Topology nodes are keyed by ``topology_key`` rather than by raw position, so
    multiple waypoints can accumulate under the same higher-level area.
    """

    topology_id = make_topology_node_id(topology_key)
    if graph.has_node(topology_id):
        return topology_id

    bucket = grid_key(position, float(graph.graph["config"]["topology_cell_size"]))
    graph.add_node(
        topology_id,
        node_type="topology",
        topology_key=topology_key,
        label=topology_key,
        centroid=[position[0], position[1], position[2]],
        waypoint_count=0,
        object_count=0,
        first_seen=timestamp,
        last_seen=timestamp,
        bucket=make_bucket_id(bucket),
    )
    topology_index.setdefault(bucket, set()).add(topology_id)
    return topology_id


def update_topology_with_waypoint(
    graph: "nx.Graph",
    topology_id: str,
    position: Sequence[float],
    timestamp: str,
    topology_index: Dict[Tuple[int, int, int], set],
) -> None:
    """Update topology centroid and counters after attaching a new waypoint."""

    attrs = graph.nodes[topology_id]
    old_count = int(attrs.get("waypoint_count", 0))
    new_count = old_count + 1
    old_bucket = parse_bucket_id(attrs.get("bucket")) or (0, 0, 0)
    centroid = attrs.get("centroid") or [position[0], position[1], position[2]]

    new_centroid = [
        ((float(centroid[0]) * old_count) + position[0]) / new_count,
        ((float(centroid[1]) * old_count) + position[1]) / new_count,
        ((float(centroid[2]) * old_count) + position[2]) / new_count,
    ]

    attrs["centroid"] = new_centroid
    attrs["waypoint_count"] = new_count
    if timestamp:
        if not normalize_text(attrs.get("first_seen")) or timestamp < normalize_text(attrs.get("first_seen")):
            attrs["first_seen"] = timestamp
        if not normalize_text(attrs.get("last_seen")) or timestamp > normalize_text(attrs.get("last_seen")):
            attrs["last_seen"] = timestamp

    new_bucket = grid_key(new_centroid, float(graph.graph["config"]["topology_cell_size"]))
    attrs["bucket"] = make_bucket_id(new_bucket)
    move_index_bucket(topology_id, old_bucket, new_bucket, topology_index)


def add_waypoint_node(
    graph: "nx.Graph",
    pending: PendingFile,
    payload: dict,
    topology_id: str,
    waypoint_index: Dict[Tuple[int, int, int], set],
) -> Tuple[str, Tuple[float, float, float]]:
    """Create one waypoint node and attach it to its topology parent."""

    position = parse_vec3(payload.get("position"))
    if position is None:
        raise ValueError(f"{pending.path} missing valid 'position'")

    rotation = parse_vec4(payload.get("rotation"))
    waypoint_id = new_waypoint_node_id(graph)
    bucket = grid_key(position, float(graph.graph["config"]["waypoint_cell_size"]))
    graph.add_node(
        waypoint_id,
        node_type="waypoint",
        topology_id=topology_id,
        topology_key=pending.topology_key,
        source_path=pending.source_path,
        sort_key=pending.sort_key,
        timestamp=pending.timestamp,
        position=[position[0], position[1], position[2]],
        rotation=[rotation[0], rotation[1], rotation[2], rotation[3]],
        observations=[],
        observation_count=0,
        bucket=make_bucket_id(bucket),
    )
    waypoint_index.setdefault(bucket, set()).add(waypoint_id)
    graph.add_edge(
        topology_id,
        waypoint_id,
        edge_type="contains_waypoint",
        relation="topology_attachment",
        traversable=False,
    )
    return waypoint_id, position


def link_waypoints(
    graph: "nx.Graph",
    previous: Optional[Tuple[str, Tuple[float, float, float], str, str]],
    current: Tuple[str, Tuple[float, float, float], str, str],
) -> None:
    """Link consecutive waypoints in the global traversal sequence.

    The waypoint chain is the primary traversable structure used for step-wise
    navigation. Topology nodes are only attached to their member waypoints; we
    intentionally do not create direct topology-to-topology shortcut edges.
    """

    if previous is None:
        return
    previous_id, previous_position, previous_sort_key, _previous_topology_id = previous
    current_id, current_position, current_sort_key, _current_topology_id = current
    if previous_sort_key and current_sort_key and current_sort_key < previous_sort_key:
        return

    distance = point_distance(previous_position, current_position, dims=3)
    graph.add_edge(
        previous_id,
        current_id,
        edge_type="temporal",
        relation="adjacent_waypoint",
        distance=distance,
        weight=distance,
        traversable=True,
    )


def merge_optional_average(old_value: Optional[float], new_value: Optional[float], old_count: int) -> Optional[float]:
    """Incrementally update an average without storing all historical samples."""

    if new_value is None:
        return old_value
    if old_value is None or old_count <= 0:
        return new_value
    return ((old_value * old_count) + new_value) / (old_count + 1)


def pick_object(
    graph: "nx.Graph",
    topology_id: str,
    class_name: str,
    position: Sequence[float],
    object_index: Dict[Tuple[int, int, int], set],
) -> Optional[str]:
    """Find the best existing object candidate inside the same topology.

    Object merging is intentionally local to one topology. This avoids unrelated
    areas sharing the same object node just because they contain similar labels.
    """

    config = graph.graph["config"]
    merge_distance = float(config["merge_distance"])
    distance_dims = int(config["distance_dims"])
    object_cell_size = float(config["object_cell_size"])
    origin_key = grid_key(position, object_cell_size)

    best_id: Optional[str] = None
    best_distance = float("inf")
    for bucket_key in bucket_neighbors(origin_key):
        for node_id in object_index.get(bucket_key, ()):
            attrs = graph.nodes[node_id]
            if attrs.get("topology_id") != topology_id:
                continue
            if attrs.get("class_name") != class_name:
                continue
            centroid = attrs.get("centroid")
            if not isinstance(centroid, list) or len(centroid) < 3:
                continue
            distance = point_distance(position, centroid, distance_dims)
            if distance <= merge_distance and distance < best_distance:
                best_distance = distance
                best_id = node_id
    return best_id


def create_object(
    graph: "nx.Graph",
    topology_id: str,
    class_name: str,
    position: Sequence[float],
    confidence: Optional[float],
    depth: Optional[float],
    timestamp: str,
    image_path: Optional[str],
    nav_candidate: bool,
    is_dynamic: bool,
    object_index: Dict[Tuple[int, int, int], set],
) -> str:
    """Create a new object node and register it in the object spatial index."""

    object_id = new_object_node_id(graph)
    bucket = grid_key(position, float(graph.graph["config"]["object_cell_size"]))
    graph.add_node(
        object_id,
        node_type="object",
        topology_id=topology_id,
        class_name=class_name,
        centroid=[position[0], position[1], position[2]],
        observation_count=1,
        waypoint_count=0,
        mean_confidence=confidence,
        min_confidence=confidence,
        max_confidence=confidence,
        mean_depth=depth,
        first_seen=timestamp,
        last_seen=timestamp,
        nav_candidate=nav_candidate,
        is_dynamic=is_dynamic,
        sample_image_path=image_path,
        bucket=make_bucket_id(bucket),
    )
    object_index.setdefault(bucket, set()).add(object_id)
    graph.nodes[topology_id]["object_count"] = int(graph.nodes[topology_id].get("object_count", 0)) + 1
    return object_id


def update_object(
    graph: "nx.Graph",
    object_id: str,
    position: Sequence[float],
    confidence: Optional[float],
    depth: Optional[float],
    timestamp: str,
    image_path: Optional[str],
    object_index: Dict[Tuple[int, int, int], set],
) -> None:
    """Merge one more observation into an existing object node."""

    attrs = graph.nodes[object_id]
    centroid = attrs["centroid"]
    old_count = int(attrs.get("observation_count", 0))
    new_count = old_count + 1
    old_bucket = parse_bucket_id(attrs.get("bucket")) or (0, 0, 0)

    new_centroid = [
        ((float(centroid[0]) * old_count) + position[0]) / new_count,
        ((float(centroid[1]) * old_count) + position[1]) / new_count,
        ((float(centroid[2]) * old_count) + position[2]) / new_count,
    ]

    attrs["centroid"] = new_centroid
    attrs["observation_count"] = new_count
    attrs["mean_confidence"] = merge_optional_average(attrs.get("mean_confidence"), confidence, old_count)
    attrs["mean_depth"] = merge_optional_average(attrs.get("mean_depth"), depth, old_count)
    if confidence is not None:
        current_min = attrs.get("min_confidence")
        current_max = attrs.get("max_confidence")
        attrs["min_confidence"] = confidence if current_min is None else min(float(current_min), confidence)
        attrs["max_confidence"] = confidence if current_max is None else max(float(current_max), confidence)
    if timestamp:
        if not normalize_text(attrs.get("first_seen")) or timestamp < normalize_text(attrs.get("first_seen")):
            attrs["first_seen"] = timestamp
        if not normalize_text(attrs.get("last_seen")) or timestamp > normalize_text(attrs.get("last_seen")):
            attrs["last_seen"] = timestamp
    if not attrs.get("sample_image_path") and image_path:
        attrs["sample_image_path"] = image_path

    new_bucket = grid_key(new_centroid, float(graph.graph["config"]["object_cell_size"]))
    attrs["bucket"] = make_bucket_id(new_bucket)
    move_index_bucket(object_id, old_bucket, new_bucket, object_index)


def append_observation_record(
    graph: "nx.Graph",
    waypoint_id: str,
    detection: dict,
    object_id: Optional[str],
) -> None:
    """Store the raw detection under the waypoint node.

    Even if the detection cannot be merged into an object node, we still keep
    the raw observation so later debugging or reprocessing is possible.
    """

    attrs = graph.nodes[waypoint_id]
    bbox = detection.get("bbox") or {}
    global_position = parse_vec3(detection.get("global_position"))
    attrs["observations"].append(
        {
            "observation_id": new_observation_id(graph),
            "class_name": normalize_text(detection.get("class_name")),
            "confidence": safe_float(detection.get("confidence")),
            "global_position": list(global_position) if global_position is not None else None,
            "depth": safe_float(detection.get("depth")),
            "bbox": {
                "x1": safe_float(bbox.get("x1")),
                "y1": safe_float(bbox.get("y1")),
                "x2": safe_float(bbox.get("x2")),
                "y2": safe_float(bbox.get("y2")),
            },
            "image_path": normalize_text(detection.get("test_img_file")) or None,
            "object_id": object_id,
        }
    )
    attrs["observation_count"] = int(attrs.get("observation_count", 0)) + 1


def update_waypoint_object_edge(
    graph: "nx.Graph",
    waypoint_id: str,
    object_id: str,
    confidence: Optional[float],
    depth: Optional[float],
    timestamp: str,
) -> None:
    """Create or update the edge that says a waypoint mounts an object."""

    if graph.has_edge(waypoint_id, object_id):
        edge = graph.edges[waypoint_id, object_id]
        if edge.get("edge_type") == "mounts_object":
            old_count = int(edge.get("observation_count", 0))
            edge["observation_count"] = old_count + 1
            edge["mean_confidence"] = merge_optional_average(edge.get("mean_confidence"), confidence, old_count)
            edge["mean_depth"] = merge_optional_average(edge.get("mean_depth"), depth, old_count)
            if timestamp and timestamp > normalize_text(edge.get("last_seen")):
                edge["last_seen"] = timestamp
            return

    graph.add_edge(
        waypoint_id,
        object_id,
        edge_type="mounts_object",
        relation="waypoint_observation",
        observation_count=1,
        mean_confidence=confidence,
        mean_depth=depth,
        first_seen=timestamp,
        last_seen=timestamp,
        traversable=False,
    )
    graph.nodes[object_id]["waypoint_count"] = int(graph.nodes[object_id].get("waypoint_count", 0)) + 1


def append_payload(
    graph: "nx.Graph",
    payload: dict,
    source_path: str,
    previous_waypoint: Optional[Tuple[str, Tuple[float, float, float], str, str]],
    topology_index: Dict[Tuple[int, int, int], set],
    waypoint_index: Dict[Tuple[int, int, int], set],
    object_index: Dict[Tuple[int, int, int], set],
    sort_key_override: Optional[str] = None,
) -> Tuple[str, Tuple[float, float, float], str, str]:
    """Ingest one payload into the three-layer graph.

    Import order inside this function:
    1. determine the topology group for the payload
    2. create the waypoint node under that topology
    3. connect it to the previous waypoint if needed
    4. convert each detection into either:
       - an updated object node
       - a new object node
       - or only a raw observation if it lacks global position
    """

    config = graph.graph["config"]
    topology_key_field = str(config["topology_key_field"])
    topology_key = topology_key_from_payload(payload, topology_key_field)

    pending = PendingFile(
        path=Path(source_path),
        source_path=source_path,
        sort_key=normalize_text(sort_key_override)
        or build_sort_key(payload, Path(source_path), topology_key_field),
        timestamp=normalize_text(payload.get("timestamp")),
        topology_key=topology_key,
    )

    position = parse_vec3(payload.get("position"))
    if position is None:
        raise ValueError(f"{source_path} missing valid 'position'")

    min_confidence = float(config["min_confidence"])
    dynamic_classes = set(config["dynamic_classes"])

    # The topology layer is the parent area/container for this waypoint.
    topology_id = get_or_create_topology(
        graph=graph,
        topology_key=topology_key,
        position=position,
        timestamp=pending.timestamp,
        topology_index=topology_index,
    )
    update_topology_with_waypoint(graph, topology_id, position, pending.timestamp, topology_index)

    # The waypoint layer preserves frame-level traversal history.
    waypoint_id, waypoint_position = add_waypoint_node(
        graph=graph,
        pending=pending,
        payload=payload,
        topology_id=topology_id,
        waypoint_index=waypoint_index,
    )
    current_waypoint = (waypoint_id, waypoint_position, pending.sort_key, topology_id)
    link_waypoints(graph, previous_waypoint, current_waypoint)

    detections = payload.get("detections")
    if not isinstance(detections, list):
        detections = []

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        class_name = normalize_text(detection.get("class_name"))
        if not class_name:
            continue

        confidence = safe_float(detection.get("confidence"))
        if confidence is not None and confidence < min_confidence:
            continue

        global_position = parse_vec3(detection.get("global_position"))
        depth = safe_float(detection.get("depth"))
        image_path = normalize_text(detection.get("test_img_file")) or None
        is_dynamic = class_name in dynamic_classes
        nav_candidate = not is_dynamic
        object_id: Optional[str] = None

        if global_position is not None:
            # Object nodes are matched only inside the same topology.
            object_id = pick_object(
                graph=graph,
                topology_id=topology_id,
                class_name=class_name,
                position=global_position,
                object_index=object_index,
            )
            if object_id is None:
                object_id = create_object(
                    graph=graph,
                    topology_id=topology_id,
                    class_name=class_name,
                    position=global_position,
                    confidence=confidence,
                    depth=depth,
                    timestamp=pending.timestamp,
                    image_path=image_path,
                    nav_candidate=nav_candidate,
                    is_dynamic=is_dynamic,
                    object_index=object_index,
                )
            else:
                update_object(
                    graph=graph,
                    object_id=object_id,
                    position=global_position,
                    confidence=confidence,
                    depth=depth,
                    timestamp=pending.timestamp,
                    image_path=image_path,
                    object_index=object_index,
                )
            # This edge expresses that the object is mounted under the waypoint.
            update_waypoint_object_edge(graph, waypoint_id, object_id, confidence, depth, pending.timestamp)

        # Raw evidence is kept regardless of whether object merging succeeded.
        append_observation_record(graph, waypoint_id, detection, object_id)

    return current_waypoint


def build_graph(args: argparse.Namespace) -> int:
    """Batch-import JSON files and save the resulting graph to disk."""

    ensure_networkx()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    config_path = Path(DEFAULT_CONFIG_PATH).resolve()
    dynamic_classes = build_dynamic_classes(args.extra_dynamic_classes)
    config = build_config(args, dynamic_classes)

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    if output_path.exists() and args.overwrite:
        output_path.unlink()
    elif output_path.exists() and not args.append:
        raise FileExistsError(
            f"output graph already exists: {output_path}. "
            f"Please update 'build.overwrite' or 'build.append' in {DEFAULT_CONFIG_PATH}."
        )

    pending_files, scan_errors = collect_pending_files(
        input_dir=input_dir,
        recursive=args.recursive,
        topology_key_field=args.topology_key_field,
        ignored_paths=[config_path],
    )
    if not pending_files:
        print_warnings(scan_errors)
        raise RuntimeError(f"no JSON files found under {input_dir}")

    if args.append and output_path.exists():
        # Append mode reuses the existing graph but demands config compatibility.
        graph = read_graph(output_path)
        ensure_graph_metadata(graph, config, append=True)
    else:
        graph = new_nav_graph(config)

    # Spatial indexes are runtime acceleration structures rebuilt from nodes.
    topology_index, waypoint_index, object_index = rebuild_spatial_indexes(graph)
    existing_sources = get_existing_source_paths(graph) if args.append else set()
    previous_waypoint = get_last_waypoint(graph) if args.append else None

    imported_waypoints = 0
    imported_observations = 0
    skipped_existing = 0
    build_errors: List[str] = []
    objects_before = sum(1 for _ in object_nodes(graph))

    for pending in pending_files:
        if pending.source_path == str(output_path):
            continue
        if pending.source_path in existing_sources:
            skipped_existing += 1
            continue

        try:
            payload = load_json(pending.path)
            previous_waypoint = append_payload(
                graph=graph,
                payload=payload,
                source_path=pending.source_path,
                previous_waypoint=previous_waypoint,
                topology_index=topology_index,
                waypoint_index=waypoint_index,
                object_index=object_index,
            )
            imported_waypoints += 1
            detections = payload.get("detections")
            if isinstance(detections, list):
                imported_observations += sum(1 for item in detections if isinstance(item, dict))
        except Exception as exc:  # noqa: BLE001
            build_errors.append(f"skip {pending.path}: {exc}")

    write_graph(graph, output_path)

    objects_after = sum(1 for _ in object_nodes(graph))
    print_warnings(scan_errors)
    print_warnings(build_errors)
    print(
        json.dumps(
            {
                "graph_path": str(output_path),
                "input_dir": str(input_dir),
                "scan_files": len(pending_files),
                "imported_waypoints": imported_waypoints,
                "imported_observations": imported_observations,
                "new_objects": objects_after - objects_before,
                "skipped_existing_files": skipped_existing,
                "scan_errors": len(scan_errors),
                "build_errors": len(build_errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def graph_stats(graph: "nx.Graph") -> Dict[str, object]:
    """Summarize graph contents for quick inspection and debugging."""

    topology_count = 0
    waypoint_count = 0
    object_count = 0
    nav_object_count = 0
    dynamic_object_count = 0
    raw_observation_count = 0
    temporal_edge_count = 0
    waypoint_object_edge_count = 0

    for _, attrs in topology_nodes(graph):
        topology_count += 1

    for _, attrs in waypoint_nodes(graph):
        waypoint_count += 1
        raw_observation_count += len(attrs.get("observations", []))

    for _, attrs in object_nodes(graph):
        object_count += 1
        if attrs.get("nav_candidate"):
            nav_object_count += 1
        if attrs.get("is_dynamic"):
            dynamic_object_count += 1

    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get("edge_type")
        if edge_type == "temporal":
            temporal_edge_count += 1
        elif edge_type == "mounts_object":
            waypoint_object_edge_count += 1

    return {
        "schema_version": graph.graph.get("schema_version"),
        "config": graph.graph.get("config", {}),
        "topology_nodes": topology_count,
        "waypoint_nodes": waypoint_count,
        "object_nodes": object_count,
        "nav_objects": nav_object_count,
        "dynamic_objects": dynamic_object_count,
        "temporal_edges": temporal_edge_count,
        "waypoint_object_edges": waypoint_object_edge_count,
        "raw_observations": raw_observation_count,
    }


def print_stats(args: argparse.Namespace) -> int:
    """Load a saved graph and print a compact statistics report."""

    ensure_networkx()
    graph_path = Path(args.graph).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"graph not found: {graph_path}")
    graph = read_graph(graph_path)
    payload = {"graph_path": str(graph_path), **graph_stats(graph)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def query_topologies(
    graph: "nx.Graph",
    center: Tuple[float, float, float],
    radius: float,
    topology_limit: int,
    topology_index: Dict[Tuple[int, int, int], set],
) -> List[dict]:
    """Query nearby topology nodes using the topology spatial index."""

    config = graph.graph["config"]
    topology_cell_size = float(config["topology_cell_size"])
    distance_dims = int(config["distance_dims"])

    candidates: List[dict] = []
    seen = set()
    for bucket in iter_bucket_keys(center, radius, topology_cell_size):
        for node_id in topology_index.get(bucket, ()):
            if node_id in seen:
                continue
            seen.add(node_id)
            attrs = graph.nodes[node_id]
            centroid = attrs.get("centroid")
            if not isinstance(centroid, list) or len(centroid) < 3:
                continue
            distance = point_distance(center, centroid, distance_dims)
            if distance <= radius:
                candidates.append(
                    {
                        "topology_id": node_id,
                        "topology_key": attrs.get("topology_key"),
                        "distance": distance,
                        "centroid": centroid,
                        "waypoint_count": attrs.get("waypoint_count"),
                        "object_count": attrs.get("object_count"),
                    }
                )

    candidates.sort(key=lambda item: (item["distance"], item["topology_id"]))
    return candidates[:topology_limit]


def query_waypoints(
    graph: "nx.Graph",
    center: Tuple[float, float, float],
    radius: float,
    waypoint_limit: int,
    waypoint_index: Dict[Tuple[int, int, int], set],
) -> List[dict]:
    """Query nearby waypoint nodes using the waypoint spatial index."""

    config = graph.graph["config"]
    waypoint_cell_size = float(config["waypoint_cell_size"])
    distance_dims = int(config["distance_dims"])

    candidates: List[dict] = []
    seen = set()
    for bucket in iter_bucket_keys(center, radius, waypoint_cell_size):
        for node_id in waypoint_index.get(bucket, ()):
            if node_id in seen:
                continue
            seen.add(node_id)
            attrs = graph.nodes[node_id]
            position = attrs.get("position")
            if not isinstance(position, list) or len(position) < 3:
                continue
            distance = point_distance(center, position, distance_dims)
            if distance <= radius:
                candidates.append(
                    {
                        "waypoint_id": node_id,
                        "distance": distance,
                        "timestamp": attrs.get("timestamp"),
                        "topology_id": attrs.get("topology_id"),
                        "topology_key": attrs.get("topology_key"),
                        "source_path": attrs.get("source_path"),
                        "position": position,
                    }
                )

    candidates.sort(key=lambda item: (item["distance"], item["waypoint_id"]))
    return candidates[:waypoint_limit]


def query_objects(
    graph: "nx.Graph",
    center: Tuple[float, float, float],
    radius: float,
    object_limit: int,
    include_dynamic: bool,
    object_index: Dict[Tuple[int, int, int], set],
) -> List[dict]:
    """Query nearby object nodes using the object spatial index."""

    config = graph.graph["config"]
    object_cell_size = float(config["object_cell_size"])
    distance_dims = int(config["distance_dims"])

    candidates: List[dict] = []
    seen = set()
    for bucket in iter_bucket_keys(center, radius, object_cell_size):
        for node_id in object_index.get(bucket, ()):
            if node_id in seen:
                continue
            seen.add(node_id)
            attrs = graph.nodes[node_id]
            if not include_dynamic and not attrs.get("nav_candidate"):
                continue
            centroid = attrs.get("centroid")
            if not isinstance(centroid, list) or len(centroid) < 3:
                continue
            distance = point_distance(center, centroid, distance_dims)
            if distance <= radius:
                candidates.append(
                    {
                        "object_id": node_id,
                        "class_name": attrs.get("class_name"),
                        "topology_id": attrs.get("topology_id"),
                        "distance": distance,
                        "centroid": centroid,
                        "observation_count": attrs.get("observation_count"),
                        "waypoint_count": attrs.get("waypoint_count"),
                        "mean_confidence": attrs.get("mean_confidence"),
                        "mean_depth": attrs.get("mean_depth"),
                        "nav_candidate": bool(attrs.get("nav_candidate")),
                        "is_dynamic": bool(attrs.get("is_dynamic")),
                    }
                )

    candidates.sort(key=lambda item: (item["distance"], item["object_id"]))
    return candidates[:object_limit]


def summarize_waypoint_for_navigation(
    graph: "nx.Graph",
    waypoint_id: str,
    distance: Optional[float] = None,
) -> dict:
    """Build a compact waypoint summary used by route-planning results."""

    attrs = graph.nodes[waypoint_id]
    summary = {
        "waypoint_id": waypoint_id,
        "timestamp": attrs.get("timestamp"),
        "topology_id": attrs.get("topology_id"),
        "topology_key": attrs.get("topology_key"),
        "source_path": attrs.get("source_path"),
        "position": attrs.get("position"),
        "observation_count": attrs.get("observation_count"),
    }
    if distance is not None:
        summary["distance"] = distance
    return summary


def summarize_object_for_navigation(graph: "nx.Graph", object_id: str) -> dict:
    """Build a compact object summary used by navigation results."""

    attrs = graph.nodes[object_id]
    topology_id = normalize_text(attrs.get("topology_id"))
    topology_key = None
    if topology_id and topology_id in graph.nodes:
        topology_key = graph.nodes[topology_id].get("topology_key")

    return {
        "object_id": object_id,
        "class_name": attrs.get("class_name"),
        "topology_id": topology_id or None,
        "topology_key": topology_key,
        "centroid": attrs.get("centroid"),
        "observation_count": attrs.get("observation_count"),
        "waypoint_count": attrs.get("waypoint_count"),
        "mean_confidence": attrs.get("mean_confidence"),
        "mean_depth": attrs.get("mean_depth"),
        "nav_candidate": bool(attrs.get("nav_candidate")),
        "is_dynamic": bool(attrs.get("is_dynamic")),
    }


def find_nearest_waypoint_to_position(
    graph: "nx.Graph",
    current_position: Sequence[float],
) -> Optional[dict]:
    """Return the nearest waypoint to the current robot position."""

    distance_dims = int(graph.graph["config"]["distance_dims"])
    best_summary: Optional[dict] = None
    best_distance = float("inf")

    for waypoint_id, attrs in waypoint_nodes(graph):
        position = attrs.get("position")
        if not isinstance(position, list) or len(position) < 3:
            continue
        distance = point_distance(current_position, position, distance_dims)
        if distance < best_distance:
            best_distance = distance
            best_summary = summarize_waypoint_for_navigation(graph, waypoint_id, distance=distance)

    return best_summary


def find_object_matches(
    graph: "nx.Graph",
    object_query: object,
    include_dynamic: bool = False,
) -> List[str]:
    """Resolve one user-provided object token into candidate object node ids.

    Matching priority:
    1. exact object id
    2. exact class name
    3. substring match in object id or class name
    """

    query = normalize_text(object_query).lower()
    if not query:
        return []

    exact_id_matches: List[str] = []
    exact_class_matches: List[str] = []
    fuzzy_matches: List[str] = []

    for object_id, attrs in object_nodes(graph):
        if not include_dynamic and not attrs.get("nav_candidate"):
            continue

        object_id_text = normalize_text(object_id).lower()
        class_name_text = normalize_text(attrs.get("class_name")).lower()
        if query == object_id_text:
            exact_id_matches.append(object_id)
        elif query == class_name_text:
            exact_class_matches.append(object_id)
        elif query in object_id_text or query in class_name_text:
            fuzzy_matches.append(object_id)

    candidates = exact_id_matches or exact_class_matches or fuzzy_matches
    candidates.sort(
        key=lambda node_id: (
            normalize_text(graph.nodes[node_id].get("class_name")),
            node_id,
        )
    )
    return candidates


def linked_waypoints_for_object(graph: "nx.Graph", object_id: str) -> List[dict]:
    """List waypoint nodes directly connected to one object node."""

    if object_id not in graph.nodes or graph.nodes[object_id].get("node_type") != "object":
        return []

    object_attrs = graph.nodes[object_id]
    centroid = object_attrs.get("centroid")
    distance_dims = int(graph.graph["config"]["distance_dims"])
    linked: List[dict] = []

    for neighbor_id in graph.neighbors(object_id):
        neighbor_attrs = graph.nodes[neighbor_id]
        if neighbor_attrs.get("node_type") != "waypoint":
            continue

        edge = graph.edges[object_id, neighbor_id]
        if edge.get("edge_type") != "mounts_object":
            continue

        position = neighbor_attrs.get("position")
        distance_to_object = None
        if isinstance(centroid, list) and len(centroid) >= 3 and isinstance(position, list) and len(position) >= 3:
            distance_to_object = point_distance(centroid, position, distance_dims)

        waypoint_summary = summarize_waypoint_for_navigation(graph, neighbor_id, distance=distance_to_object)
        waypoint_summary["distance_to_object"] = distance_to_object
        waypoint_summary["edge_observation_count"] = edge.get("observation_count")
        waypoint_summary["edge_mean_confidence"] = edge.get("mean_confidence")
        waypoint_summary["edge_mean_depth"] = edge.get("mean_depth")
        waypoint_summary["first_seen"] = edge.get("first_seen")
        waypoint_summary["last_seen"] = edge.get("last_seen")
        linked.append(waypoint_summary)

    linked.sort(
        key=lambda item: (
            float("inf") if item.get("distance_to_object") is None else float(item["distance_to_object"]),
            str(item["waypoint_id"]),
        )
    )
    return linked


def build_traversable_waypoint_graph(graph: "nx.Graph") -> "nx.Graph":
    """Project the full graph into a waypoint-only traversable graph."""

    route_graph = nx.Graph()
    for waypoint_id, _ in waypoint_nodes(graph):
        route_graph.add_node(waypoint_id)

    for source_id, target_id, attrs in graph.edges(data=True):
        if attrs.get("edge_type") != "temporal":
            continue
        if not bool(attrs.get("traversable")):
            continue
        if graph.nodes[source_id].get("node_type") != "waypoint":
            continue
        if graph.nodes[target_id].get("node_type") != "waypoint":
            continue

        weight = safe_float(attrs.get("weight"))
        distance = safe_float(attrs.get("distance"))
        route_graph.add_edge(
            source_id,
            target_id,
            weight=weight if weight is not None else (distance if distance is not None else 1.0),
            distance=distance,
        )

    return route_graph


def plan_navigation_to_object(
    graph: "nx.Graph",
    current_position: Sequence[float],
    object_query: object,
    include_dynamic: bool = False,
) -> dict:
    """Plan a waypoint path from the current position to a target object.

    Workflow:
    1. resolve the user-provided object query into candidate object nodes
    2. collect all waypoints mounted under those objects
    3. choose the nearest current waypoint to the robot position
    4. search the waypoint traversal graph for a reachable path
    5. return waypoint metadata plus coordinate lists for downstream navigation
    """

    position = parse_vec3(current_position)
    if position is None:
        raise ValueError("current_position must be a 3D coordinate like [x, y, z]")

    current_waypoint = find_nearest_waypoint_to_position(graph, position)
    matched_object_ids = find_object_matches(graph, object_query, include_dynamic=include_dynamic)
    route_graph = build_traversable_waypoint_graph(graph)

    matched_objects: List[dict] = []
    target_waypoint_candidates: List[dict] = []
    best_candidate: Optional[dict] = None

    current_waypoint_id = None if current_waypoint is None else str(current_waypoint["waypoint_id"])

    for object_id in matched_object_ids:
        object_summary = summarize_object_for_navigation(graph, object_id)
        attached_waypoints = linked_waypoints_for_object(graph, object_id)
        object_summary["attached_waypoints"] = attached_waypoints
        matched_objects.append(object_summary)

        for goal_waypoint in attached_waypoints:
            goal_waypoint_id = str(goal_waypoint["waypoint_id"])
            candidate = {
                "object_id": object_summary["object_id"],
                "class_name": object_summary["class_name"],
                "topology_id": object_summary["topology_id"],
                "topology_key": object_summary["topology_key"],
                "object_centroid": object_summary["centroid"],
                "goal_waypoint": goal_waypoint,
                "reachable": False,
                "path_distance": None,
                "path_waypoint_ids": [],
            }

            if current_waypoint_id and current_waypoint_id in route_graph and goal_waypoint_id in route_graph:
                try:
                    path_waypoint_ids = nx.shortest_path(
                        route_graph,
                        source=current_waypoint_id,
                        target=goal_waypoint_id,
                        weight="weight",
                    )
                    path_distance = nx.shortest_path_length(
                        route_graph,
                        source=current_waypoint_id,
                        target=goal_waypoint_id,
                        weight="weight",
                    )
                    candidate["reachable"] = True
                    candidate["path_distance"] = path_distance
                    candidate["path_waypoint_ids"] = path_waypoint_ids
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

            target_waypoint_candidates.append(candidate)

    target_waypoint_candidates.sort(
        key=lambda item: (
            not bool(item["reachable"]),
            float("inf") if item["path_distance"] is None else float(item["path_distance"]),
            float("inf")
            if item["goal_waypoint"].get("distance_to_object") is None
            else float(item["goal_waypoint"]["distance_to_object"]),
            str(item["class_name"]),
            str(item["object_id"]),
            str(item["goal_waypoint"]["waypoint_id"]),
        )
    )

    for candidate in target_waypoint_candidates:
        if candidate["reachable"]:
            best_candidate = candidate
            break

    path_waypoints: List[dict] = []
    path_coordinates: List[List[float]] = []
    if best_candidate is not None:
        for path_index, waypoint_id in enumerate(best_candidate["path_waypoint_ids"]):
            waypoint_summary = summarize_waypoint_for_navigation(graph, waypoint_id)
            waypoint_summary["path_index"] = path_index
            path_waypoints.append(waypoint_summary)

            position_value = waypoint_summary.get("position")
            if isinstance(position_value, list) and len(position_value) >= 3:
                path_coordinates.append(
                    [float(position_value[0]), float(position_value[1]), float(position_value[2])]
                )

    message = ""
    if not matched_object_ids:
        message = "未找到匹配的目标物体。"
    elif current_waypoint is None:
        message = "图中没有可用的 waypoint，无法规划路径。"
    elif best_candidate is None:
        message = "已找到目标物体，但当前位置对应的 waypoint 与目标关联 waypoint 之间不可达。"
    else:
        message = "已找到从当前位置到目标物体关联 waypoint 的可达路径。"

    return {
        "query": {
            "object_query": normalize_text(object_query),
            "current_position": [position[0], position[1], position[2]],
            "include_dynamic": include_dynamic,
            "distance_dims": graph.graph["config"]["distance_dims"],
        },
        "message": message,
        "current_waypoint": current_waypoint,
        "matched_objects": matched_objects,
        "target_waypoint_candidates": target_waypoint_candidates,
        "selected_target": best_candidate,
        "path_found": best_candidate is not None,
        "path_waypoints": path_waypoints,
        "path_coordinates": path_coordinates,
    }


def print_nearby(args: argparse.Namespace) -> int:
    """Load a saved graph and print the nearby query result bundle."""

    ensure_networkx()
    graph_path = Path(args.graph).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"graph not found: {graph_path}")

    graph = read_graph(graph_path)
    topology_index, waypoint_index, object_index = rebuild_spatial_indexes(graph)
    center = (args.x, args.y, args.z)
    payload = {
        "query": {
            "x": args.x,
            "y": args.y,
            "z": args.z,
            "radius": args.radius,
            "distance_dims": graph.graph["config"]["distance_dims"],
        },
        "nearby_topologies": query_topologies(
            graph=graph,
            center=center,
            radius=args.radius,
            topology_limit=args.topology_limit,
            topology_index=topology_index,
        ),
        "nearest_waypoints": query_waypoints(
            graph=graph,
            center=center,
            radius=args.radius,
            waypoint_limit=args.waypoint_limit,
            waypoint_index=waypoint_index,
        ),
        "nearby_objects": query_objects(
            graph=graph,
            center=center,
            radius=args.radius,
            object_limit=args.object_limit,
            include_dynamic=args.include_dynamic,
            object_index=object_index,
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(mode: Optional[object] = None) -> int:
    """Run one of the three supported modes.

    When ``mode`` is provided, the function uses it directly.
    When ``mode`` is omitted, the script falls back to argv or the interactive
    menu-based selector.
    """

    args = resolve_args(mode)
    if args.command == "build":
        return build_graph(args)
    if args.command == "stats":
        return print_stats(args)
    if args.command == "nearby":
        return print_nearby(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
