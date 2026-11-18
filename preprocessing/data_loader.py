"""
PathSnap Dataset Loader and Validator
=====================================
Module for loading, parsing, validating, and converting raw PathSnap files
(.track, .nodes, .arcs, .route) into structured objects and NetworkX graphs.
Preserves trip IDs, partitions, and ensures strictly metric geodesic distances.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pyproj

# Standard WGS84 Geodetic Calculator for high-precision geodesic distances
_GEOD = pyproj.Geod(ellps="WGS84")
DEFAULT_DATA_ROOT = Path("data/raw/Student folder")


def calculate_geodesic_distance(
    lon1: Union[float, np.ndarray],
    lat1: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Calculate geodesic distance in meters on the WGS84 ellipsoid.
    
    Parameters
    ----------
    lon1, lat1 : float or array-like
        Starting coordinates in degrees.
    lon2, lat2 : float or array-like
        Ending coordinates in degrees.

    Returns
    -------
    np.ndarray
        Geodesic distances in meters.
    """
    lons1 = np.asarray(lon1, dtype=np.float64)
    lats1 = np.asarray(lat1, dtype=np.float64)
    lons2 = np.asarray(lon2, dtype=np.float64)
    lats2 = np.asarray(lat2, dtype=np.float64)

    if lons1.size == 0 or lons2.size == 0:
        return np.array([], dtype=np.float64)

    # pyproj.Geod.inv returns (forward_azimuth, back_azimuth, distance)
    _, _, dists = _GEOD.inv(lons1, lats1, lons2, lats2)
    return np.asarray(dists, dtype=np.float64)


@dataclass
class TripData:
    """
    Container for all data associated with a single PathSnap trip.

    Attributes
    ----------
    trip_id : str
        Unique identifier (e.g., '00000001').
    partition : str
        Partition name ('train' or 'student_test').
    track : pd.DataFrame
        Columns: ['lon', 'lat', 'timestamp', 'dist_m', 'dt_s', 'speed_mps', 'speed_kmh'].
    nodes : pd.DataFrame
        Columns: ['node_id', 'lon', 'lat'].
    arcs : pd.DataFrame
        Columns: ['arc_id', 'u', 'v', 'length_m'].
    route : pd.DataFrame
        Columns: ['step_idx', 'arc_id'].
    """

    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame
    _graph: Optional[nx.MultiDiGraph] = field(default=None, repr=False)

    @property
    def num_points(self) -> int:
        """Number of GPS observations."""
        return len(self.track)

    @property
    def num_nodes(self) -> int:
        """Number of road network nodes."""
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        """Number of road network arcs."""
        return len(self.arcs)

    @property
    def num_route_steps(self) -> int:
        """Number of ground-truth traversed arcs in route."""
        return len(self.route)

    @property
    def duration_seconds(self) -> float:
        """Duration of GPS track in seconds."""
        if len(self.track) < 2:
            return 0.0
        return float(self.track["timestamp"].iloc[-1] - self.track["timestamp"].iloc[0])

    @property
    def gps_distance_meters(self) -> float:
        """Total geodesic distance of GPS track in meters."""
        if "dist_m" in self.track.columns:
            return float(self.track["dist_m"].sum())
        return 0.0

    @property
    def route_distance_meters(self) -> float:
        """Total traversed road distance along the ground-truth route in meters."""
        if len(self.route) == 0:
            return 0.0
        route_arcs = self.arcs.iloc[self.route["arc_id"].values]
        if "length_m" in route_arcs.columns:
            return float(route_arcs["length_m"].sum())
        return 0.0

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Spatial bounding box: (min_lon, min_lat, max_lon, max_lat)."""
        if len(self.track) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(self.track["lon"].min()),
            float(self.track["lat"].min()),
            float(self.track["lon"].max()),
            float(self.track["lat"].max()),
        )

    def get_route_arcs(self) -> pd.DataFrame:
        """
        Return a detailed DataFrame of ground-truth traversed arcs with geometries.

        Columns:
        ['step_idx', 'arc_id', 'u', 'v', 'u_lon', 'u_lat', 'v_lon', 'v_lat', 'length_m']
        """
        if len(self.route) == 0:
            return pd.DataFrame(
                columns=[
                    "step_idx", "arc_id", "u", "v",
                    "u_lon", "u_lat", "v_lon", "v_lat", "length_m"
                ]
            )

        route_arc_ids = self.route["arc_id"].values
        sub_arcs = self.arcs.iloc[route_arc_ids].copy().reset_index(drop=True)
        sub_arcs["step_idx"] = self.route["step_idx"].values

        u_idx = sub_arcs["u"].values
        v_idx = sub_arcs["v"].values

        u_coords = self.nodes.iloc[u_idx][["lon", "lat"]].reset_index(drop=True)
        v_coords = self.nodes.iloc[v_idx][["lon", "lat"]].reset_index(drop=True)

        sub_arcs["u_lon"] = u_coords["lon"]
        sub_arcs["u_lat"] = u_coords["lat"]
        sub_arcs["v_lon"] = v_coords["lon"]
        sub_arcs["v_lat"] = v_coords["lat"]

        ordered_cols = [
            "step_idx", "arc_id", "u", "v",
            "u_lon", "u_lat", "v_lon", "v_lat", "length_m"
        ]
        return sub_arcs[ordered_cols]

    def to_networkx(self, multi: bool = True) -> Union[nx.MultiDiGraph, nx.DiGraph]:
        """
        Build a NetworkX directed graph representation of the trip's road network.

        Nodes carry attributes: `lon`, `lat`.
        Edges carry attributes: `arc_id`, `length_m`.
        """
        if self._graph is not None and multi:
            return self._graph

        G = nx.MultiDiGraph() if multi else nx.DiGraph()

        # Add all nodes with coordinate attributes
        for row in self.nodes.itertuples():
            G.add_node(row.node_id, lon=row.lon, lat=row.lat)

        # Add all directed arcs
        if multi:
            for row in self.arcs.itertuples():
                G.add_edge(
                    row.u,
                    row.v,
                    key=row.arc_id,
                    arc_id=row.arc_id,
                    length_m=row.length_m,
                )
        else:
            for row in self.arcs.itertuples():
                if not G.has_edge(row.u, row.v):
                    G.add_edge(
                        row.u,
                        row.v,
                        arc_id=row.arc_id,
                        length_m=row.length_m,
                    )

        if multi:
            self._graph = G
        return G

    def validate(self) -> Dict[str, Any]:
        """
        Perform rigorous data validation on the trip.

        Validates:
        - GPS coordinate ranges and absence of NaN/infinite values.
        - Timestamp strictly increasing / monotonic properties.
        - Timestamp gaps and missing records.
        - Arc node indices referencing valid nodes within [0, num_nodes - 1].
        - Route arc indices referencing valid arcs within [0, num_arcs - 1].
        - Route continuity (target node of arc i == source node of arc i+1).
        """
        # 1. NaN and Inf checks
        track_nan = int(self.track[["lon", "lat", "timestamp"]].isna().sum().sum())
        nodes_nan = int(self.nodes[["lon", "lat"]].isna().sum().sum())
        arcs_nan = int(self.arcs[["u", "v"]].isna().sum().sum())
        route_nan = int(self.route[["arc_id"]].isna().sum().sum())

        # 2. Coordinate range checks
        track_valid_coords = bool(
            (self.track["lon"] >= -180.0).all()
            and (self.track["lon"] <= 180.0).all()
            and (self.track["lat"] >= -90.0).all()
            and (self.track["lat"] <= 90.0).all()
        )
        nodes_valid_coords = bool(
            (self.nodes["lon"] >= -180.0).all()
            and (self.nodes["lon"] <= 180.0).all()
            and (self.nodes["lat"] >= -90.0).all()
            and (self.nodes["lat"] <= 90.0).all()
        )

        # 3. Timestamp checks
        ts = self.track["timestamp"].values
        dts = np.diff(ts) if len(ts) > 1 else np.array([])
        zero_dt = int((dts == 0).sum()) if len(dts) > 0 else 0
        neg_dt = int((dts < 0).sum()) if len(dts) > 0 else 0
        strictly_increasing = bool(neg_dt == 0 and zero_dt == 0)
        monotonic = bool(neg_dt == 0)

        dt_min = float(dts.min()) if len(dts) > 0 else 0.0
        dt_median = float(np.median(dts)) if len(dts) > 0 else 0.0
        dt_mean = float(dts.mean()) if len(dts) > 0 else 0.0
        dt_max = float(dts.max()) if len(dts) > 0 else 0.0

        gaps_gt_2s = int((dts > 2.0).sum()) if len(dts) > 0 else 0
        gaps_gt_5s = int((dts > 5.0).sum()) if len(dts) > 0 else 0
        gaps_gt_10s = int((dts > 10.0).sum()) if len(dts) > 0 else 0
        gaps_gt_60s = int((dts > 60.0).sum()) if len(dts) > 0 else 0

        # 4. Arc-to-node reference checks
        arcs_u = self.arcs["u"].values
        arcs_v = self.arcs["v"].values
        u_valid = (arcs_u >= 0) & (arcs_u < self.num_nodes)
        v_valid = (arcs_v >= 0) & (arcs_v < self.num_nodes)
        valid_arc_nodes = bool(u_valid.all() and v_valid.all())
        invalid_arc_nodes_count = int((~u_valid).sum() + (~v_valid).sum())

        # Self loops and multi-edges
        self_loops = int((arcs_u == arcs_v).sum())
        unique_arcs = len(self.arcs.drop_duplicates(subset=["u", "v"]))
        parallel_arcs = self.num_arcs - unique_arcs

        # 5. Route reference and continuity checks
        route_arc_ids = self.route["arc_id"].values
        r_valid = (route_arc_ids >= 0) & (route_arc_ids < self.num_arcs)
        valid_route_arcs = bool(r_valid.all())
        invalid_route_arcs_count = int((~r_valid).sum())

        route_discontinuities = 0
        route_continuity_rate = 100.0
        if len(route_arc_ids) > 1 and valid_route_arcs:
            curr_v = arcs_v[route_arc_ids[:-1]]
            next_u = arcs_u[route_arc_ids[1:]]
            route_discontinuities = int((curr_v != next_u).sum())
            total_transitions = len(route_arc_ids) - 1
            route_continuity_rate = float(
                (total_transitions - route_discontinuities) / total_transitions * 100.0
            )

        # 6. Overall validity flag
        is_valid = bool(
            track_nan == 0
            and nodes_nan == 0
            and arcs_nan == 0
            and route_nan == 0
            and track_valid_coords
            and nodes_valid_coords
            and monotonic
            and valid_arc_nodes
            and valid_route_arcs
        )

        return {
            "trip_id": self.trip_id,
            "partition": self.partition,
            "is_valid": is_valid,
            "track_valid_coords": track_valid_coords,
            "nodes_valid_coords": nodes_valid_coords,
            "nan_total": track_nan + nodes_nan + arcs_nan + route_nan,
            "nan_track": track_nan,
            "nan_nodes": nodes_nan,
            "nan_arcs": arcs_nan,
            "nan_route": route_nan,
            "strictly_increasing_ts": strictly_increasing,
            "monotonic_ts": monotonic,
            "zero_dt_count": zero_dt,
            "neg_dt_count": neg_dt,
            "dt_min": dt_min,
            "dt_median": dt_median,
            "dt_mean": dt_mean,
            "dt_max": dt_max,
            "gaps_gt_2s": gaps_gt_2s,
            "gaps_gt_5s": gaps_gt_5s,
            "gaps_gt_10s": gaps_gt_10s,
            "gaps_gt_60s": gaps_gt_60s,
            "valid_arc_nodes": valid_arc_nodes,
            "invalid_arc_nodes_count": invalid_arc_nodes_count,
            "self_loops": self_loops,
            "parallel_arcs": parallel_arcs,
            "valid_route_arcs": valid_route_arcs,
            "invalid_route_arcs_count": invalid_route_arcs_count,
            "route_discontinuities": route_discontinuities,
            "route_continuity_rate": route_continuity_rate,
        }


def load_trip(
    partition: str,
    trip_id: str,
    base_dir: Union[str, Path] = DEFAULT_DATA_ROOT,
    compute_metrics: bool = True,
) -> TripData:
    """
    Load all files for a specific trip into a TripData object.

    Parameters
    ----------
    partition : str
        'train' or 'student_test'.
    trip_id : str
        Trip identifier string (e.g., '00000001').
    base_dir : str or Path
        Root path to raw student folder.
    compute_metrics : bool
        Whether to calculate geodesic distances, intervals and speeds.

    Returns
    -------
    TripData
    """
    base_dir = Path(base_dir)
    trip_dir = base_dir / partition / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    for f_path, label in [
        (track_file, "track"),
        (nodes_file, "nodes"),
        (arcs_file, "arcs"),
        (route_file, "route"),
    ]:
        if not f_path.is_file():
            raise FileNotFoundError(f"Missing {label} file for trip {trip_id}: {f_path}")

    # 1. Parse track: lon, lat, timestamp
    track = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64},
        engine="c",
    )

    # 2. Parse nodes: lon, lat
    nodes = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
        engine="c",
    )
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))

    # 3. Parse arcs: u, v
    arcs = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["u", "v"],
        dtype={"u": np.int64, "v": np.int64},
        engine="c",
    )
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int64))

    # 4. Parse route: arc_id
    route = pd.read_csv(
        route_file,
        sep=r"\s+",
        header=None,
        names=["arc_id"],
        dtype={"arc_id": np.int64},
        engine="c",
    )
    route.insert(0, "step_idx", np.arange(len(route), dtype=np.int64))

    # Compute metric distances and speeds if requested
    if compute_metrics:
        # Arc lengths
        u_coords = nodes.iloc[arcs["u"].values][["lon", "lat"]].values
        v_coords = nodes.iloc[arcs["v"].values][["lon", "lat"]].values
        arc_lengths = calculate_geodesic_distance(
            u_coords[:, 0], u_coords[:, 1], v_coords[:, 0], v_coords[:, 1]
        )
        arcs["length_m"] = arc_lengths

        # Track point-to-point geodesic distances, dt, and speed
        if len(track) > 1:
            p_lons = track["lon"].values
            p_lats = track["lat"].values
            p_ts = track["timestamp"].values

            dists = calculate_geodesic_distance(
                p_lons[:-1], p_lats[:-1], p_lons[1:], p_lats[1:]
            )
            dts = np.diff(p_ts)
            dts_safe = np.maximum(dts, 1e-6)
            speed_mps = dists / dts_safe
            speed_kmh = speed_mps * 3.6

            # Prepend zero for first point
            track["dist_m"] = np.insert(dists, 0, 0.0)
            track["dt_s"] = np.insert(dts, 0, 0.0)
            track["speed_mps"] = np.insert(speed_mps, 0, 0.0)
            track["speed_kmh"] = np.insert(speed_kmh, 0, 0.0)
        else:
            track["dist_m"] = 0.0
            track["dt_s"] = 0.0
            track["speed_mps"] = 0.0
            track["speed_kmh"] = 0.0
    else:
        arcs["length_m"] = 0.0
        track["dist_m"] = 0.0
        track["dt_s"] = 0.0
        track["speed_mps"] = 0.0
        track["speed_kmh"] = 0.0

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )


def list_trips(
    partition: Optional[str] = None,
    base_dir: Union[str, Path] = DEFAULT_DATA_ROOT,
) -> List[Tuple[str, str]]:
    """
    List all available trip identifiers and their partitions.

    Parameters
    ----------
    partition : Optional[str]
        If specified ('train' or 'student_test'), filter by partition.
    base_dir : str or Path
        Base raw data folder.

    Returns
    -------
    List[Tuple[str, str]]
        List of (partition, trip_id) tuples sorted chronologically/lexicographically.
    """
    base_dir = Path(base_dir)
    partitions = [partition] if partition else ["train", "student_test"]
    trips: List[Tuple[str, str]] = []

    for part in partitions:
        part_dir = base_dir / part
        if not part_dir.is_dir():
            continue
        subdirs = sorted([d.name for d in part_dir.iterdir() if d.is_dir()])
        for trip_id in subdirs:
            trips.append((part, trip_id))

    return trips


def iter_trips(
    partition: Optional[str] = None,
    base_dir: Union[str, Path] = DEFAULT_DATA_ROOT,
    compute_metrics: bool = True,
) -> Generator[TripData, None, None]:
    """
    Generator yielding TripData objects one by one to keep memory minimal.
    """
    trips = list_trips(partition=partition, base_dir=base_dir)
    for part, trip_id in trips:
        yield load_trip(
            partition=part,
            trip_id=trip_id,
            base_dir=base_dir,
            compute_metrics=compute_metrics,
        )


def load_all_trips(
    partition: Optional[str] = None,
    base_dir: Union[str, Path] = DEFAULT_DATA_ROOT,
    compute_metrics: bool = True,
) -> Dict[str, TripData]:
    """
    Load all trips into a dictionary keyed by trip_id.
    """
    all_data = {}
    for trip in iter_trips(
        partition=partition, base_dir=base_dir, compute_metrics=compute_metrics
    ):
        all_data[trip.trip_id] = trip
    return all_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PathSnap Data Loader Test CLI")
    parser.add_argument("--trip_id", type=str, default="00000001", help="Trip ID to load")
    parser.add_argument("--partition", type=str, default="train", help="train or student_test")
    parser.add_argument(
        "--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Path to raw dataset"
    )
    args = parser.parse_args()

    print(f"Loading {args.partition}/{args.trip_id} from {args.data_root}...")
    trip = load_trip(args.partition, args.trip_id, base_dir=args.data_root)
    print(f"Loaded Trip {trip.trip_id} successfully!")
    print(f"  GPS points: {trip.num_points:,}")
    print(f"  Duration: {trip.duration_seconds:.1f} s ({trip.duration_seconds/60:.1f} min)")
    print(f"  GPS distance: {trip.gps_distance_meters/1000.0:.2f} km")
    print(f"  Route distance: {trip.route_distance_meters/1000.0:.2f} km")
    print(f"  Road nodes: {trip.num_nodes:,}")
    print(f"  Road arcs: {trip.num_arcs:,}")
    print(f"  Route arc count: {trip.num_route_steps:,}")
    print(f"  Bounds: {trip.bounds}")

    val = trip.validate()
    print("Validation Result:")
    for k, v in val.items():
        print(f"  {k}: {v}")
