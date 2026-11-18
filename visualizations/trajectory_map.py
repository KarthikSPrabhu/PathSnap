"""
PathSnap Interactive Trajectory and Road Network Map Generator
==============================================================
Generates interactive, multi-layer Folium maps rendering:
- Raw GPS trajectory points (with timestamps, speed, and coordinates in popups)
- GPS trajectory polyline
- Ground-truth road network route (reconstructed from traversed arc geometries)
- Surrounding road network subgraph (alternative intersections, turns, and ramps)
- Trip Start and End markers
- Layer controls and legend
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import folium
from folium.plugins import Fullscreen, MiniMap, MeasureControl
import numpy as np
import pandas as pd

from preprocessing.data_loader import DEFAULT_DATA_ROOT, TripData, load_trip


def get_corridor_arcs(
    trip: TripData,
    hops: int = 2,
    max_arcs: int = 3000,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the topological road corridor surrounding the ground truth route.
    
    Includes the k-hop neighborhood of all traversed route nodes, capturing
    candidate alternative roads, intersections, on-ramps, and parallel lanes.
    """
    if trip.num_route_steps == 0:
        # Fallback: take a subset of arcs closest to GPS track
        return np.array([]), pd.DataFrame()

    route_arc_ids = trip.route["arc_id"].values
    arcs_array = trip.arcs[["u", "v"]].values

    # Start with nodes traversed by the route
    route_nodes: Set[int] = set(arcs_array[route_arc_ids, 0]).union(
        set(arcs_array[route_arc_ids, 1])
    )
    current_nodes = set(route_nodes)

    # Expand by k-hops in the road network graph
    for _ in range(hops):
        mask = np.isin(arcs_array[:, 0], list(current_nodes)) | np.isin(
            arcs_array[:, 1], list(current_nodes)
        )
        expanded_nodes = set(arcs_array[mask, 0]).union(set(arcs_array[mask, 1]))
        current_nodes.update(expanded_nodes)
        if len(current_nodes) > max_arcs:
            break

    # Select all arcs where both or either node is in current_nodes
    corridor_mask = np.isin(arcs_array[:, 0], list(current_nodes)) & np.isin(
        arcs_array[:, 1], list(current_nodes)
    )
    corridor_indices = np.where(corridor_mask)[0]

    if len(corridor_indices) > max_arcs:
        corridor_indices = corridor_indices[:max_arcs]

    sub_arcs = trip.arcs.iloc[corridor_indices].copy()
    u_coords = trip.nodes.iloc[sub_arcs["u"].values][["lon", "lat"]].values
    v_coords = trip.nodes.iloc[sub_arcs["v"].values][["lon", "lat"]].values
    sub_arcs["u_lon"] = u_coords[:, 0]
    sub_arcs["u_lat"] = u_coords[:, 1]
    sub_arcs["v_lon"] = v_coords[:, 0]
    sub_arcs["v_lat"] = v_coords[:, 1]

    return corridor_indices, sub_arcs


def generate_trip_map(
    trip: TripData,
    output_html_path: Union[str, Path],
    corridor_hops: int = 2,
    max_network_arcs: int = 4000,
    point_sample_rate: int = 1,
) -> folium.Map:
    """
    Generate an interactive Folium map for the given TripData.
    """
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    track = trip.track
    if len(track) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no GPS track points to visualize.")

    # Center map on median GPS coordinate
    center_lat = float(track["lat"].median())
    center_lon = float(track["lon"].median())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Road Network Corridor Layer
    fg_network = folium.FeatureGroup(name="Road Network (Corridor Arcs)", show=True)
    _, corridor_arcs_df = get_corridor_arcs(trip, hops=corridor_hops, max_arcs=max_network_arcs)

    route_arc_set = set(trip.route["arc_id"].values)
    for row in corridor_arcs_df.itertuples():
        if row.arc_id in route_arc_set:
            continue  # Drawn separately with distinct styling in the Route layer
        folium.PolyLine(
            locations=[[row.u_lat, row.u_lon], [row.v_lat, row.v_lon]],
            color="#7f8c8d",
            weight=2.0,
            opacity=0.55,
            tooltip=f"Road Arc #{row.arc_id} (Length: {row.length_m:.1f}m)",
        ).add_to(fg_network)
    fg_network.add_to(fmap)

    # 2. Ground-Truth Traversed Route Layer
    fg_route = folium.FeatureGroup(name="Ground-Truth Route (Traversed Arcs)", show=True)
    route_arcs_df = trip.get_route_arcs()
    for row in route_arcs_df.itertuples():
        folium.PolyLine(
            locations=[[row.u_lat, row.u_lon], [row.v_lat, row.v_lon]],
            color="#2980b9",
            weight=5.0,
            opacity=0.90,
            tooltip=f"Route Step {row.step_idx}: Arc #{row.arc_id} ({row.length_m:.1f}m)",
        ).add_to(fg_route)
    fg_route.add_to(fmap)

    # 3. GPS Trajectory Line Layer
    fg_track_line = folium.FeatureGroup(name="GPS Trajectory Line", show=True)
    gps_coords = track[["lat", "lon"]].values.tolist()
    folium.PolyLine(
        locations=gps_coords,
        color="#e74c3c",
        weight=2.5,
        opacity=0.75,
        dash_array="6, 6",
        tooltip=f"GPS Path ({trip.gps_distance_meters/1000.0:.2f} km, {len(track)} points)",
    ).add_to(fg_track_line)
    fg_track_line.add_to(fmap)

    # 4. GPS Observation Points Layer
    fg_track_pts = folium.FeatureGroup(name="GPS Observation Points", show=True)
    sampled_track = track.iloc[::point_sample_rate].reset_index(drop=True)

    speeds = sampled_track["speed_kmh"].values if "speed_kmh" in sampled_track.columns else np.zeros(len(sampled_track))
    max_s = max(float(np.max(speeds)), 30.0)

    for i, row in sampled_track.iterrows():
        spd = float(row.get("speed_kmh", 0.0))
        # Color coding: Green (slow/moderate) -> Yellow/Orange -> Red (high speed)
        ratio = min(spd / max_s, 1.0)
        if ratio < 0.33:
            pt_color = "#27ae60"  # Green
        elif ratio < 0.66:
            pt_color = "#f39c12"  # Orange
        else:
            pt_color = "#e74c3c"  # Red

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; line-height: 1.4; width: 180px;">
            <b style="color: #2c3e50;">GPS Point #{i*point_sample_rate}</b><br/>
            <b>Time:</b> {row['timestamp']:.1f} s<br/>
            <b>Speed:</b> {spd:.1f} km/h<br/>
            <b>Dist from prev:</b> {row.get('dist_m', 0.0):.1f} m<br/>
            <b>Lat:</b> {row['lat']:.6f}<br/>
            <b>Lon:</b> {row['lon']:.6f}
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4.0,
            color="#2c3e50",
            weight=1,
            fill=True,
            fill_color=pt_color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"Point #{i*point_sample_rate}: {spd:.1f} km/h (t={row['timestamp']:.1f}s)",
        ).add_to(fg_track_pts)
    fg_track_pts.add_to(fmap)

    # 5. Start and End Markers Layer
    fg_markers = folium.FeatureGroup(name="Start & End Landmarks", show=True)
    start_pt = track.iloc[0]
    end_pt = track.iloc[-1]

    folium.Marker(
        location=[start_pt["lat"], start_pt["lon"]],
        popup=f"<b>Trip Start</b><br/>Trip ID: {trip.trip_id}<br/>Time: {start_pt['timestamp']:.1f}s",
        tooltip="Trip Origin (Start)",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fg_markers)

    folium.Marker(
        location=[end_pt["lat"], end_pt["lon"]],
        popup=f"<b>Trip End</b><br/>Trip ID: {trip.trip_id}<br/>Time: {end_pt['timestamp']:.1f}s<br/>Total dist: {trip.gps_distance_meters/1000.0:.2f} km",
        tooltip="Trip Destination (End)",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(fg_markers)
    fg_markers.add_to(fmap)

    # 6. Fit map bounds to encompass trajectory
    fmap.fit_bounds([
        [float(track["lat"].min()), float(track["lon"].min())],
        [float(track["lat"].max()), float(track["lon"].max())],
    ])

    # 7. Add Interactive Plugins
    Fullscreen(position="topleft").add_to(fmap)
    MiniMap(toggle_display=True, position="bottomleft").add_to(fmap)
    MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(fmap)
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    # 8. Floating Informational Legend
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; 
        right: 30px; 
        width: 260px; 
        height: auto; 
        background-color: white; 
        z-index: 9999; 
        font-family: Arial; 
        font-size: 12px; 
        border: 2px solid #bdc3c7; 
        border-radius: 8px; 
        padding: 12px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
    ">
        <h4 style="margin: 0 0 8px 0; color: #2c3e50; font-size: 13px; font-weight: bold;">
            Trip {trip.trip_id} ({trip.partition})
        </h4>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:18px; height:4px; background:#2980b9; vertical-align:middle; margin-right:6px;"></span>
            <b>Ground-Truth Route:</b> {trip.route_distance_meters/1000.0:.2f} km
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:18px; height:3px; border-top:2px dashed #e74c3c; vertical-align:middle; margin-right:6px;"></span>
            <b>GPS Track:</b> {trip.gps_distance_meters/1000.0:.2f} km ({trip.num_points} pts)
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:18px; height:2px; background:#7f8c8d; vertical-align:middle; margin-right:6px;"></span>
            <b>Corridor Road Network</b>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#27ae60; vertical-align:middle; margin-right:6px;"></span>
            Low Speed / Urban
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#e74c3c; vertical-align:middle; margin-right:6px;"></span>
            High Speed / Highway
        </div>
        <hr style="margin: 8px 0; border: none; border-top: 1px solid #ecf0f1;"/>
        <div style="font-size: 11px; color: #7f8c8d;">
            Duration: {trip.duration_seconds/60.0:.1f} min | Road Nodes: {trip.num_nodes:,}
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Save to file
    fmap.save(str(output_html_path))
    print(f"Interactive trajectory map saved to: {output_html_path}")
    return fmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap Interactive Trajectory Map Builder")
    parser.add_argument("--trip_id", type=str, default="00000001", help="Trip ID to map")
    parser.add_argument("--partition", type=str, default="train", help="train or student_test")
    parser.add_argument(
        "--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Path to raw student folder"
    )
    parser.add_argument(
        "--output_dir", type=str, default="visualizations", help="Output directory for HTML maps"
    )
    parser.add_argument(
        "--corridor_hops", type=int, default=2, help="Network neighborhood hops for corridor"
    )
    parser.add_argument(
        "--max_arcs", type=int, default=4000, help="Maximum background corridor arcs"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=1, help="Subsample GPS points (1 = every point)"
    )
    args = parser.parse_args()

    print(f"Loading {args.partition}/{args.trip_id}...")
    trip_obj = load_trip(args.partition, args.trip_id, base_dir=args.data_root)

    out_file = Path(args.output_dir) / f"trajectory_map_trip_{args.trip_id}.html"
    generate_trip_map(
        trip=trip_obj,
        output_html_path=out_file,
        corridor_hops=args.corridor_hops,
        max_network_arcs=args.max_arcs,
        point_sample_rate=args.sample_rate,
    )

    # Also duplicate into results/ for convenience
    res_copy = Path("results") / f"trajectory_map_trip_{args.trip_id}.html"
    res_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_file, res_copy)
    print(f"Duplicated to: {res_copy}")
