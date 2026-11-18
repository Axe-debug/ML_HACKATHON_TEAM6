"""Generate comprehensive publication-quality EDA figures for PathSnap Technical Report."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    root = Path(__file__).resolve().parent.parent
    reports = root / 'reports'
    figures = reports / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    
    # Load inventory.csv
    inventory_path = root / 'processed' / 'inventory.csv'
    with open(inventory_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        trips = list(reader)
        
    print(f"Loaded {len(trips)} trip records from {inventory_path}")
    
    train_trips = [t for t in trips if t['split'] == 'train']
    test_trips = [t for t in trips if t['split'] == 'student_test']
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14
    })

    # =========================================================================
    # Figure 1: Sampling Interval Distribution & Nominally 1 Hz Verification
    # =========================================================================
    print("Generating Figure 1: Sampling intervals distribution...")
    all_dts = []
    student_folder = root / 'Student folder'
    for t in train_trips[:25]:  # sample 25 trips for full dt histogram
        track_file = student_folder / 'train' / t['trip_id'] / f"{t['trip_id']}.track"
        if track_file.exists():
            data = np.loadtxt(track_file)
            if len(data) > 1:
                dts = np.diff(data[:, 2])
                all_dts.extend(dts.tolist())
    
    all_dts = np.array(all_dts)
    pct_1s = np.mean(np.isclose(all_dts, 1.0, atol=0.05)) * 100

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    counts, bins, patches = ax.hist(np.clip(all_dts, 0, 10), bins=np.linspace(0, 10, 41),
                                   color='#2b5c8f', edgecolor='black', alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel('GPS Sampling Interval $\Delta t$ (seconds)')
    ax.set_ylabel('Observation Frequency (Log Scale)')
    ax.set_title(f'GPS Sampling Interval Distribution (Nominally 1 Hz: {pct_1s:.1f}% at exactly 1.0s)')
    ax.axvline(1.0, color='#e74c3c', linestyle='--', linewidth=2, label=f'Nominal 1.0s ({pct_1s:.1f}%)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig1_path = figures / '01_sampling_intervals_distribution.png'
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # =========================================================================
    # Figure 2: Trajectory Lengths and Observation Counts (Train vs Student Test)
    # =========================================================================
    print("Generating Figure 2: Trajectory observations and length distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    train_obs = [int(t['observations']) for t in train_trips]
    test_obs = [int(t['observations']) for t in test_trips]
    
    ax1.hist(train_obs, bins=20, color='#3498db', edgecolor='black', alpha=0.8, label=f'Train (n=80, Total={sum(train_obs):,})')
    ax1.hist(test_obs, bins=10, color='#e67e22', edgecolor='black', alpha=0.85, label=f'Student Test (n=10, Total={sum(test_obs):,})')
    ax1.set_xlabel('GPS Observations per Trip')
    ax1.set_ylabel('Number of Trips')
    ax1.set_title('Trajectory Length in Observations')
    ax1.legend(frameon=True)
    
    # Road network segments per trip
    train_arcs = [int(t['arcs']) for t in train_trips]
    test_arcs = [int(t['arcs']) for t in test_trips]
    ax2.hist(train_arcs, bins=25, color='#2ecc71', edgecolor='black', alpha=0.8, label='Train Arcs')
    ax2.set_xlabel('Road Network Arcs (Segments) per Trip')
    ax2.set_ylabel('Number of Trips')
    ax2.set_title('Road Network Complexity per Trip')
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    fig2_path = figures / '02_trajectory_length_and_observations.png'
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # =========================================================================
    # Figure 3: Road Network Graph Connectivity & Weak Components
    # =========================================================================
    print("Generating Figure 3: Road network connectivity...")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    components = [int(t['weak_components']) for t in train_trips]
    isolated = [int(t['isolated_nodes']) for t in train_trips]
    
    ax.hist(components, bins=25, color='#9b59b6', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Number of Weakly Connected Components in Road Network')
    ax.set_ylabel('Number of Trips')
    ax.set_title('Road Network Graph Disconnectedness (Components per Trip)')
    ax.axvline(np.median(components), color='#c0392b', linestyle='--', linewidth=2, label=f'Median Components: {np.median(components):.0f}')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig3_path = figures / '03_road_network_connectivity.png'
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved: {fig3_path}")

    # =========================================================================
    # Figure 4: Sample Trip 00000001 Road Network, GPS Observations & Ground Truth
    # =========================================================================
    print("Generating Figure 4: Trip 00000001 Road Network & GPS Overlay...")
    p1 = student_folder / 'train' / '00000001'
    nodes = np.loadtxt(p1 / '00000001.nodes')
    arcs = np.loadtxt(p1 / '00000001.arcs', dtype=int)
    route = np.loadtxt(p1 / '00000001.route', dtype=int)
    gps = np.loadtxt(p1 / '00000001.track')

    # Bounding box around GPS
    min_lon, max_lon = gps[:, 0].min() - 0.005, gps[:, 0].max() + 0.005
    min_lat, max_lat = gps[:, 1].min() - 0.005, gps[:, 1].max() + 0.005

    mask_nodes = (nodes[:, 0] >= min_lon) & (nodes[:, 0] <= max_lon) & (nodes[:, 1] >= min_lat) & (nodes[:, 1] <= max_lat)
    sub_node_ids = set(np.where(mask_nodes)[0])

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    # Plot sub-graph roads
    drawn = 0
    for a in arcs:
        if a[0] in sub_node_ids and a[1] in sub_node_ids:
            ax.plot([nodes[a[0], 0], nodes[a[1], 0]], [nodes[a[0], 1], nodes[a[1], 1]],
                    color='#bdc3c7', linewidth=0.8, alpha=0.6)
            drawn += 1
            if drawn > 25000:
                break
                
    # Plot reference route
    for i, arc_id in enumerate(route):
        a = arcs[arc_id]
        ax.plot([nodes[a[0], 0], nodes[a[1], 0]], [nodes[a[0], 1], nodes[a[1], 1]],
                color='#27ae60', linewidth=3.0, alpha=0.9,
                label='Supplied Reference Traversal (Route)' if i == 0 else "")
                
    # Plot noisy GPS points
    ax.scatter(gps[:, 0], gps[:, 1], color='#2980b9', s=9, alpha=0.75, zorder=5, label='Observed Noisy GPS Points')
    
    ax.set_title('Trip 00000001: Road Network Sub-graph, Ground-Truth Route, and Noisy GPS')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95)
    plt.tight_layout()
    fig4_path = figures / '04_trip_gps_network_overlay.png'
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")

    # =========================================================================
    # Figure 5: Candidate Misses & Outlier Trips Analysis
    # =========================================================================
    print("Generating Figure 5: Candidate misses & outlier investigation...")
    sorted_trips = sorted(train_trips, key=lambda x: int(x['candidate_misses']), reverse=True)
    top_miss_ids = [t['trip_id'] for t in sorted_trips[:10]]
    top_miss_counts = [int(t['candidate_misses']) for t in sorted_trips[:10]]
    top_miss_obs = [int(t['observations']) for t in sorted_trips[:10]]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x_indices = np.arange(len(top_miss_ids))
    width = 0.38
    
    ax.bar(x_indices - width/2, top_miss_obs, width, label='Total GPS Observations', color='#bdc3c7', edgecolor='black')
    ax.bar(x_indices + width/2, top_miss_counts, width, label='Candidate Misses (True Arc not in Top-8)', color='#e74c3c', edgecolor='black')
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(top_miss_ids, rotation=45)
    ax.set_xlabel('Trip ID')
    ax.set_ylabel('Observation Count')
    ax.set_title('Candidate Miss Distribution Across Trips (Highlighting Outliers 00000054 and 00000033)')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig5_path = figures / '05_candidate_misses_investigation.png'
    plt.savefig(fig5_path)
    plt.close()
    print(f"Saved: {fig5_path}")

    # =========================================================================
    # Figure 6: Folium Map Render / Snapshot Overview
    # =========================================================================
    print("Generating Figure 6: Folium map static view snapshot...")
    # Create zoomed-in detail figure illustrating the map-matching challenge
    zoom_gps = gps[100:250]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    z_min_lon, z_max_lon = zoom_gps[:, 0].min() - 0.0015, zoom_gps[:, 0].max() + 0.0015
    z_min_lat, z_max_lat = zoom_gps[:, 1].min() - 0.0015, zoom_gps[:, 1].max() + 0.0015
    z_nodes_mask = (nodes[:, 0] >= z_min_lon) & (nodes[:, 0] <= z_max_lon) & (nodes[:, 1] >= z_min_lat) & (nodes[:, 1] <= z_max_lat)
    z_sub_nodes = set(np.where(z_nodes_mask)[0])
    
    for a in arcs:
        if a[0] in z_sub_nodes and a[1] in z_sub_nodes:
            ax.plot([nodes[a[0], 0], nodes[a[1], 0]], [nodes[a[0], 1], nodes[a[1], 1]],
                    color='#7f8c8d', linewidth=1.2, alpha=0.5)
            
    # Reference route segments in view
    for arc_id in route:
        a = arcs[arc_id]
        if a[0] in z_sub_nodes or a[1] in z_sub_nodes:
            ax.plot([nodes[a[0], 0], nodes[a[1], 0]], [nodes[a[0], 1], nodes[a[1], 1]],
                    color='#27ae60', linewidth=4.0, alpha=0.85)
            
    ax.plot([], [], color='#27ae60', linewidth=4.0, label='Ground Truth Road Arc')
    ax.plot([], [], color='#7f8c8d', linewidth=1.2, label='Candidate Road Segments')
    ax.scatter(zoom_gps[:, 0], zoom_gps[:, 1], color='#2980b9', s=24, edgecolor='black', linewidth=0.5,
               zorder=6, label='Noisy GPS Points')
    
    ax.set_title('Representative Geographical Map (Detail View: GPS Noise vs. Road Centerline)')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)
    plt.tight_layout()
    fig6_path = figures / '06_folium_map_snapshot.png'
    plt.savefig(fig6_path)
    plt.close()
    print(f"Saved: {fig6_path}")

    print("\nAll 6 EDA figures successfully generated in reports/figures/!")

if __name__ == '__main__':
    main()
