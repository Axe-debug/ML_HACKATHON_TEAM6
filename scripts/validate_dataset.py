"""Validate generated artifacts and produce a development Folium map."""
import csv, json, hashlib, sys
from pathlib import Path
import numpy as np
import folium

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import candidates, project_segments, FEATURES

root = Path('processed')
reports = Path('reports')
reports.mkdir(exist_ok=True)
figures = reports / 'figures'
figures.mkdir(exist_ok=True)

summary = {}
failures = []
fold_counts = {}
max_distortion = 0.0

for split, expected in [('train', 80), ('student_test', 10)]:
    files = sorted((root / split).glob('*.npz'))
    assert len(files) == expected, f"Expected {expected} files in {split}, found {len(files)}"
    stats = dict(trips=len(files), observations=0, candidate_rows=0, training_observations=0, reliable_labels=0, candidate_misses=0)
    for p in files:
        folder = Path('Student folder') / split / p.stem
        nodes = np.loadtxt(folder / f'{p.stem}.nodes')
        arcs = np.loadtxt(folder / f'{p.stem}.arcs', dtype=int)
        
        with np.load(p, allow_pickle=False) as d:
            X = d['X']
            y = d['y']
            n = len(X)
            assert X.shape == (n, 8, len(FEATURES)) and y.shape == (n, 8)
            assert np.isfinite(X).all()
            
            if 'gps' in d:
                gps = d['gps']
                assert np.all(np.diff(gps[:, 2]) >= 0)
            else:
                gps = np.loadtxt(folder / f'{p.stem}.track')
                assert np.all(np.diff(gps[:, 2]) >= 0)
                
            assert np.array_equal(y, d['candidate_arc_ids'] == d['reference_arc_id'][:, None])
            assert np.all(y.sum(axis=1) <= 1)
            assert np.all(np.diff(d['reference_occurrence']) >= 0)
            assert np.array_equal(d['training_observation_mask'], d['label_reliable'] & y.any(axis=1))
            assert np.all(np.diff(X[:, :, 0], axis=1) >= -1e-5)
            
            if 'original_row' in d:
                assert len(np.unique(d['original_row'])) == n
                
            fold = int(d['fold'])
            assert fold in (range(5) if split == 'train' else [-1])
            if split == 'train':
                fold_counts[fold] = fold_counts.get(fold, 0) + 1
            stats['observations'] += n
            stats['candidate_rows'] += n * 8
            stats['training_observations'] += int(d['training_observation_mask'].sum())
            stats['reliable_labels'] += int(d['label_reliable'].sum())
            stats['candidate_misses'] += int((~y.any(axis=1)).sum())

        if 'projection_origin_lonlat' in d:
            lon, lat = d['projection_origin_lonlat']
        else:
            lon, lat = np.median(nodes, axis=0)

        rad = np.deg2rad(nodes)
        a = rad[arcs[:, 0]]
        b = rad[arcs[:, 1]]
        diff = b - a
        h = np.sin(diff[:, 1] / 2) ** 2 + np.cos(a[:, 1]) * np.cos(b[:, 1]) * np.sin(diff[:, 0] / 2) ** 2
        geodesic = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        xy = 6371008.8 * np.deg2rad(nodes - [lon, lat]) * [np.cos(np.deg2rad(lat)), 1]
        planar = np.linalg.norm(xy[arcs[:, 1]] - xy[arcs[:, 0]], axis=1)
        good = geodesic > 1
        max_distortion = max(max_distortion, float(np.max(np.abs(planar[good] / geodesic[good] - 1))))
        
    stats['candidate_miss_rate'] = stats['candidate_misses'] / stats['observations']
    stats['fitting_coverage'] = stats['training_observations'] / stats['observations']
    summary[split] = stats

assert fold_counts == {i: 16 for i in range(5)}
summary['fold_trip_counts'] = fold_counts
summary['max_segment_distance_distortion_vs_haversine'] = max_distortion

# Independently check exact candidates against all road segments for ten real GPS points.
p = sorted((root / 'train').glob('*.npz'))[0]
folder = Path('Student folder/train') / p.stem
nodes = np.loadtxt(folder / f'{p.stem}.nodes')
arcs = np.loadtxt(folder / f'{p.stem}.arcs', dtype=int)
route = np.loadtxt(folder / f'{p.stem}.route', dtype=int)
gps = np.loadtxt(folder / f'{p.stem}.track')
lon, lat = np.median(nodes, axis=0)

with np.load(p) as d:
    xy_nodes = 6371008.8 * np.deg2rad(nodes - [lon, lat]) * [np.cos(np.deg2rad(lat)), 1]
    xy_gps = 6371008.8 * np.deg2rad(gps[:, :2] - [lon, lat]) * [np.cos(np.deg2rad(lat)), 1]
    sample_indices = np.linspace(0, len(d['X']) - 1, 10, dtype=int)
    points = xy_gps[sample_indices]
    ids, dist, _ = candidates(points, xy_nodes[arcs[:, 0]], xy_nodes[arcs[:, 1]])
    brute, _ = project_segments(points[:, None, :], xy_nodes[arcs[:, 0]][None, :, :], xy_nodes[arcs[:, 1]][None, :, :])
    np.testing.assert_allclose(dist, np.sort(brute, axis=1)[:, :8], atol=1e-7)
    summary['real_candidate_checks'] = 10

    # Generate Folium development map
    m = folium.Map(location=[float(gps[:, 1].mean()), float(gps[:, 0].mean())], zoom_start=13, tiles="OpenStreetMap")
    roads = folium.FeatureGroup(name='Road network (near observed trip)', show=True)
    selected = np.unique(d['candidate_arc_ids'])
    for arc in selected:
        folium.PolyLine(nodes[arcs[arc]][:, [1, 0]].tolist(), color='gray', weight=1, opacity=0.4).add_to(roads)
    roads.add_to(m)

    ref = folium.FeatureGroup(name='Supplied reference traversal (Ground Truth)', show=True)
    for arc in route:
        folium.PolyLine(nodes[arcs[arc]][:, [1, 0]].tolist(), color='green', weight=4, opacity=0.8).add_to(ref)
    ref.add_to(m)

    raw = folium.FeatureGroup(name='Raw GPS observations', show=True)
    for i, row in enumerate(gps):
        folium.CircleMarker([float(row[1]), float(row[0])], radius=2, color='blue', tooltip=f'GPS {i}').add_to(raw)
    raw.add_to(m)

    baseline = folium.FeatureGroup(name='Nearest-road baseline (untrained, disconnected segments possible)', show=False)
    for arc in np.unique(d['candidate_arc_ids'][:, 0]):
        folium.PolyLine(nodes[arcs[arc]][:, [1, 0]].tolist(), color='orange', weight=2).add_to(baseline)
    baseline.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(gps[:, 1].min()), float(gps[:, 0].min())], [float(gps[:, 1].max()), float(gps[:, 0].max())]])
    
    html_out = reports / f'development_trip_{p.stem}.html'
    m.save(str(html_out))
    print(f"Generated Folium map: {html_out}")

# Recheck provenance of every original input.
manifest = json.loads((root / 'raw_sha256.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((Path('Student folder') / name).read_bytes()).hexdigest() == digest
summary['raw_files_hash_verified'] = len(manifest)
summary['all_checks_passed'] = True

(reports / 'validation_summary.json').write_text(json.dumps(summary, indent=2))
print("Validation summary successfully updated:")
print(json.dumps(summary, indent=2))
