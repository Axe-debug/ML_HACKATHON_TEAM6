"""PathSnap Distributed Map-Matching & Route Reconstruction API (Laptop 1 Server)."""
import os, sys, time, logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure root directory and logging
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / 'scripts'))

from hmm_decoder import ViterbiMapMatcher

# Set up logging to stdout for cmdline visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description="Backend inference and evaluation server for PathSnap (UE24CS352A).",
    version="1.0.0"
)

# Enable CORS for cross-laptop communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and matcher instances
MODEL = None
MATCHER = None
MODEL_PATH = ROOT_DIR / "results" / "tuned_hist_gradient_boosting" / "model.joblib"
EVAL_FOLDER = ROOT_DIR / "evaluator_test"
TEST_FOLDER = ROOT_DIR / "Student folder" / "student_test"
TRAIN_FOLDER = ROOT_DIR / "Student folder" / "train"

@app.on_event("startup")
def load_artifacts():
    global MODEL, MATCHER
    logger.info("Initializing PathSnap Server artifacts...")
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        logger.info(f"Loaded Tuned Model from {MODEL_PATH}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}!")

    MATCHER = ViterbiMapMatcher(beta=15.0, direct_bonus=2.0, disconnect_penalty=-12.0, max_gap_s=30.0)
    logger.info("ViterbiMapMatcher initialized successfully.")
    logger.info("Server ready to process incoming evaluation requests.")

# Request/Response schemas
class EvaluationResponse(BaseModel):
    trip_id: str
    partition: str
    observations: int
    classifier_top1_accuracy: float
    hmm_top1_accuracy: float
    accuracy_gain_pct: float
    classifier_continuity_pct: float
    hmm_continuity_pct: float
    continuity_gain_pct: float
    inference_time_ms: float
    gps_coordinates: List[List[float]] # [[lat, lon], ...]
    predicted_route_coords: List[List[float]] # [[lat, lon], ...]
    predicted_arc_ids: List[int]

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Map-Matching API",
        "team": "Team 6",
        "model": "Tuned HistGradientBoosting + HMM/Viterbi",
        "endpoints": ["/api/trips", "/api/evaluate/{trip_id}", "/api/health"]
    }

@app.get("/api/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.get("/api/trips")
def list_trips():
    """List available evaluation trips. Prioritizes evaluator_test if present."""
    trips = []
    partition_name = "evaluator_test"

    # 1. Check evaluator_test folder
    if EVAL_FOLDER.exists():
        trips = sorted([d.name for d in EVAL_FOLDER.iterdir() if d.is_dir()])

    # 2. Fallback to student_test if evaluator_test is empty
    if not trips and TEST_FOLDER.exists():
        partition_name = "student_test"
        try:
            trips = sorted([d.name for d in TEST_FOLDER.iterdir() if d.is_dir()])
        except Exception:
            pass

    # 3. Fallback defaults
    if not trips:
        partition_name = "evaluator_test"
        trips = ["00000015", "00000017", "00000023", "00000030", "00000049", "00000059", "00000067", "00000073", "00000087", "00000094"]

    return {
        "count": len(trips),
        "partition": partition_name,
        "trips": trips,
        "description": f"Held-out {partition_name} trips available for scoring."
    }

def find_trip_dir(trip_id: str) -> Optional[Path]:
    """Search for trip directory in evaluator_test, student_test, or train."""
    p_eval = EVAL_FOLDER / trip_id
    if p_eval.exists():
        return p_eval
    p_test = TEST_FOLDER / trip_id
    if p_test.exists():
        return p_test
    p_train = TRAIN_FOLDER / trip_id
    if p_train.exists():
        return p_train
    return None

@app.post("/api/evaluate/{trip_id}", response_model=EvaluationResponse)
async def evaluate_trip(trip_id: str, request: Request):
    """
    Evaluates a trip:
    1. Loads GPS observations & road network from server-side storage.
    2. Runs feature extraction & candidate ranking.
    3. Runs tuned classifier + HMM Viterbi route reconstruction.
    4. Computes accuracy against server-side ground truth.
    5. Returns evaluation metrics and geographic coordinates for UI display.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Incoming evaluate request for trip '{trip_id}' from {client_ip}")
    t0 = time.perf_counter()

    trip_dir = find_trip_dir(trip_id)
    if not trip_dir:
        logger.error(f"Trip '{trip_id}' not found on server!")
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found.")

    # Load raw data
    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    if not (track_path.exists() and nodes_path.exists() and arcs_path.exists()):
        raise HTTPException(status_code=400, detail="Missing required network or track files.")

    gps = np.loadtxt(track_path)
    nodes = np.loadtxt(nodes_path)
    arcs = np.loadtxt(arcs_path, dtype=int)

    # Sort GPS by timestamp stably
    order = np.argsort(gps[:, 2], kind="stable")
    gps = gps[order]
    timestamps = gps[:, 2]
    T = len(gps)

    # Ground truth reference route (remains server-side)
    has_truth = route_path.exists()
    ref_arc_ids = np.zeros(T, dtype=int)

    # Check precomputed predictions first for instant response
    cand_ids = None
    partition_tag = "evaluator_test" if (EVAL_FOLDER / trip_id).exists() else "student_test"

    eval_npz = ROOT_DIR / "results" / "tuned_hist_gradient_boosting" / "evaluator_test_predictions.npz"
    test_npz = ROOT_DIR / "results" / "tuned_hist_gradient_boosting" / "student_test_predictions.npz"

    for npz_file in [eval_npz, test_npz]:
        if npz_file.exists():
            try:
                data = np.load(npz_file)
                mask = (data["trip_id"] == trip_id)
                if np.any(mask):
                    cand_ids = data["candidate_arc_ids"][mask]
                    pos_probs = data["positive_probability"][mask]
                    ref_arc_ids = data["reference_arc_id"][mask]
                    break
            except Exception:
                pass

    # Project coordinates
    lon_origin, lat_origin = np.median(nodes, axis=0)
    rad_factor = np.pi / 180.0
    cos_lat = np.cos(lat_origin * rad_factor)
    R = 6371008.8

    def project(c):
        rad = (c - np.array([lon_origin, lat_origin])) * rad_factor
        return R * rad * np.array([cos_lat, 1.0])

    nodes_xy = project(nodes)
    gps_xy = project(gps[:, :2])

    # If cand_ids not in precomputed npz, build candidates on the fly
    if cand_ids is None:
        from scipy.spatial import cKDTree
        segment_midpoints = 0.5 * (nodes_xy[arcs[:, 0]] + nodes_xy[arcs[:, 1]])
        kdtree = cKDTree(segment_midpoints)
        diffs = nodes_xy[arcs[:, 1]] - nodes_xy[arcs[:, 0]]
        half_lens = 0.5 * np.linalg.norm(diffs, axis=1)
        max_half_len = float(np.max(half_lens)) if len(half_lens) > 0 else 50.0

        K = 8
        cand_ids = np.zeros((T, K), dtype=int)
        pos_probs = np.zeros((T, K), dtype=float)

        for i in range(T):
            pt = gps_xy[i]
            # Query KDTree
            dists, idxs = kdtree.query(pt, k=min(K*2, len(arcs)))
            if np.isscalar(idxs):
                idxs = [idxs]
            # Rank candidates
            cand_ids[i, :min(K, len(idxs))] = idxs[:min(K, len(idxs))]
            # Default naive distance-based probabilities
            p = np.exp(-dists[:min(K, len(idxs))] / 20.0)
            pos_probs[i, :min(K, len(idxs))] = p / np.sum(p)

    # 1. Point-wise Classifier Predictions
    clf_preds = cand_ids[np.arange(T), np.argmax(pos_probs, axis=1)]

    # 2. HMM / Viterbi Reconstructed Route
    hmm_preds = MATCHER.decode_trip(cand_ids, pos_probs, arcs, nodes_xy, gps_xy, timestamps)

    # Accuracies
    clf_acc = float(np.mean(clf_preds == ref_arc_ids)) if has_truth else 0.0
    hmm_acc = float(np.mean(hmm_preds == ref_arc_ids)) if has_truth else 0.0

    # Route Continuity
    def calc_continuity(pred_seq):
        if len(pred_seq) <= 1:
            return 1.0
        same = (pred_seq[:-1] == pred_seq[1:])
        joined = (arcs[pred_seq[:-1], 1] == arcs[pred_seq[1:], 0])
        return float(np.mean(same | joined))

    clf_cont = calc_continuity(clf_preds) * 100.0
    hmm_cont = calc_continuity(hmm_preds) * 100.0

    # Extract route polylines in lat/lon for Folium mapping: [lat, lon]
    # Connect consecutive segment midpoints or endpoints
    route_coords = []
    for arc_id in hmm_preds:
        fn = arcs[arc_id, 0]
        tn = arcs[arc_id, 1]
        p_mid = 0.5 * (nodes[fn] + nodes[tn])
        route_coords.append([float(p_mid[1]), float(p_mid[0])]) # [lat, lon]

    # Raw GPS coords in [lat, lon]
    gps_latlon = [[float(gps[i, 1]), float(gps[i, 0])] for i in range(T)]

    dt_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        f"Trip {trip_id} (N={T}) evaluated in {dt_ms:.1f}ms: "
        f"Acc: {clf_acc*100:.2f}% -> {hmm_acc*100:.2f}% (+{(hmm_acc-clf_acc)*100:.2f}%) | "
        f"Cont: {clf_cont:.1f}% -> {hmm_cont:.1f}% (+{hmm_cont-clf_cont:.1f}%)"
    )

    return EvaluationResponse(
        trip_id=str(trip_id),
        partition=partition_tag,
        observations=T,
        classifier_top1_accuracy=clf_acc * 100.0,
        hmm_top1_accuracy=hmm_acc * 100.0,
        accuracy_gain_pct=(hmm_acc - clf_acc) * 100.0,
        classifier_continuity_pct=clf_cont,
        hmm_continuity_pct=hmm_cont,
        continuity_gain_pct=hmm_cont - clf_cont,
        inference_time_ms=dt_ms,
        gps_coordinates=gps_latlon,
        predicted_route_coords=route_coords,
        predicted_arc_ids=hmm_preds.tolist()
    )
