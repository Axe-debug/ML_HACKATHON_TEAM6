"""Evaluate Tuned HistGradientBoosting + HMM Viterbi on Evaluator-Test Partition."""
import sys, os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess import candidates, align_reference, project_segments, FEATURES
from hmm_decoder import ViterbiMapMatcher

EVAL_DIR = ROOT_DIR / "evaluator_test"
OUT_DIR = ROOT_DIR / "results" / "evaluator_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT_DIR / "results" / "tuned_hist_gradient_boosting" / "model.joblib"

def process_and_evaluate_trip(tid, model, matcher):
    folder = EVAL_DIR / tid
    gps_raw = np.loadtxt(folder / f"{tid}.track")
    nodes = np.loadtxt(folder / f"{tid}.nodes")
    arcs = np.loadtxt(folder / f"{tid}.arcs", dtype=int)
    route = np.loadtxt(folder / f"{tid}.route", dtype=int).ravel()
    
    # Sort GPS by timestamp stably
    order = np.argsort(gps_raw[:, 2], kind='stable')
    gps = gps_raw[order]
    timestamps = gps[:, 2]
    T = len(gps)
    
    # Project coordinates
    lon, lat = np.median(nodes, axis=0)
    def project(coords):
        radians = np.deg2rad(coords - np.array([lon, lat]))
        return 6371008.8 * radians * np.array([np.cos(np.deg2rad(lat)), 1.0])
        
    xy = project(gps[:, :2])
    nxy = project(nodes)
    
    starts = nxy[arcs[:, 0]]
    ends = nxy[arcs[:, 1]]
    delta = ends - starts
    length = np.linalg.norm(delta, axis=1)
    
    # Candidate arcs (K=8)
    cand_ids, distance, fraction = candidates(xy, starts, ends, k=8)
    
    # Reference alignment (proxy truth)
    ref_labels, occ, residual, margin = align_reference(xy, starts, ends, route)
    
    # 10D Features
    dt = np.r_[0, np.diff(gps[:, 2])]
    movement = np.vstack([np.zeros(2), np.diff(xy, axis=0)])
    step = np.linalg.norm(movement, axis=1)
    usable = (dt > 0) & (dt <= 30) & (step >= 2)
    speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    cos = np.sum(movement[:, None, :] * delta[cand_ids], axis=-1) / np.maximum(step[:, None] * length[cand_ids], 1e-12)
    cos = np.where(usable[:, None], cos, 0)
    
    shape = cand_ids.shape
    repeat = lambda x: np.broadcast_to(x[:, None], shape)
    X = np.stack([
        distance,
        cos,
        repeat(usable.astype(float)),
        length[cand_ids],
        fraction,
        repeat(speed),
        repeat(dt),
        repeat(step),
        np.broadcast_to(np.arange(shape[1]), shape),
        distance - distance[:, :1]
    ], axis=-1).astype(np.float32)
    
    # Reshape for scikit-learn model: (T*K, 10)
    X_flat = X.reshape(-1, 10)
    
    # Predict candidate probabilities
    probs_flat = model.predict_proba(X_flat)[:, 1] # positive class probability
    pos_probs = probs_flat.reshape(T, 8)
    
    # 1. Point-wise Classifier Predictions: argmax of candidate probabilities
    clf_pred = cand_ids[np.arange(T), np.argmax(pos_probs, axis=1)]
    
    # 2. HMM / Viterbi Reconstructed Route
    t0 = time.perf_counter()
    hmm_pred = matcher.decode_trip(cand_ids, pos_probs, arcs, nxy, xy, timestamps)
    hmm_time_s = time.perf_counter() - t0
    
    # Accuracies
    clf_acc = float(np.mean(clf_pred == ref_labels))
    hmm_acc = float(np.mean(hmm_pred == ref_labels))
    
    # Route Continuity
    def calc_continuity(pred_seq):
        if len(pred_seq) <= 1:
            return 1.0
        same = (pred_seq[:-1] == pred_seq[1:])
        joined = (arcs[pred_seq[:-1], 1] == arcs[pred_seq[1:], 0])
        return float(np.mean(same | joined))
        
    clf_cont = calc_continuity(clf_pred)
    hmm_cont = calc_continuity(hmm_pred)
    
    print(f"Trip {tid} (N={T:5d}): Clf Acc={clf_acc*100:6.2f}% -> HMM Acc={hmm_acc*100:6.2f}% (+{(hmm_acc-clf_acc)*100:5.2f}%) | "
          f"Clf Cont={clf_cont*100:5.1f}% -> HMM Cont={hmm_cont*100:5.1f}% (+{(hmm_cont-clf_cont)*100:5.1f}%) in {hmm_time_s:.2f}s")
          
    return {
        "trip_id": str(tid),
        "observations": T,
        "classifier_top1_accuracy": clf_acc,
        "hmm_top1_accuracy": hmm_acc,
        "accuracy_diff": hmm_acc - clf_acc,
        "classifier_continuity": clf_cont,
        "hmm_continuity": hmm_cont,
        "continuity_diff": hmm_cont - clf_cont,
        "hmm_decode_seconds": hmm_time_s,
        "X": X,
        "candidate_arc_ids": cand_ids,
        "positive_probability": pos_probs,
        "reference_arc_id": ref_labels,
        "classifier_predictions": clf_pred,
        "hmm_predictions": hmm_pred,
        "gps": gps,
        "nodes": nodes,
        "arcs": arcs
    }

def main():
    print("=" * 80)
    print("PATHSNAP EVALUATOR-TEST BENCHMARK (10 EVALUATOR TRIPS)")
    print("=" * 80)
    
    print(f"Loading tuned model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    matcher = ViterbiMapMatcher(beta=15.0, direct_bonus=2.0, disconnect_penalty=-12.0, max_gap_s=30.0)
    
    trips = sorted([d.name for d in EVAL_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(trips)} trips in evaluator_test: {trips}")
    
    trip_summaries = []
    all_cand_ids = []
    all_pos_probs = []
    all_ref_labels = []
    all_clf_preds = []
    all_hmm_preds = []
    all_trip_ids = []
    
    for tid in trips:
        res = process_and_evaluate_trip(tid, model, matcher)
        trip_summaries.append({
            "trip_id": res["trip_id"],
            "observations": res["observations"],
            "classifier_top1_accuracy": res["classifier_top1_accuracy"],
            "hmm_top1_accuracy": res["hmm_top1_accuracy"],
            "accuracy_diff": res["accuracy_diff"],
            "classifier_continuity": res["classifier_continuity"],
            "hmm_continuity": res["hmm_continuity"],
            "continuity_diff": res["continuity_diff"],
            "hmm_decode_seconds": res["hmm_decode_seconds"]
        })
        
        T = res["observations"]
        all_cand_ids.append(res["candidate_arc_ids"])
        all_pos_probs.append(res["positive_probability"])
        all_ref_labels.append(res["reference_arc_id"])
        all_clf_preds.append(res["classifier_predictions"])
        all_hmm_preds.append(res["hmm_predictions"])
        all_trip_ids.append(np.full(T, tid))
        
    df = pd.DataFrame(trip_summaries)
    df.to_csv(OUT_DIR / "evaluator_test_comparison.csv", index=False)
    
    # Save combined NPZ predictions for instantaneous API serving
    np.savez_compressed(
        ROOT_DIR / "results" / "tuned_hist_gradient_boosting" / "evaluator_test_predictions.npz",
        candidate_arc_ids=np.vstack(all_cand_ids),
        positive_probability=np.vstack(all_pos_probs),
        reference_arc_id=np.concatenate(all_ref_labels),
        classifier_predictions=np.concatenate(all_clf_preds),
        hmm_predictions=np.concatenate(all_hmm_preds),
        trip_id=np.concatenate(all_trip_ids)
    )
    
    total_obs = sum(df["observations"])
    overall_clf_acc = float(np.mean(np.concatenate(all_clf_preds) == np.concatenate(all_ref_labels)))
    overall_hmm_acc = float(np.mean(np.concatenate(all_hmm_preds) == np.concatenate(all_ref_labels)))
    mean_clf_cont = float(df["classifier_continuity"].mean())
    mean_hmm_cont = float(df["hmm_continuity"].mean())
    total_time = float(df["hmm_decode_seconds"].sum())
    
    summary = {
        "partition": "evaluator_test",
        "trip_count": len(trips),
        "total_observations": total_obs,
        "overall_classifier_accuracy": overall_clf_acc,
        "overall_hmm_accuracy": overall_hmm_acc,
        "accuracy_gain": overall_hmm_acc - overall_clf_acc,
        "mean_classifier_continuity": mean_clf_cont,
        "mean_hmm_continuity": mean_hmm_cont,
        "continuity_gain": mean_hmm_cont - mean_clf_cont,
        "total_decode_seconds": total_time
    }
    
    (OUT_DIR / "evaluator_test_summary.json").write_text(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 80)
    print("OVERALL EVALUATOR-TEST RESULTS")
    print(f"Total Observations:        {total_obs:,}")
    print(f"Overall Top-1 Accuracy:    Classifier: {overall_clf_acc*100:6.2f}%  ->  HMM+Viterbi: {overall_hmm_acc*100:6.2f}%  (+{(overall_hmm_acc-overall_clf_acc)*100:5.2f}%)")
    print(f"Mean Route Continuity:     Classifier: {mean_clf_cont*100:6.2f}%  ->  HMM+Viterbi: {mean_hmm_cont*100:6.2f}%  (+{(mean_hmm_cont-mean_clf_cont)*100:5.2f}%)")
    print(f"Total HMM Decoding Time:   {total_time:.2f}s (~0.1ms per observation)")
    print("=" * 80)

if __name__ == "__main__":
    main()
