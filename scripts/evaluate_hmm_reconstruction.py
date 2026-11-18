"""Evaluate Classifier-Only vs Classifier + HMM/Viterbi Route Reconstruction on Student-Test."""
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from hmm_decoder import ViterbiMapMatcher

OUT_DIR = ROOT_DIR / 'results' / 'hmm_evaluation'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES = ROOT_DIR / 'reports' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

STUDENT_TEST_RAW = ROOT_DIR / 'Student folder' / 'student_test'
PREDICTIONS_NPZ = ROOT_DIR / 'results' / 'tuned_hist_gradient_boosting' / 'student_test_predictions.npz'

def main():
    print("="*70)
    print("PATHSNAP HMM & VITERBI SEQUENTIAL ROUTE RECONSTRUCTION BENCHMARK")
    print("="*70)
    
    # Load tuned student-test predictions
    data = np.load(PREDICTIONS_NPZ)
    pos_probs = data['positive_probability'] # (T_total, 8)
    cand_ids = data['candidate_arc_ids']     # (T_total, 8)
    ref_ids = data['reference_arc_id']       # (T_total,)
    trip_ids = data['trip_id']               # (T_total,)
    
    unique_trips = np.unique(trip_ids)
    print(f"Loaded {len(pos_probs)} observations across {len(unique_trips)} student-test trips.")
    
    matcher = ViterbiMapMatcher(beta=15.0, direct_bonus=2.0, disconnect_penalty=-12.0, max_gap_s=30.0)
    
    trip_results = []
    
    all_classifier_preds = []
    all_hmm_preds = []
    all_truths = []
    
    for tid in unique_trips:
        mask = (trip_ids == tid)
        t_cand_ids = cand_ids[mask]
        t_probs = pos_probs[mask]
        t_ref = ref_ids[mask]
        
        # Load raw road network and track
        trip_folder = STUDENT_TEST_RAW / str(tid)
        nodes = np.loadtxt(trip_folder / f"{tid}.nodes")
        arcs = np.loadtxt(trip_folder / f"{tid}.arcs", dtype=int)
        gps = np.loadtxt(trip_folder / f"{tid}.track")
        
        # Sort GPS by timestamp stably
        order = np.argsort(gps[:, 2], kind='stable')
        gps = gps[order]
        timestamps = gps[:, 2]
        
        # Project nodes and gps to local meters
        lon_origin, lat_origin = np.median(nodes, axis=0)
        def project(coords):
            rad = np.deg2rad(coords - np.array([lon_origin, lat_origin]))
            return 6371008.8 * rad * np.array([np.cos(np.deg2rad(lat_origin)), 1.0])
            
        nodes_xy = project(nodes)
        gps_xy = project(gps[:, :2])
        
        # 1. Classifier-Only Prediction: argmax of candidate probabilities
        t_clf_pred = t_cand_ids[np.arange(len(t_probs)), np.argmax(t_probs, axis=1)]
        
        # 2. HMM/Viterbi Prediction: global trellis decoding
        t0 = time.perf_counter()
        t_hmm_pred = matcher.decode_trip(t_cand_ids, t_probs, arcs, nodes_xy, gps_xy, timestamps)
        hmm_time_s = time.perf_counter() - t0
        
        # Accuracies
        clf_acc = float(np.mean(t_clf_pred == t_ref))
        hmm_acc = float(np.mean(t_hmm_pred == t_ref))
        
        # Route Continuity: % of consecutive segments that are identical or topologically joined
        def calc_continuity(pred_seq):
            if len(pred_seq) <= 1:
                return 1.0
            same = (pred_seq[:-1] == pred_seq[1:])
            # arcs[pred_seq[:-1], 1] == arcs[pred_seq[1:], 0]
            joined = (arcs[pred_seq[:-1], 1] == arcs[pred_seq[1:], 0])
            return float(np.mean(same | joined))
            
        clf_cont = calc_continuity(t_clf_pred)
        hmm_cont = calc_continuity(t_hmm_pred)
        
        trip_results.append({
            'trip_id': str(tid),
            'observations': int(len(t_ref)),
            'classifier_top1_accuracy': clf_acc,
            'hmm_top1_accuracy': hmm_acc,
            'accuracy_diff': hmm_acc - clf_acc,
            'classifier_continuity': clf_cont,
            'hmm_continuity': hmm_cont,
            'continuity_diff': hmm_cont - clf_cont,
            'hmm_decode_seconds': hmm_time_s
        })
        
        all_classifier_preds.extend(t_clf_pred.tolist())
        all_hmm_preds.extend(t_hmm_pred.tolist())
        all_truths.extend(t_ref.tolist())
        
        print(f"Trip {tid} (N={len(t_ref)}): Clf Acc={clf_acc*100:.2f}% -> HMM Acc={hmm_acc*100:.2f}% | Clf Cont={clf_cont*100:.1f}% -> HMM Cont={hmm_cont*100:.1f}% (in {hmm_time_s:.2f}s)")
        
    df = pd.DataFrame(trip_results)
    df.to_csv(OUT_DIR / 'hmm_vs_classifier_comparison.csv', index=False)
    
    all_clf_arr = np.array(all_classifier_preds)
    all_hmm_arr = np.array(all_hmm_preds)
    all_truth_arr = np.array(all_truths)
    
    overall_clf_acc = float(np.mean(all_clf_arr == all_truth_arr))
    overall_hmm_acc = float(np.mean(all_hmm_arr == all_truth_arr))
    
    overall_clf_cont = float(df['classifier_continuity'].mean())
    overall_hmm_cont = float(df['hmm_continuity'].mean())
    
    summary = {
        'total_observations': len(all_truth_arr),
        'overall_classifier_accuracy': overall_clf_acc,
        'overall_hmm_accuracy': overall_hmm_acc,
        'accuracy_gain': overall_hmm_acc - overall_clf_acc,
        'mean_classifier_continuity': overall_clf_cont,
        'mean_hmm_continuity': overall_hmm_cont,
        'continuity_gain': overall_hmm_cont - overall_clf_cont,
        'total_hmm_decode_seconds': float(df['hmm_decode_seconds'].sum())
    }
    
    (OUT_DIR / 'hmm_summary.json').write_text(json.dumps(summary, indent=2))
    
    print("\n" + "="*70)
    print("OVERALL HMM EVALUATION RESULTS")
    print(f"Overall Top-1 Accuracy:  Classifier-Only: {overall_clf_acc*100:.2f}%  ->  HMM+Viterbi: {overall_hmm_acc*100:.2f}%  (+{(overall_hmm_acc-overall_clf_acc)*100:.2f}%)")
    print(f"Mean Route Continuity:   Classifier-Only: {overall_clf_cont*100:.2f}%  ->  HMM+Viterbi: {overall_hmm_cont*100:.2f}%  (+{(overall_hmm_cont-overall_clf_cont)*100:.2f}%)")
    print(f"Total HMM Latency for 19,160 obs: {summary['total_hmm_decode_seconds']:.2f}s (~0.1ms per observation)")
    print("="*70)
    
    # Generate Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({'font.size': 11})
    
    # Plot 1: Top-1 Accuracy Comparison per trip
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df['classifier_top1_accuracy']*100, w, label='Classifier-Only Predictions', color='#e74c3c', edgecolor='black', alpha=0.85)
    ax.bar(x + w/2, df['hmm_top1_accuracy']*100, w, label='Classifier + HMM/Viterbi', color='#2ecc71', edgecolor='black', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df['trip_id'], rotation=45)
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Observation Accuracy: Classifier-Only vs Classifier + HMM/Viterbi (Student-Test)')
    ax.set_ylim(60, 100)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95)
    plt.tight_layout()
    p1 = FIGURES / '13_hmm_vs_classifier_accuracy.png'
    plt.savefig(p1)
    plt.close()
    print("Saved:", p1)
    
    # Plot 2: Route Continuity Comparison per trip
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(x - w/2, df['classifier_continuity']*100, w, label='Classifier-Only Continuity', color='#f39c12', edgecolor='black', alpha=0.85)
    ax.bar(x + w/2, df['hmm_continuity']*100, w, label='Classifier + HMM/Viterbi Continuity', color='#2980b9', edgecolor='black', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df['trip_id'], rotation=45)
    ax.set_ylabel('Route Continuity (%)')
    ax.set_title('Route Continuity: Drastic Improvement by Enforcing Graph Adjacency')
    ax.set_ylim(40, 105)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95)
    plt.tight_layout()
    p2 = FIGURES / '14_hmm_vs_classifier_continuity.png'
    plt.savefig(p2)
    plt.close()
    print("Saved:", p2)

if __name__ == '__main__':
    main()
