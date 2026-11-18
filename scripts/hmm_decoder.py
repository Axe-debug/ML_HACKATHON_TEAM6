"""HMM and Viterbi Sequential Route Reconstruction for PathSnap."""
import numpy as np
import networkx as nx
from pathlib import Path

class ViterbiMapMatcher:
    def __init__(self, beta=15.0, direct_bonus=2.0, disconnect_penalty=-12.0, max_gap_s=30.0):
        self.beta = beta
        self.direct_bonus = direct_bonus
        self.disconnect_penalty = disconnect_penalty
        self.max_gap_s = max_gap_s

    @staticmethod
    def build_network_graph(nodes_xy, arcs):
        """Construct directed networkx graph with segment lengths in meters as edge weights."""
        G = nx.DiGraph()
        for arc_id, (fn, tn) in enumerate(arcs):
            p1 = nodes_xy[fn]
            p2 = nodes_xy[tn]
            dist = float(np.linalg.norm(p2 - p1))
            G.add_edge(int(fn), int(tn), arc_id=int(arc_id), weight=dist)
        return G

    def decode_trip(self, candidate_arc_ids, candidate_probs, arcs, nodes_xy, gps_xy, timestamps):
        """
        Viterbi algorithm to decode the optimal sequential path over K=8 candidate states.
        
        Parameters:
        - candidate_arc_ids: (T, K) integer arc IDs for each candidate
        - candidate_probs: (T, K) float probabilities from classifier
        - arcs: (M, 2) from_node, to_node integer table
        - nodes_xy: (N, 2) projected node coordinates in meters
        - gps_xy: (T, 2) projected GPS coordinates in meters
        - timestamps: (T,) GPS timestamps in seconds
        
        Returns:
        - decoded_arc_ids: (T,) sequence of optimal road segment arc IDs
        """
        T, K = candidate_arc_ids.shape
        if T == 0:
            return np.array([], dtype=int)
            
        G = self.build_network_graph(nodes_xy, arcs)
        
        # 1. Emission Log-Probabilities
        eps = 1e-12
        norm_probs = np.maximum(candidate_probs, eps)
        norm_probs /= norm_probs.sum(axis=1, keepdims=True)
        log_emissions = np.log(norm_probs) # (T, K)
        
        # 2. Viterbi Trellis
        V = np.zeros((T, K), dtype=float)
        backpointer = np.zeros((T, K), dtype=int)
        
        # Initialization at t=0
        V[0] = log_emissions[0]
        
        # Arc lookup tables for fast transition evaluation
        from_nodes = arcs[:, 0]
        to_nodes = arcs[:, 1]
        
        # Cache for shortest path queries between node pairs
        sp_cache = {}
        
        def get_shortest_path_dist(u_node, v_node):
            pair = (u_node, v_node)
            if pair in sp_cache:
                return sp_cache[pair]
            if u_node == v_node:
                sp_cache[pair] = 0.0
                return 0.0
            try:
                d = nx.shortest_path_length(G, source=u_node, target=v_node, weight='weight')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                d = float('inf')
            sp_cache[pair] = d
            return d

        # Precompute step displacements and time differences
        dts = np.diff(timestamps)
        step_dists = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)

        # 3. Dynamic Programming Forward Recurrence
        for t in range(1, T):
            dt = dts[t-1]
            gps_step = step_dists[t-1]
            prev_arcs = candidate_arc_ids[t-1]
            curr_arcs = candidate_arc_ids[t]
            
            # If large time gap, relax transition constraints
            is_large_gap = (dt > self.max_gap_s) or (dt <= 0)
            
            for j in range(K): # current state j (arc curr_arcs[j])
                curr_arc = curr_arcs[j]
                curr_from = from_nodes[curr_arc]
                
                log_trans = np.empty(K, dtype=float)
                for i in range(K): # previous state i (arc prev_arcs[i])
                    prev_arc = prev_arcs[i]
                    prev_to = to_nodes[prev_arc]
                    
                    if is_large_gap:
                        log_trans[i] = 0.0 # Uniform transition on sensor gaps
                    elif prev_arc == curr_arc:
                        # Vehicle continues along the exact same arc
                        log_trans[i] = self.direct_bonus - abs(gps_step) / self.beta
                    elif prev_to == curr_from:
                        # Direct connected topological adjacency
                        log_trans[i] = self.direct_bonus - abs(gps_step) / self.beta
                    else:
                        # Network shortest path distance from end of prev_arc to start of curr_arc
                        net_dist = get_shortest_path_dist(prev_to, curr_from)
                        if np.isfinite(net_dist) and net_dist <= 250.0:
                            diff = abs(net_dist - gps_step)
                            log_trans[i] = -diff / self.beta
                        else:
                            log_trans[i] = self.disconnect_penalty
                            
                # Viterbi step
                candidates_val = V[t-1] + log_trans
                best_prev = int(np.argmax(candidates_val))
                V[t, j] = candidates_val[best_prev] + log_emissions[t, j]
                backpointer[t, j] = best_prev
                
        # 4. Viterbi Backtrace
        decoded = np.empty(T, dtype=int)
        best_last = int(np.argmax(V[-1]))
        decoded[-1] = candidate_arc_ids[-1, best_last]
        
        curr_idx = best_last
        for t in range(T - 1, 0, -1):
            curr_idx = backpointer[t, curr_idx]
            decoded[t - 1] = candidate_arc_ids[t - 1, curr_idx]
            
        return decoded
