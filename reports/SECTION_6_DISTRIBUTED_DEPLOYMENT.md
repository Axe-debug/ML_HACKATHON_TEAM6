# Technical Report — Section 6: Two-Laptop Distributed Deployment & Evaluation System

**UE24CS352A: Machine Learning — PathSnap Distributed Map-Matching & Route Reconstruction**  
**Team 6:** Yaswanth, Madhu, Hrisheek, Joshwin  
**Architecture:** Distributed Client-Server over Local Area Network (FastAPI + Uvicorn + Leaflet / Streamlit).  
**Source Code:** `api/main.py`, `ui/index.html`, `ui/app.py`.

---

## 6.1 Architectural Overview: Two-Laptop Distributed Setup

The PathSnap hackathon guidelines mandate a distributed two-laptop deployment to emulate a production intelligent transportation system (ITS):
* **Laptop 1 (Inference & Evaluation Server):** Runs the high-performance FastAPI backend. Houses the trained ML models, HMM sequential decoding engine, and strictly guarded server-side evaluation data and reference route files.
* **Laptop 2 (Evaluator Client Interface):** Runs the evaluator interface (interactive HTML5/Tailwind/Leaflet dashboard or Streamlit UI). It allows the evaluator to browse available test trips, submit evaluation triggers, view real-time scorecards, and inspect the reconstructed route visually on an interactive map.

```
┌────────────────────────────────────────┐          HTTP / JSON          ┌────────────────────────────────────────┐
│               LAPTOP 2                 │ ◄───────────────────────────► │               LAPTOP 1                 │
│         (Evaluator Client UI)          │     Port 8000 (LAN / Wi-Fi)   │      (FastAPI Inference Server)        │
├────────────────────────────────────────┤                               ├────────────────────────────────────────┤
│ • Interactive Web UI (Leaflet / HTML5) │                               │ • FastAPI REST API (`api/main.py`)     │
│ • Streamlit Client (`ui/app.py`)       │                               │ • Tuned HistGradientBoosting Model     │
│ • Trip Selector Dropdown               │                               │ • HMM / Viterbi Sequential Decoder     │
│ • Real-time Metrics Scorecard          │                               │ • Strictly Guarded Server-Side Routes  │
│ • Trajectory Polyline Visualization    │                               │ • Live Standard Output CMD Logs        │
└────────────────────────────────────────┘                               └────────────────────────────────────────┘
```

---

## 6.2 Strict Academic Integrity & Zero Data Leakage Guarantee

To comply with the Assessment & Evaluation policy:
1. **Zero Client Data Leakage:** The 10 reserved evaluator testing trips and ground-truth `.route` files reside exclusively on Laptop 1. The client UI on Laptop 2 **never receives raw reference route files or ground-truth arrays**.
2. **Server-Side Scoring:** Metrics (Observation Top-1 Accuracy and Route Continuity) are computed strictly on the backend server. The client receives only aggregated metrics and the predicted geographic route polyline coordinates for map rendering.
3. **Transparent Live Execution:** Every prediction and dynamic programming step executes live upon request. No pre-rendered prediction files or hardcoded static lookup responses are used. All actions log live to stdout in real time.

---

## 6.3 RESTful API Specification (Laptop 1 Server)

The FastAPI server (`api/main.py`) exposes three RESTful endpoints with full Cross-Origin Resource Sharing (CORS) support:

### 1. `GET /api/health`
Checks server availability and artifact loading status.
* **Request:** `GET http://<laptop1_ip>:8000/api/health`
* **Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. `GET /api/trips`
Lists all available evaluation trips for the UI dropdown selector.
* **Request:** `GET http://<laptop1_ip>:8000/api/trips`
* **Response:**
```json
{
  "count": 10,
  "trips": ["00000000", "00000007", "00000009", "00000040", "00000052", "00000056", "00000065", "00000076", "00000077", "00000091"],
  "description": "Held-out evaluation trips available for scoring."
}
```

### 3. `POST /api/evaluate/{trip_id}`
Primary inference and scoring endpoint. Accepts a `trip_id`, runs point-wise classification and Viterbi route reconstruction, computes metrics against server-side ground truth, and returns evaluation results and map coordinates.
* **Request:** `POST http://<laptop1_ip>:8000/api/evaluate/00000052`
* **Response Payload:**
```json
{
  "trip_id": "00000052",
  "observations": 262,
  "classifier_top1_accuracy": 91.98,
  "hmm_top1_accuracy": 96.95,
  "accuracy_gain_pct": 4.96,
  "classifier_continuity_pct": 91.19,
  "hmm_continuity_pct": 96.17,
  "continuity_gain_pct": 4.98,
  "inference_time_ms": 1032.4,
  "gps_coordinates": [[48.1402, 11.5621], [48.1404, 11.5623], "..."],
  "predicted_route_coords": [[48.1401, 11.5620], [48.1405, 11.5625], "..."],
  "predicted_arc_ids": [482, 482, 483, 483, 501, "..."]
}
```

---

## 6.4 Live Command-Line Logging & Verifiability

When an evaluation request is triggered from Laptop 2, Laptop 1 emits standardized, human-readable log entries directly to the console in real time:

```
2026-11-18 14:26:01,102 [INFO] Initializing PathSnap Server artifacts...
2026-11-18 14:26:01,450 [INFO] Loaded Tuned Model from results/tuned_hist_gradient_boosting/model.joblib
2026-11-18 14:26:01,452 [INFO] ViterbiMapMatcher initialized successfully.
2026-11-18 14:26:01,455 [INFO] Server ready to process incoming evaluation requests.
2026-11-18 14:27:15,310 [INFO] Incoming evaluate request for trip '00000052' from 192.168.1.105
2026-11-18 14:27:16,342 [INFO] Trip 00000052 (N=262) evaluated in 1032.4ms: Acc: 91.98% -> 96.95% (+4.96%) | Cont: 91.2% -> 96.2% (+5.0%)
```

This ensures that evaluators can verify on the terminal that inference is executing dynamically.

---

## 6.5 Evaluator Client UI (Laptop 2)

We provide two distinct client UI options on Laptop 2:

### Option A: Lightweight Web Dashboard (`ui/index.html`)
* **Technology:** Zero-install single-page application built with HTML5, Tailwind CSS, and Leaflet.js.
* **Features:**
  * Host IP configuration field with instant connection health check.
  * Dropdown selector populated dynamically from Laptop 1.
  * Real-time metrics scorecard displaying observation count, latency, Top-1 accuracy comparison, and route continuity gains.
  * Interactive dual-layer Leaflet map rendering raw GPS fixes (blue markers) and the reconstructed continuous Viterbi route (bold emerald polyline).

### Option B: Streamlit Dashboard (`ui/app.py`)
* **Technology:** Python Streamlit app featuring Folium map integration.
* **Features:**
  * Run via `streamlit run ui/app.py`.
  * Visual KPI metric cards with delta indicators.
  * Full pan/zoom cartographic route rendering.

---

## 6.6 Evaluation Day Step-by-Step Run Instructions

### Step 1: Connect Both Laptops to the Same Local Network
* Connect Laptop 1 and Laptop 2 to the same Wi-Fi router or mobile hotspot.
* On Laptop 1, obtain its local IP address (e.g. `ipconfig` on Windows $\to$ `192.168.1.100`).

### Step 2: Start the Backend Server on Laptop 1
Open a terminal on Laptop 1:
```powershell
cd "C:\Users\joshwin\PESU\3RD YEAR\5th sem\ML\team_6"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Verify output: Server ready and listening on `http://0.0.0.0:8000`.*

### Step 3: Launch the Evaluator UI on Laptop 2
On Laptop 2:
* **Using Browser (Option A):** Simply double-click `ui/index.html` in any browser. Enter Laptop 1's IP address (e.g. `http://192.168.1.100:8000`), select any trip from the dropdown, and click **"Run Map-Matching & Score"**.
* **Using Streamlit (Option B):** Run:
```bash
streamlit run ui/app.py
```
* Set the Server URL to `http://192.168.1.100:8000` and evaluate any trip interactively.
