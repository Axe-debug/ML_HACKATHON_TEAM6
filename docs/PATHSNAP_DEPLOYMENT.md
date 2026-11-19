# PathSnap two-laptop deployment

Run the **Python API on Laptop 1** with the trusted Extra Trees artifact, processed inputs, raw road network and private reference data. Run the **React UI on Laptop 2**. The browser sends requests to Laptop 1. Its dropdown comes from the active partition on the server. The browser receives GPS and predicted road geometry plus aggregate scores. The decoder and model never consume reference labels. Server-side scoring is a separate step after prediction.

## Install and stage the files

Use Python 3.12 and Node 20 or newer. From the project root on Laptop 1:

```sh
python -m venv .venv-api
# Ubuntu / Linux
source .venv-api/bin/activate
# Windows PowerShell: .\.venv-api\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

Keep the following trusted files on the API laptop:

| Item | Configuration |
|---|---|
| Tuned Extra Trees Joblib file | `PATHSNAP_MODEL_PATH`; verified SHA-256 `e5a913e5db2ad5289a08b2c36498f7448e8e33b24158a29d1be81b39ae7a7c92` |
| Optional checksum enforcement | `PATHSNAP_MODEL_SHA256` |
| Training scikit-learn version | `PATHSNAP_MODEL_SKLEARN`, defaults to the verified 1.9.1 |
| Ordered feature schema | `PATHSNAP_SCHEMA_PATH`, default `processed/schema.json` |
| Prepared inference inputs | `PATHSNAP_PREPARED_DIR`, default `processed/<partition>` |
| Raw trip folders | `PATHSNAP_RAW_DIR`; each valid trip directory contains `.track`, `.nodes` and `.arcs` with the trip ID basename |
| Private scoring inputs | `PATHSNAP_REFERENCE_DIR`; prealigned `<trip>.npz` or a `<trip>.route` at the root or in its trip subfolder |
| Decoder definitions when enabled | `PATHSNAP_HMM_NOTEBOOK`, a trusted copy of `notebooks/06_hmm_viterbi.ipynb` |
| Global model benchmark, optional | `PATHSNAP_COMPARISON_PATH`, points to the Bayesian `comparison.json` |

If the prepared trip NPZ is absent, the server regenerates the same features from the raw GPS and directed network. It reads only GPS, candidates and features during inference. Raw reference-route alignment happens afterward in the scoring service. Very large raw alignment jobs require a prealigned reference NPZ to limit memory use. When labels are absent, metrics are blank and the predicted geometry still runs. When network files are absent but valid prepared features exist, classifier inference remains available and map/HMM capability is reduced. With the explicit baseline variant, supply the matching `untuned_model.joblib`; the app rejects a variant mismatch.

## Laptop 1 — API and logs

Find Laptop 1’s Wi-Fi LAN IPv4 address with `ip -4 addr` on Ubuntu or `ipconfig` in PowerShell. Use a trusted hotspot or a private LAN. The examples below bind the API to the LAN. Restrict inbound TCP 8000 to Laptop 2 where possible and avoid exposing the service to the internet. CORS controls browser access and is not authentication.

Ubuntu / Linux:

```sh
export PATHSNAP_HOST=0.0.0.0
export PATHSNAP_PORT=8000
export PATHSNAP_MODEL_PATH="/absolute/path/to/tuned_model.joblib"
export PATHSNAP_MODEL_SHA256=e5a913e5db2ad5289a08b2c36498f7448e8e33b24158a29d1be81b39ae7a7c92
export PATHSNAP_PARTITION=student_test
export PATHSNAP_CORS_ORIGINS=http://127.0.0.1:5173
python -m uvicorn backend.main:app --host "$PATHSNAP_HOST" --port "$PATHSNAP_PORT" --no-access-log
```

Windows PowerShell:

```powershell
$env:PATHSNAP_HOST = "0.0.0.0"
$env:PATHSNAP_PORT = "8000"
$env:PATHSNAP_MODEL_PATH = "C:\PathSnap\artifacts\tuned_model.joblib"
$env:PATHSNAP_MODEL_SHA256 = "e5a913e5db2ad5289a08b2c36498f7448e8e33b24158a29d1be81b39ae7a7c92"
$env:PATHSNAP_PARTITION = "student_test"
$env:PATHSNAP_CORS_ORIGINS = "http://127.0.0.1:5173"
python -m uvicorn backend.main:app --host $env:PATHSNAP_HOST --port $env:PATHSNAP_PORT --no-access-log
```

Classifier-only is the default. To also enable the real HMM before starting Uvicorn, set `PATHSNAP_ENABLE_HMM=1` and optionally `PATHSNAP_HMM_NOTEBOOK` on Laptop 1. Classifier-only remains the UI default. The HMM adapter runs only the trusted decoder-definition cell at startup. It rejects missing GPS or road topology and does not read evaluator labels. For the currently measured tradeoff, use the [HMM results](https://github.com/kalluri088/ML_hackathon_212_210_209_208/tree/feature/hmm-viterbi/reports/hmm).

For the final evaluator partition, set `PATHSNAP_PARTITION=evaluator_test`; explicitly set `PATHSNAP_RAW_DIR` to the supplied evaluator trips and `PATHSNAP_REFERENCE_DIR` to the private scoring location on Laptop 1. Optionally set an evaluator-only prepared directory. Confirm that `GET /api/trips` lists all ten assigned IDs before the demo. The evaluator data was not available for this implementation test.

Leave the API terminal open. For a second terminal view use `tail -f backend/logs/server.log` on Linux or `Get-Content backend/logs/server.log -Wait` on Windows. Rotation keeps up to three backups at about 2 MB each. Copy the request ID from the frontend results card and find it in the log. Use `grep -F <request_id> backend/logs/server.log` or PowerShell `Select-String -Path backend/logs/server.log -SimpleMatch '<request_id>'`. Console events show the model variant, data loading, inference, optional decoder, metrics, status and timings. Debug stack traces are kept in the server file. The application exposes no log-download endpoint.

## Laptop 2 — UI

Replace `192.168.1.20` with Laptop 1’s IP and confirm reachability.

```sh
curl http://192.168.1.20:8000/health
cd frontend
npm ci
# Linux
VITE_API_BASE_URL=http://192.168.1.20:8000 npm run build
# Windows PowerShell instead: $env:VITE_API_BASE_URL = "http://192.168.1.20:8000"; npm run build
python -m http.server 5173 --bind 127.0.0.1 --directory dist
```

On Windows use `curl.exe` or `Invoke-RestMethod http://192.168.1.20:8000/health`. Open `http://127.0.0.1:5173` on Laptop 2. You can also enter the API origin at the top of the UI. The backend allowlist contains the frontend page origin, which is the local 127.0.0.1 origin in this example, not Laptop 1’s IP. If using a different frontend address or port, set the exact new origin and restart the API.

Select a trip and click **Run prediction**. Confirm the model label, mode, selected-trip metrics and visible map. Use the optional Folium link for an interactive HTML view. Its Leaflet assets need network access, while the built-in SVG map uses the API geometry directly. For the report, capture the selected-trip screen and the server terminal showing the same request ID. Capture the development reference-route overlay from the separate HMM notebook map, which stays local.

If health does not load from Laptop 2, check the IP and port, firewall, Wi-Fi client isolation, and whether Uvicorn bound to 0.0.0.0. If health works from a terminal but the browser shows a CORS error, compare the page origin to `PATHSNAP_CORS_ORIGINS`. A 503 response signals a missing model or dataset. Check startup logs, model hash/version, and partition paths.

## API contract

| Endpoint | Result |
|---|---|
| `GET /health` | Readiness, active model variant, active partition and HMM capability |
| `GET /api/model` | Extra Trees metadata, feature order, artifact hash, class index and optional aggregate benchmark |
| `GET /api/trips` | IDs and geometry availability for the active data partition |
| `POST /api/predict` | Given trip ID and mode, return ordered predicted arc IDs, original row IDs, permitted scores, timing, map URL and request ID |
| `GET /api/trips/{id}/map?mode=classifier_only` | GeoJSON layers. Add `&format=html` for Folium |
| `GET /api/trips/{id}/results?mode=classifier_only` | In-memory summary after a successful prediction for that trip and mode |

A prediction request body is `{"trip_id":"00000000","mode":"classifier_only"}`, using an actual student-test example. HMM requests use `"mode":"hmm"`. Typical responses include 404 for unknown trips, 409 when HMM is disabled, 422 for invalid input, 429 when the two prediction slots are busy, and 503 when the model cannot load. Route continuity in classifier-only mode is direct endpoint adjacency, not directed-route feasibility. The server returns HMM feasibility and sequence breaks only if measured. Where available, Top-1 and Macro F1 use the shared aligned-reference convention. Student model benchmark scores are identified separately from the selected trip's results.

## Tests

```sh
python -m unittest discover -s tests -v
# Linux
PATHSNAP_REAL_TESTS=1 python -m unittest discover -s backend/tests -v
# Windows PowerShell: $env:PATHSNAP_REAL_TESTS = "1"; python -m unittest discover -s backend/tests -v
npm --prefix frontend test
npm --prefix frontend run build
```

The full student test expects the trusted tuned Joblib file and its saved predictions to share a directory, and a local copy of the HMM notebook. Logs and test-generated map/fixture files under `backend/logs/` are ignored. Do not publish trip payloads or reference data.
