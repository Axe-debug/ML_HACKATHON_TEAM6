# PathSnap application runtime and validation

This application was added locally in the hackathon workspace. The remote base inspected was main at `afd03dadb7637982d13d45c8167107712b4ea0d8`, with the Bayesian reference at `633cf9a` and the published HMM at `dbd6adf` on its review branch. Git transport could not resolve GitHub and the workspace did not have a usable Git worktree. The application is a new-file review package. It has not been committed or pushed. Review `backend/`, `frontend/`, and these two deployment documents. On a network-enabled, clean checkout, create `feature/extra-trees-fastapi-frontend` from the intended HMM base and copy the reviewed source files into it. Check `git status --short` and `git diff --stat` after staging. Exclude built output, local logs, model binaries, GPS, predictions and reference archives from any public Git commit.

## Fastest demo

1. Connect both laptops to the same mobile hotspot.
2. Laptop 1 installs the Python requirements and starts Uvicorn as shown in [the deployment guide](PATHSNAP_DEPLOYMENT.md). Leave the server terminal open.
3. Laptop 2 can use the prebuilt `frontend/dist` from the handoff ZIP. Run `python -m http.server 5173 --bind 127.0.0.1 --directory frontend/dist` from the project root. Open `http://127.0.0.1:5173`, enter Laptop 1's hotspot IP and port, then connect. A clean source build is also documented.
4. Select a trip and run classifier-only first. Match the results card's request ID to the FastAPI console. Enable HMM on Laptop 1 only for the explicit comparison.
5. Switch the backend to the organiser's evaluator partition when the ten reserved trips and private references arrive.

## Verified evidence

- The API loaded the verified tuned Extra Trees artifact with ten features and positive-class column 1. One FastAPI lifespan reused the loaded model across all requests.
- The real ASGI-backed student-test sweep returned **10 successful predictions** for 19,160 GPS points. Every chosen arc and observation row matched the stored evaluation outputs. Top-1 was **81.1795%** and the candidate miss rate was **0.3497%**.
- On this verification machine, median classifier inference was **226.1 ms per trip** and maximum was **570.1 ms**. The peak Python verification process RSS was **653,024 KiB** across all test cases, including repeated model startup. It is not per-request memory.
- The raw-only API path reproduced prepared features and candidate IDs on a real trip. Its optional HMM response matched the saved Viterbi route on that trip. Logs measured **843.7 ms** decoder time and eight sequence breaks on that request.
- Seven existing preprocessing/scoring tests, three backend test cases with multiple HTTP assertions, and four frontend tests passed. The Vite build passed. The frontend contract test consumed actual responses captured from FastAPI via a replay transport.
- Uvicorn completed application startup but could not bind a listener in the sandbox. Chromium also could not start under this sandbox. No live socket, screenshot, second-device or reserved evaluator-trip test was completed here.

The expected classifier accuracy is a student-test result against derived proxy labels. HMM improved motion-feasible continuity and Macro F1 in its separate all-trip experiment, while reducing Top-1. The app defaults to classifier-only.

The actual log for one of the verified requests included the same request ID on all stages. The closing event showed status 200, 2,503 observations, **313.727 ms inference**, and **410.540 ms total**. See `backend/logs/server.log` and `backend/logs/evidence/request_events.json` for the full local event chain. `backend/logs/hmm.log` contains HMM events. `backend/logs/evidence/api_summary.json` records the sweep. The server rotates logs and emits the same INFO events on the terminal. The local map and response fixture remain under the ignored evidence folder.

## Viva

**60 seconds.** The UI on Laptop 2 discovers trips through FastAPI on Laptop 1. FastAPI loads the tuned Extra Trees model at startup. It reads eight candidates with ten features per GPS point, predicts candidate scores and maps the chosen candidate back to its trip-local arc. The server optionally decodes a connected sequence through Viterbi, then computes metrics using private reference data. The frontend displays the prediction and summary. A request ID connects each visible result to live logs. All ten student trips reproduced the stored model results at 81.18% Top-1.

**Three minutes.** Describe the two laptops and the hotspot. Trace a request from the trip dropdown to the model, road geometry, optional decoder, separate scoring and API output. Explain class column 1 and trip-local arc IDs. Describe the fastest classifier-only route and why Viterbi can lower Top-1 while producing more feasible paths. Show startup and request logs using the request ID. Distinguish the verified student-test sweep from the reserved evaluator demonstration.

| Question | Answer |
|---|---|
| Why FastAPI? | It defines typed request and response contracts and hosts the trained Python model. |
| When does the model load? | Once during server lifespan startup. |
| Why Extra Trees? | It led the team's comparable baseline CV and the chosen tuned artifact improved held-out matching slightly. |
| What is the model input? | Eight candidates by ten ordered features per observation. |
| How is one arc selected? | Use the score for class 1, then pick the highest-scoring valid candidate and its original arc ID. |
| Why a request ID? | It ties the frontend result to loading, inference, decoder, scoring and completion logs. |
| How do the laptops connect? | Both join the hotspot. The UI uses Laptop 1's IP and port. The server allows the specific UI origin. |
| Where are evaluator labels? | They remain in Laptop 1's scoring service and are read only after prediction. |
| What changes with HMM? | Viterbi combines candidate evidence and directed road transitions across observations. |
| What does the demo still need? | Run the two real laptops on the hotspot and check the reserved trip list once the organiser supplies it. |
