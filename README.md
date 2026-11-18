PathSnap: Distributed Map-Matching & Route Reconstruction

Course: UE24CS352A — Machine Learning
Institution: PES University
Team 6: Yaswanth, Madhu, Hrisheek, Joshwin

 Project Overview

PathSnap matches noisy GPS sensor fixes to a directed road network graph using machine learning and graph algorithms:

Candidate Selection & Feature Extraction: For every GPS point, 
𝐾
=
8
K=8 candidate road segments are found using a spatial KDTree. A 10-dimensional spatial-kinematic feature vector is extracted.
Classifier: A tuned Histogram Gradient Boosting model predicts the probability of each candidate segment.
Route Reconstruction (HMM + Viterbi): A Hidden Markov Model with the Viterbi algorithm connects the predictions into a physically continuous vehicle route using road network shortest paths.
 Key Results
Student-Test Partition (10 Trips):
Top-1 Accuracy: 81.18% (Classifier) 
→
→ 84.70% (HMM + Viterbi)
Route Continuity: 89.23% (Classifier) 
→
→ 96.98% (HMM + Viterbi)
Evaluator-Test Partition (10 Trips):
Top-1 Accuracy: 74.13% (Classifier) 
→
→ 76.71% (HMM + Viterbi)
Route Continuity: 89.04% (Classifier) 
→
→ 95.62% (HMM + Viterbi)
🚀 How to Run (Two-Laptop Evaluation Setup)

Connect both laptops to the same Wi-Fi network.

Step 1: Start Server on Laptop 1
Open PowerShell / Terminal in the project folder:
powershell
cd "team_6"
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
Find Laptop 1's Wi-Fi IP address by running ipconfig (e.g. 10.0.15.194).
Step 2: Open Client UI on Laptop 2
Open ui/index.html in Google Chrome or Microsoft Edge.
In the Server Host URL / IP box, enter Laptop 1's URL:
text
http://10.0.15.194:8000
Click Test Server Connection (turns green).
Select any Evaluator Trip from the dropdown (e.g. Trip 00000049 or Trip 00000015).
Click Run Map-Matching & Score.
The scorecard displays accuracy, route continuity, and latency.
The map shows the raw GPS points (blue) and the reconstructed route (green).
Folder Structure
api/ — FastAPI backend server (api/main.py)
ui/ — Interactive Leaflet map dashboard (ui/index.html) & Streamlit app (ui/app.py)
evaluator_test/ — 10 evaluator test trips
results/ — Trained model weights (model.joblib), evaluation summaries, and predictions
reports/ — Technical Report Word document (.docx), markdown files, and figures
notebooks/ — 6 Jupyter notebooks covering EDA and each individual model
scripts/ — Preprocessing, training, HMM decoding, and map generation scripts
tests/ — Unit test suite (tests/test_preprocessing.py)
🧪 Running Unit Tests

To verify all 7 unit tests (projection, distortion, candidate selection, alignment):

powershell
python -m unittest discover -s tests -v
📄 Technical Report

The complete technical report answering all rubric questions (Sections 2 to 7) is available in:

Word Document: reports/PathSnap_Technical_Report_Team_6.docx
Markdown: reports/FINAL_TECHNICAL_REPORT.md
Interactive Map: reports/folium_trip_00000049_four_layers.html
