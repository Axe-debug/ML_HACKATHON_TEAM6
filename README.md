# PathSnap – Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A – Machine Learning  

**Team:** 6 – Yaswanth, Madhu, Hrisheek, Joshwin

## About the Project

PathSnap is a machine learning project that reconstructs a vehicle's route using GPS data and a road network.

GPS signals are not always accurate. They can contain errors or show a vehicle on the wrong road, especially when multiple roads are close to each other.

Our system uses machine learning and graph algorithms to identify the most likely road segments and reconstruct a continuous route.

## How It Works

The project follows three main steps:

1. **Candidate Selection:** For every GPS point, the system finds 8 nearby road segments using a KDTree and extracts 10 useful features.
2. **Machine Learning:** A tuned Histogram Gradient Boosting model predicts the probability of each candidate road segment.
3. **Route Reconstruction:** An HMM combined with the Viterbi algorithm uses these predictions and road network connections to find a continuous route.

## Results

We evaluated our system on two separate test datasets.

| Test Dataset | Classifier Accuracy | HMM Accuracy | Classifier Continuity | HMM Continuity |
|---|---:|---:|---:|---:|
| Student Test | 81.18% | **84.70%** | 89.23% | **96.98%** |
| Evaluator Test | 74.13% | **76.71%** | 89.04% | **95.62%** |

The results show that combining the classifier with HMM and Viterbi improves both prediction accuracy and route continuity.

## Project Structure

| Folder | Description |
|---|---|
| `api/` | FastAPI backend server |
| `ui/` | Interactive map dashboard and Streamlit application |
| `evaluator_test/` | Evaluator testing data |
| `results/` | Trained models, predictions and evaluation results |
| `reports/` | Technical report, documentation and figures |
| `notebooks/` | Jupyter notebooks for EDA and model experiments |
| `scripts/` | Preprocessing, training and route reconstruction scripts |
| `tests/` | Unit tests for the project |

## Running the Project

The project uses two laptops connected to the same Wi-Fi network.

### 1. Start the Backend (Laptop 1)

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Find the laptop's IP address using:

```bash
ipconfig
```

Note down the Wi-Fi IPv4 address.

### 2. Open the User Interface (Laptop 2)

1. Open `ui/index.html` in Chrome or Edge.
2. Enter the backend URL using Laptop 1's IP address.

   Example: `http://10.0.15.194:8000`

3. Click **Test Server Connection**.
4. Select an evaluator trip from the dropdown.
5. Click **Run Map-Matching & Score**.

The dashboard displays the evaluation metrics and reconstructed route on an interactive map.

## Running Unit Tests

To run the project's unit tests, execute:

```bash
python -m unittest discover -s tests -v
```

The tests cover coordinate projection, distortion, candidate selection and route alignment.

## Documentation

The following files contain additional project details:

- **Technical Report:** `reports/PathSnap_Technical_Report_Team_6.docx`
- **Final Technical Report:** `reports/FINAL_TECHNICAL_REPORT.md`
- **Interactive Map:** `reports/folium_trip_00000049_four_layers.html`

## Technologies Used

- **Python** – Main programming language
- **Scikit-learn** – Machine learning models
- **Optuna** – Hyperparameter optimisation
- **FastAPI** – Backend API
- **Folium & Leaflet.js** – Map visualisation
- **Tailwind CSS** – User interface styling
- **Jupyter Notebook** – Data analysis and experiments

---

**Team 6 | PES University**
