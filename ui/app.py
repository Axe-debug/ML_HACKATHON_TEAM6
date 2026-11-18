"""PathSnap Evaluator Dashboard (Streamlit Client for Laptop 2)."""
import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="PathSnap — Distributed Map-Matching Evaluator",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption("UE24CS352A Machine Learning — Team 6 (Distributed Two-Laptop Deployment)")

# Sidebar Controls
st.sidebar.header("Deployment Configuration")
server_url = st.sidebar.text_input("Laptop 1 Server URL", value="http://localhost:8000")

# Health Check
try:
    health_res = requests.get(f"{server_url}/api/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("✅ Laptop 1 Server Connected")
    else:
        st.sidebar.error("⚠️ Server returned non-200 status")
except Exception:
    st.sidebar.error("❌ Laptop 1 Server Unreachable")

# Trip Selection
try:
    trips_res = requests.get(f"{server_url}/api/trips", timeout=2)
    trip_list = trips_res.json().get("trips", []) if trips_res.status_code == 200 else []
except Exception:
    trip_list = ["00000015", "00000017", "00000023", "00000030", "00000049", "00000059", "00000067", "00000073", "00000087", "00000094"]

selected_trip = st.sidebar.selectbox("Select Evaluation Trip", options=trip_list)
run_btn = st.sidebar.button("🚀 Run Map-Matching & Scoring", type="primary")

# Main Execution
if run_btn:
    with st.spinner(f"Requesting inference & evaluation from Laptop 1 for Trip {selected_trip}..."):
        try:
            resp = requests.post(f"{server_url}/api/evaluate/{selected_trip}", timeout=60)
            if resp.status_code != 200:
                st.error(f"Error from server: {resp.text}")
            else:
                data = resp.json()
                
                # Metric Cards
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Observations", f"{data['observations']:,}")
                c2.metric(
                    "Top-1 Accuracy",
                    f"{data['hmm_top1_accuracy']:.2f}%",
                    f"+{data['accuracy_gain_pct']:.2f}% vs Classifier"
                )
                c3.metric(
                    "Route Continuity",
                    f"{data['hmm_continuity_pct']:.1f}%",
                    f"+{data['continuity_gain_pct']:.1f}% vs Classifier"
                )
                c4.metric("Inference Latency", f"{data['inference_time_ms']:.1f} ms")
                
                # Interactive Folium Map
                st.subheader(f"🗺️ Reconstructed Route & Trajectory — Trip {selected_trip}")
                
                route_pts = data["predicted_route_coords"]
                gps_pts = data["gps_coordinates"]
                
                if route_pts:
                    center_lat = sum(p[0] for p in route_pts) / len(route_pts)
                    center_lon = sum(p[1] for p in route_pts) / len(route_pts)
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")
                    
                    # Add Route Polyline
                    folium.PolyLine(
                        route_pts,
                        color="#10b981",
                        weight=5,
                        opacity=0.85,
                        tooltip="Reconstructed Viterbi Route"
                    ).addTo(m)
                    
                    # Add Raw GPS fixes
                    step = max(1, len(gps_pts) // 400)
                    for i in range(0, len(gps_pts), step):
                        folium.CircleMarker(
                            gps_pts[i],
                            radius=3,
                            color="#3b82f6",
                            fill=True,
                            fill_color="#60a5fa",
                            fill_opacity=0.6,
                            tooltip=f"GPS #{i}"
                        ).addTo(m)
                        
                    st_folium(m, width="100%", height=550)
                else:
                    st.warning("No route coordinates returned.")
                    
        except Exception as e:
            st.error(f"Failed to communicate with Laptop 1 server: {e}")
else:
    st.info("👈 Select an evaluation trip from the sidebar and click **Run Map-Matching & Scoring** to evaluate.")
