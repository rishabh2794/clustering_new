# Clustering + Batch Navigation with Browser Geolocation + IP fallback + Manual
# ------------------------------------------------------------------------------

import math
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from openpyxl import load_workbook

# Try browser geolocation
try:
    from streamlit_geolocation import geolocation
    HAVE_GEO = True
except ImportError:
    HAVE_GEO = False

# Optional: improve KML handling
try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False

# Constants
EARTH_RADIUS_M = 6_371_000.0
REQUIRED_COLS = {
    'ISSUE ID','CITY','ZONE','WARD','SUBCATEGORY','CREATED AT',
    'STATUS','LATITUDE','LONGITUDE','BEFORE PHOTO','AFTER PHOTO','ADDRESS'
}

st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Browser Geolocation + IP fallback)")

subcategory_options = [
    "Pothole",
    "Sand piled on roadsides + Mud/slit on roadside",
    "Garbage dumped on public land",
    "Unpaved Road",
    "Broken Footpath / Divider",
    "Malba, bricks, bori, etc dumped on public land",
    "Construction/ demolition activity without safeguards",
    "Encroachment-Building Materials Dumped on Road",
    "Burning of garbage, plastic, leaves, branches etc.",
    "Overflowing Dustbins",
    "Barren land to be greened",
    "Greening of Central Verges",
    "Unsurfaced Parking Lots"
]

# ---------------- Utility functions ----------------
def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    origin = f"&origin={origin_lat},{origin_lon}"
    dest = f"&destination={dest_lat},{dest_lon}"
    travel = f"&travelmode={mode}"
    wp = ""
    if waypoints:
        wp_str = "|".join([f"{lat},{lon}" for (lat, lon) in waypoints])
        wp = f"&waypoints={wp_str}"
    return f"{base}{origin}{dest}{travel}{wp}"

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row + 1):
            for col_idx in (11, 12):
                link = ws.cell(row, col_idx).value
                if isinstance(link, str) and link.startswith(("http://", "https://")):
                    ws.cell(row, col_idx).hyperlink = link
                    ws.cell(row, col_idx).style = "Hyperlink"
        wb.save(excel_path)
    except Exception as e:
        st.warning(f"Excel hyperlinking skipped: {e}")

def load_wards_uploaded(file):
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        if suffix in ("geojson","json"):
            wards = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if HAS_FIONA:
                wards = None
                for lyr in fiona.listlayers(tmp_path):
                    gdf_try = gpd.read_file(tmp_path, driver="KML", layer=lyr)
                    if not gdf_try.empty:
                        wards = gdf_try
                        break
            else:
                wards = gpd.read_file(tmp_path, driver="KML")
        else:
            st.error("Unsupported ward file type.")
            return None

        if wards is None or wards.empty:
            st.error("Ward file loaded but empty.")
            return None
        if wards.crs is None:
            wards.set_crs(epsg=4326, inplace=True)
        elif wards.crs.to_string() != "EPSG:4326":
            wards = wards.to_crs(epsg=4326)
        return wards[wards.geometry.notna() & ~wards.geometry.is_empty]
    except Exception as e:
        st.error(f"Error reading ward file: {e}")
        return None

def get_ip_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=5)
        if r.status_code == 200:
            d = r.json()
            return float(d['latitude']), float(d['longitude'])
    except:
        pass
    return None, None

# ---------------- Session state ----------------
for key in ["visited_ticket_ids","skipped_ticket_ids","batch_target_ids"]:
    if key not in st.session_state:
        st.session_state[key] = set()
if "current_target_id" not in st.session_state:
    st.session_state.current_target_id = None

# ---------------- UI: File uploads ----------------
csv_file = st.file_uploader("Upload CSV file with issues", type=["csv"])
ward_file = st.file_uploader("Upload WARD boundary file (optional)", type=["geojson","json","kml"])

subcategory_option = st.selectbox("Choose issue subcategory:", subcategory_options)
radius_m = st.number_input("Clustering Radius (meters)", 1, 1000, 15)
min_samples = st.number_input("Minimum Issues per Cluster", 1, 100, 2)

# ---------------- Main logic ----------------
if csv_file:
    df = pd.read_csv(csv_file)
    if REQUIRED_COLS - set(df.columns):
        st.error(f"Missing required columns: {REQUIRED_COLS - set(df.columns)}")
        st.stop()

    df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
    df = df[df['SUBCATEGORY_NORM'] == subcategory_option.lower()]
    df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
    df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
    df.dropna(subset=['LATITUDE','LONGITUDE'], inplace=True)

    coords_rad = np.radians(df[['LATITUDE','LONGITUDE']])
    eps_rad = radius_m / EARTH_RADIUS_M
    db = DBSCAN(eps=eps_rad, min_samples=int(min_samples),
                metric='haversine', algorithm='ball_tree')
    df['CLUSTER NUMBER'] = db.fit_predict(coords_rad)
    df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

    gdf_all = gpd.GeoDataFrame(df.copy(),
        geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']),
        crs="EPSG:4326"
    )

    wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
    if wards_gdf is not None:
        try:
            gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
        except:
            pass

    clustered = gdf_all[gdf_all['IS_CLUSTERED']].copy()
    excel_filename = "Clustering_Application_Summary.xlsx"
    if not clustered.empty:
        sizes = clustered.groupby('CLUSTER NUMBER')['ISSUE ID'].count().rename("NUMBER OF ISSUES")
        summary = clustered.merge(sizes, on='CLUSTER NUMBER', how='left')
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            summary.to_excel(writer, index=False, sheet_name="Clustering Application Summary")
        hyperlinkify_excel(excel_filename)

    # ---------------- Step 4A: Navigation with robust location ----------------
    st.subheader("Step 4A: Navigate to Nearest Tickets (Batch)")

    unique_statuses = sorted(gdf_all['STATUS'].dropna().astype(str).unique())
    default_statuses = [s for s in unique_statuses if s.lower() in ("open","pending","in progress")]
    include_statuses = st.multiselect("Eligible ticket statuses", unique_statuses,
                                      default=default_statuses or unique_statuses)
    wards_in_data = sorted(gdf_all['WARD'].dropna().astype(str).unique())
    ward_filter = st.multiselect("Limit to ward(s) (optional)", wards_in_data)

    travel_mode = st.selectbox("Travel mode", ["driving","walking","two_wheeler"])
    batch_size = st.slider("Batch size", 1, 10, 10)

    origin_lat = origin_lon = None
    st.markdown("### Your Location")
    colA, colB = st.columns(2)

    with colA:
        if 'origin_lat' in st.session_state and 'origin_lon' in st.session_state:
            origin_lat, origin_lon = st.session_state['origin_lat'], st.session_state['origin_lon']
            st.info(f"Using saved location: {origin_lat:.6f}, {origin_lon:.6f}")
        else:
            if HAVE_GEO:
                st.caption("Click the button below and allow location.")
                loc = geolocation()
                if loc and loc.get("latitude") and loc.get("longitude"):
                    origin_lat, origin_lon = float(loc["latitude"]), float(loc["longitude"])
                    st.session_state['origin_lat'] = origin_lat
                    st.session_state['origin_lon'] = origin_lon
                    st.success(f"Browser location: {origin_lat:.6f}, {origin_lon:.6f}")
            if origin_lat is None:
                if st.button("Use my approximate (IP-based) location"):
                    ip_lat, ip_lon = get_ip_location()
                    if ip_lat and ip_lon:
                        origin_lat, origin_lon = ip_lat, ip_lon
                        st.session_state['origin_lat'] = origin_lat
                        st.session_state['origin_lon'] = origin_lon
                        st.success(f"IP-based location: {origin_lat:.6f}, {origin_lon:.6f}")

    with colB:
        manual_lat = st.text_input("Manual latitude", "")
        manual_lon = st.text_input("Manual longitude", "")
        if manual_lat.strip() and manual_lon.strip():
            try:
                origin_lat, origin_lon = float(manual_lat), float(manual_lon)
                st.session_state['origin_lat'] = origin_lat
                st.session_state['origin_lon'] = origin_lon
                st.success(f"Manual location: {origin_lat:.6f}, {origin_lon:.6f}")
            except:
                st.error("Invalid manual coordinates.")

    # Filter pool
    pool = gdf_all.copy()
    if include_statuses:
        pool = pool[pool['STATUS'].isin(include_statuses)]
    if ward_filter:
        pool = pool[pool['WARD'].isin(ward_filter)]
    pool = pool[~pool['ISSUE ID'].isin(st.session_state.visited_ticket_ids)]
    pool = pool[~pool['ISSUE ID'].isin(st.session_state.skipped_ticket_ids)]

    st.write(f"Eligible tickets remaining: **{len(pool)}**")

    # Build nearest neighbour batch
    sequence_rows = []
    if origin_lat and origin_lon and not pool.empty:
        cur_lat, cur_lon = origin_lat, origin_lon
        pool2 = pool.copy()
        for _ in range(min(batch_size, len(pool2))):
            pool2['__dist_m'] = pool2.apply(lambda r:
                                            haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']),
                                            axis=1)
            nxt = pool2.sort_values('__dist_m').iloc[0]
            sequence_rows.append(nxt)
            cur_lat, cur_lon = nxt['LATITUDE'], nxt['LONGITUDE']
            pool2 = pool2[pool2['ISSUE ID'] != nxt['ISSUE ID']]

    if sequence_rows:
        waypoints = [(r['LATITUDE'], r['LONGITUDE']) for r in sequence_rows]
        nav_url = google_maps_url(origin_lat, origin_lon,
                                  waypoints[-1][0], waypoints[-1][1],
                                  mode=travel_mode, waypoints=waypoints[:-1])
        st.markdown(f"[🧭 Open continuous navigation in Google Maps]({nav_url})")

    # ---------------- Step 5: Map ----------------
    st.subheader("Step 5: Map Display Options")
    clusters_found = df[df['CLUSTER NUMBER'] != -1]['CLUSTER NUMBER'].nunique()
    map_type = st.radio("Select map type:",
                        ["Show all cluster markers (Type 1 Map)",
                         "Use Dynamic Clustering (Type 2 Map)"],
                        index=1 if clusters_found > 100 else 0)
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=13)
    if wards_gdf is not None:
        folium.GeoJson(wards_gdf, name="Wards").add_to(m)

    target_id = st.session_state.get('current_target_id')
    batch_ids = st.session_state.get('batch_target_ids', set())

    if map_type.startswith("Show all"):
        for _, row in gdf_all.iterrows():
            rid = str(row['ISSUE ID'])
            is_first = (rid == str(target_id))
            in_batch = rid in batch_ids
            color = 'green' if is_first else ('orange' if in_batch else 'red')
            folium.CircleMarker([row['LATITUDE'], row['LONGITUDE']],
                                radius=8, color=color, fill=True, fill_color=color).add_to(m)
    else:
        mc = MarkerCluster().add_to(m)
        for _, row in gdf_all.iterrows():
            rid = str(row['ISSUE ID'])
            is_first = (rid == str(target_id))
            in_batch = rid in batch_ids
            icon_color = 'green' if is_first else ('orange' if in_batch else 'red')
            folium.Marker([row['LATITUDE'], row['LONGITUDE']],
                          icon=folium.Icon(color=icon_color, icon='info-sign')).add_to(mc)

    folium.LayerControl().add_to(m)
    html_filename = "Clustering_Application_Map.html"
    m.save(html_filename)

    # ---------------- Step 6: Downloads ----------------
    st.subheader("Step 6: Download Outputs")
    if not clustered.empty:
        with open(excel_filename, "rb") as f:
            st.download_button("Download Clustering Application Summary (Excel)", f, file_name=excel_filename)
    with open(html_filename, "rb") as f:
        st.download_button("Download Clustering Application Map (HTML)", f, file_name=html_filename)
