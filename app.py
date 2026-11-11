# app.py — Clustering + Batch Navigation + Skip/Mark + Clickable Image Popups
# with Map-Click Start & Address Search (no GPS/IP needed)
# ----------------------------------------------------------------------------

import math
import tempfile
from html import escape
from urllib.parse import quote

import folium
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import requests
import streamlit as st
from streamlit_folium import st_folium
from openpyxl import load_workbook

# Optional heavy deps — guarded (GeoPandas/Fiona may be unavailable on Streamlit Cloud)
try:
    import geopandas as gpd
    HAVE_GPD = True
except Exception:
    HAVE_GPD = False

try:
    import fiona
    HAS_FIONA = True
except Exception:
    HAS_FIONA = False

# ----------------- Constants -----------------
EARTH_RADIUS_M = 6_371_000.0

# ----------------- Helpers -----------------
def normalize_subcategory(series: pd.Series):
    return series.astype(str).str.strip().str.lower()

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

def google_maps_url(origin_lat, origin_lon, dest_lat, dest_lon, mode="driving", waypoints=None):
    base = "https://www.google.com/maps/dir/?api=1"
    parts = []
    if origin_lat is not None and origin_lon is not None:
        parts.append(f"origin={origin_lat},{origin_lon}")
    parts.append(f"destination={dest_lat},{dest_lon}")
    mode = mode if mode in {"driving", "walking", "bicycling", "transit"} else "driving"
    parts.append(f"travelmode={mode}")
    if waypoints:
        wp = "|".join([f"{lat},{lon}" for (lat, lon) in waypoints])
        parts.append(f"waypoints={quote(wp, safe='|,')}")
    return base + "&" + "&".join(parts)

def hyperlinkify_excel(excel_path):
    try:
        wb = load_workbook(excel_path)
        ws = wb["Clustering Application Summary"]
        for row in range(2, ws.max_row + 1):
            for col in range(1, ws.max_column + 1):
                cell_val = str(ws.cell(row, col).value or "")
                if cell_val.startswith(("http://", "https://")):
                    ws.cell(row, col).hyperlink = cell_val
                    ws.cell(row, col).style = "Hyperlink"
        wb.save(excel_path)
    except Exception:
        pass

def load_wards_uploaded(file):
    """Ward file reader with KML guard. Returns GeoDataFrame or None."""
    if not HAVE_GPD:
        st.info("GeoPandas not available; ward overlay/join skipped.")
        return None
    try:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix in ("geojson", "json"):
            gdf = gpd.read_file(tmp_path)
        elif suffix == "kml":
            if not HAS_FIONA:
                st.warning("KML requires Fiona/GDAL. Upload GeoJSON/JSON instead.")
                return None
            layers = fiona.listlayers(tmp_path)
            gdf = None
            for layer in layers:
                gdf_try = gpd.read_file(tmp_path, driver="KML", layer=layer)
                if not gdf_try.empty:
                    gdf = gdf_try
                    break
            if gdf is None:
                st.warning("No non-empty KML layers found.")
                return None
        else:
            st.warning("Unsupported ward file type. Use GeoJSON/JSON/KML.")
            return None

        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    except Exception:
        return None

def geocode_query(q: str, timeout=6):
    """Address/place geocoder (OSM Nominatim). Returns (lat, lon) or (None, None)."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": 1}
        headers = {"User-Agent": "cluster-app/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

def build_nearest_neighbor_sequence(df_like: pd.DataFrame, start_lat: float, start_lon: float, limit: int):
    """Greedy NN sequence from a start; returns list of rows (dict-like)."""
    seq = []
    pool = df_like.copy()
    if pool.empty:
        return seq
    cur_lat, cur_lon = float(start_lat), float(start_lon)
    for _ in range(min(limit, len(pool))):
        pool['__dist'] = pool.apply(lambda r: haversine_m(cur_lat, cur_lon, r['LATITUDE'], r['LONGITUDE']), axis=1)
        nxt = pool.sort_values('__dist').iloc[0]
        seq.append(nxt)
        cur_lat, cur_lon = float(nxt['LATITUDE']), float(nxt['LONGITUDE'])
        pool = pool[pool['ISSUE ID'] != nxt['ISSUE ID']]
    return seq

def is_url(u):
    return isinstance(u, str) and u.startswith(("http://", "https://"))

# ----------------- App Setup -----------------
st.set_page_config(layout="wide", page_title="Clustering + Batch Navigation")
st.title("🗺️ Clustering + Batch Navigation (Map-Click Start • Skip/Mark • Downloads)")

# Session state initialization
for k in ["visited_ticket_ids", "skipped_ticket_ids"]:
    if k not in st.session_state:
        st.session_state[k] = set()
if "batch_cursor" not in st.session_state:
    st.session_state.batch_cursor = 0
if "origin_lat" not in st.session_state:
    st.session_state.origin_lat = None
if "origin_lon" not in st.session_state:
    st.session_state.origin_lon = None

# ----------------- Inputs -----------------
csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader_main")
ward_file = st.file_uploader("Upload Wards file (optional)", type=["geojson", "json", "kml"], key="ward_uploader_main")

radius_m = st.number_input("Clustering radius (m)", 1, 1000, 15, key="num_radius")
min_samples = st.number_input("Minimum per cluster", 1, 100, 2, key="num_min_samples")

# ----------------- Main Processing -----------------
if not csv_file:
    st.info("Upload the required CSV to proceed.")
    st.stop()

df = pd.read_csv(csv_file)

# --- Dynamic Column Mapping for GVP Gaya.csv & Standard Formats ---
col_mapping = {}
for col in df.columns:
    c_lower = col.strip().lower()
    if c_lower in ['issue id', 'project id', 'ticket id', 'id']:
        col_mapping[col] = 'ISSUE ID'
    elif c_lower in ['subcategory', 'action item', 'category', 'issue']:
        col_mapping[col] = 'SUBCATEGORY'
    elif c_lower in ['zone']:
        col_mapping[col] = 'ZONE'
    elif c_lower in ['city']:
        col_mapping[col] = 'CITY'
    elif c_lower in ['ward']:
        col_mapping[col] = 'WARD'
    elif c_lower in ['status']:
        col_mapping[col] = 'STATUS'
    elif c_lower in ['latitude', 'lat']:
        col_mapping[col] = 'LATITUDE'
    elif c_lower in ['longitude', 'long', 'lng']:
        col_mapping[col] = 'LONGITUDE'
    elif c_lower in ['before photo', 'raised evidence', 'photo', 'image', 'evidence']:
        col_mapping[col] = 'BEFORE PHOTO'
    elif c_lower in ['after photo', 'closed evidence']:
        col_mapping[col] = 'AFTER PHOTO'
    elif c_lower in ['address', 'landmark*', 'landmark', 'location']:
        col_mapping[col] = 'ADDRESS'
    elif c_lower in ['created at', 'raised on', 'date']:
        col_mapping[col] = 'CREATED AT'

df = df.rename(columns=col_mapping)

for req in ['ISSUE ID', 'CITY', 'ZONE', 'WARD', 'SUBCATEGORY', 'CREATED AT', 'STATUS', 'LATITUDE', 'LONGITUDE', 'BEFORE PHOTO', 'AFTER PHOTO', 'ADDRESS']:
    if req not in df.columns:
        df[req] = 'N/A'

available_subcats = df['SUBCATEGORY'].dropna().unique().tolist()
if not available_subcats:
    available_subcats = ["Pothole", "Garbage dumped on public land"]

subcategory_option = st.selectbox("Issue Subcategory", available_subcats, key="sel_subcategory")

df['SUBCATEGORY_NORM'] = normalize_subcategory(df['SUBCATEGORY'])
df = df[df['SUBCATEGORY_NORM'] == str(subcategory_option).strip().lower()].copy()

df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df = df[df['LATITUDE'].between(-90, 90, inclusive='both')]
df = df[df['LONGITUDE'].between(-180, 180, inclusive='both')]
df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

if df.empty:
    st.warning("No valid rows for the selected subcategory after cleaning LAT/LON.")
    st.stop()

# DBSCAN Clustering
coords_deg = df[['LATITUDE', 'LONGITUDE']].to_numpy(dtype=float)
coords_rad = np.radians(coords_deg)
eps_rad = float(radius_m) / EARTH_RADIUS_M
db = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric='haversine', algorithm='ball_tree')
try:
    labels = db.fit_predict(coords_rad)
except ValueError as e:
    st.error(f"Clustering failed due to invalid input: {e}")
    st.stop()

df['CLUSTER NUMBER'] = labels
df['IS_CLUSTERED'] = df['CLUSTER NUMBER'] != -1

# Optional GeoPandas overlay
if HAVE_GPD:
    gdf_all = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
    wards_gdf = load_wards_uploaded(ward_file) if ward_file else None
    if wards_gdf is not None and not wards_gdf.empty:
        try:
            gdf_all = gpd.sjoin(gdf_all, wards_gdf, how="left", predicate="within")
        except Exception:
            pass
else:
    gdf_all = df.copy()

# Excel summary generation
clustered = gdf_all[gdf_all['IS_CLUSTERED']]
excel_filename = "Clustering_Application_Summary.xlsx"
if not clustered.empty:
    (clustered.drop(columns=["geometry"]) if "geometry" in clustered.columns else clustered).to_excel(
        excel_filename, sheet_name='Clustering Application Summary', index=False
    )
    hyperlinkify_excel(excel_filename)

# ---------------- Step 4: Start Point ----------------
st.subheader("Step 4: Set Start Point (No GPS Needed)")

colA, colB = st.columns(2)
with colA:
    q = st.text_input("🔎 Search address / landmark (optional)", key="txt_addr_query")
    if st.button("Search & set start", key="btn_addr_geocode"):
        if q.strip():
            g_lat, g_lon = geocode_query(q.strip())
            if g_lat is not None:
                st.session_state.origin_lat = g_lat
                st.session_state.origin_lon = g_lon
                st.success(f"Start set from search: {g_lat:.6f}, {g_lon:.6f}")
                st.rerun()
            else:
                st.error("Could not geocode that query. Try a more specific address.")

with colB:
    man_lat = st.text_input("Manual latitude (optional)", key="txt_lat")
    man_lon = st.text_input("Manual longitude (optional)", key="txt_lon")
    if st.button("Set manual start", key="btn_set_manual"):
        try:
            st.session_state.origin_lat = float(man_lat)
            st.session_state.origin_lon = float(man_lon)
            st.success(f"Manual start set: {st.session_state.origin_lat:.6f}, {st.session_state.origin_lon:.6f}")
            st.rerun()
        except Exception:
            st.error("Invalid coordinates. Example: 24.7938, 85.0243")

st.caption("🗺️ Or click anywhere on the mini map to set your starting point:")
center_lat = float(df['LATITUDE'].mean())
center_lon = float(df['LONGITUDE'].mean())

# Using CartoDB positron tiles to avoid 403 blocks
mini = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
if st.session_state.origin_lat is not None and st.session_state.origin_lon is not None:
    folium.Marker(
        [st.session_state.origin_lat, st.session_state.origin_lon],
        popup="Start here",
        icon=folium.Icon(color="green", icon="flag")
    ).add_to(mini)

click_state = st_folium(mini, height=320, width=None, key="mini_click", returned_objects=["last_clicked"])
if isinstance(click_state, dict):
    clicked = click_state.get("last_clicked")
    if clicked and "lat" in clicked and "lng" in clicked:
        st.session_state.origin_lat = float(clicked["lat"])
        st.session_state.origin_lon = float(clicked["lng"])
        st.success(f"Start set from map click: {st.session_state.origin_lat:.6f}, {st.session_state.origin_lon:.6f}")
        st.rerun()

origin_lat = st.session_state.origin_lat
origin_lon = st.session_state.origin_lon
if origin_lat is not None and origin_lon is not None:
    st.info(f"Using start: {origin_lat:.6f}, {origin_lon:.6f}")
else:
    st.warning("No start set. Google Maps will launch using your device's live location.")

# ---------------- Step 5: Batch Navigation ----------------
st.subheader("Step 5: Batches of 10 — View • Skip • Mark")

pool = (gdf_all.drop(columns=["geometry"]) if "geometry" in getattr(gdf_all, "columns", []) else gdf_all).copy()
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.visited_ticket_ids)]
pool = pool[~pool['ISSUE ID'].astype(str).isin(st.session_state.skipped_ticket_ids)]

st.caption(f"Remaining eligible tickets: {len(pool)} | Visited: {len(st.session_state.visited_ticket_ids)} | Skipped: {len(st.session_state.skipped_ticket_ids)}")

if pool.empty:
    st.info("No tickets remaining after filters/visited/skipped.")
    batch_df = pd.DataFrame(columns=['ISSUE ID', 'WARD', 'STATUS', 'LATITUDE', 'LONGITUDE'])
else:
    if origin_lat is None or origin_lon is None:
        seed_lat, seed_lon = float(pool['LATITUDE'].mean()), float(pool['LONGITUDE'].mean())
    else:
        seed_lat, seed_lon = origin_lat, origin_lon

    long_seq_rows = build_nearest_neighbor_sequence(pool, seed_lat, seed_lon, limit=min(200, len(pool)))
    seq_df = pd.DataFrame(long_seq_rows)

    if st.session_state.batch_cursor >= len(seq_df):
        st.session_state.batch_cursor = 0

    start = st.session_state.batch_cursor
    end = min(start + 10, len(seq_df))
    batch_df = seq_df.iloc[start:end].copy()

    if batch_df.empty and not seq_df.empty:
        st.session_state.batch_cursor = 0
        start, end = 0, min(10, len(seq_df))
        batch_df = seq_df.iloc[start:end].copy()

    batch_df_display = batch_df[['ISSUE ID', 'WARD', 'STATUS', 'LATITUDE', 'LONGITUDE']].reset_index(drop=True)
    batch_df_display.index = batch_df_display.index + 1
    st.dataframe(batch_df_display, use_container_width=True)

    if not batch_df.empty:
        waypoints = [(float(r['LATITUDE']), float(r['LONGITUDE'])) for _, r in batch_df.iterrows()]
        last = waypoints[-1]
        mids = waypoints[:-1] if len(waypoints) > 1 else []
        nav_url = google_maps_url(origin_lat, origin_lon, last[0], last[1], waypoints=mids)
        st.markdown(f"[🧭 Open route in Google Maps for this batch]({nav_url})")

    c1, c2, c3, c4 = st.columns(4)
    first_id = str(batch_df.iloc[0]['ISSUE ID']) if not batch_df.empty else None

    with c1:
        if st.button("✅ Mark first as Visited", key="btn_mark_first"):
            if first_id:
                st.session_state.visited_ticket_ids.add(first_id)
                st.rerun()

    with c2:
        if st.button("⏭️ Skip first", key="btn_skip_first"):
            if first_id:
                st.session_state.skipped_ticket_ids.add(first_id)
                st.rerun()

    with c3:
        if st.button("✅ Mark entire batch as Visited", key="btn_mark_batch"):
            for _id in batch_df['ISSUE ID'].astype(str).tolist():
                st.session_state.visited_ticket_ids.add(_id)
            st.rerun()

    with c4:
        if st.button("➡️ Next batch", key="btn_next_batch"):
            st.session_state.batch_cursor = end if end < len(seq_df) else 0
            st.rerun()

# ---------------- Step 6: Map ----------------
st.subheader("Step 6: Map")

show_thumbs = st.checkbox("Show image thumbnails in popups (may slow the map)", value=False, key="cb_thumbs")

# Using CartoDB positron tiles here as well
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
mc = MarkerCluster(name="Tickets").add_to(m)

plot_df = (gdf_all.drop(columns=["geometry"]) if "geometry" in getattr(gdf_all, "columns", []) else gdf_all).copy()

batch_ids = set(batch_df['ISSUE ID'].astype(str).tolist()) if not batch_df.empty else set()
first_in_batch = str(batch_df.iloc[0]['ISSUE ID']) if not batch_df.empty else None

if origin_lat is not None and origin_lon is not None:
    folium.Marker([origin_lat, origin_lon], popup="Start here",
                  icon=folium.Icon(color="green", icon="flag")).add_to(m)

for _, row in plot_df.iterrows():
    rid = str(row['ISSUE ID'])
    lat, lon = float(row['LATITUDE']), float(row['LONGITUDE'])

    if rid == first_in_batch:
        color, size = 'green', 10
    elif rid in batch_ids:
        color, size = 'orange', 9
    elif rid in st.session_state.visited_ticket_ids:
        color, size = 'gray', 7
    elif rid in st.session_state.skipped_ticket_ids:
        color, size = 'purple', 7
    else:
        color, size = 'red', 7

    before_url = str(row.get('BEFORE PHOTO', '') or '').strip()
    after_url = str(row.get('AFTER PHOTO', '') or '').strip()
    ward = escape(str(row.get('WARD', '') or ''))
    status = escape(str(row.get('STATUS', '') or ''))
    cluster = row.get('CLUSTER NUMBER', '')
    cluster = int(cluster) if isinstance(cluster, (int, float)) and not pd.isna(cluster) else ''

    parts = [
        f"<b>Issue ID:</b> {escape(rid)}",
        f"<b>Status:</b> {status}",
        f"<b>Ward:</b> {ward}",
        f"<b>Lat, Lon:</b> {lat:.6f}, {lon:.6f}",
    ]
    if cluster != '':
        parts.insert(0, f"<b>Cluster:</b> {cluster}")

    if is_url(before_url):
        parts.append(f"<a href='{before_url}' target='_blank' rel='noopener noreferrer'>Evidence photo</a>")
        if show_thumbs:
            parts.append(f"<div><img src='{before_url}' loading='lazy' referrerpolicy='no-referrer' style='max-width:220px;border:1px solid #ccc;border-radius:6px;'/></div>")

    popup_html = "<div style='font-size:13px;line-height:1.25'>" + "<br>".join(parts) + "</div>"
    popup = folium.Popup(popup_html, max_width=280)

    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=popup
    ).add_to(mc)

st_folium(m, width="100%", height=500, key="main_map_render")
m.save("Clustering_Application_Map.html")

# ---------------- Step 7: Downloads ----------------
st.subheader("Step 7: Downloads")
if not clustered.empty:
    with open("Clustering_Application_Summary.xlsx", "rb") as f:
        st.download_button("Download Excel", f, file_name="Clustering_Application_Summary.xlsx", key="dl_excel")

with open("Clustering_Application_Map.html", "rb") as f:
    st.download_button("Download Map (HTML)", f, file_name="Clustering_Application_Map.html", mime="text/html", key="dl_html")
