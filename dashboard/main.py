import streamlit as st 
import geopandas as gpd 
import pandas as pd 
import matplotlib as mpl
import pydeck as pdk
import os



st.set_page_config(layout='wide')

#### functions to create and display map



@st.cache_data
def load_segments(shapefile_path):
    gdf = gpd.read_file(shapefile_path) 
    gdf['geometry'] = gdf.geometry.simplify(tolerance=0.0001, preserve_topology=True)

    return gdf

@st.cache_data
def load_grid(shapefile_path):
    gdf = gpd.read_file(shapefile_path) 
    gdf['geometry'] = gdf.geometry.simplify(tolerance=0.0001, preserve_topology=True)

    return gdf


def prep_geodf(_gdf):
    gdf_centroids = _gdf.copy()
    gdf_centroids["centroid"] = _gdf.geometry.centroid

    # Step 2: Convert centroids to WGS84 for map center
    centroids_ll = gdf_centroids.set_geometry("centroid").to_crs(epsg=4326)
    center_lat = centroids_ll.geometry.y.mean()
    center_lon = centroids_ll.geometry.x.mean()

    # Step 3: Convert original geometry to EPSG:4326 for display
    gdf_display = _gdf.to_crs(epsg=4326)

    color_map_1 = get_color_map(gdf_display, "risk_cat", cmap_name="Reds")
    gdf_display["fill_color_rc"] = gdf_display["risk_cat"].map(color_map_1)
    
    color_map_2 = get_color_map(gdf_display, "pred", cmap_name="Reds")
    gdf_display["fill_color_pred"] = gdf_display["pred"].map(color_map_2)
    
    color_map_3 = get_color_map(gdf_display, "delta", cmap_name="PiYG")
    gdf_display["fill_color_delta"] = gdf_display["delta"].map(color_map_3)
    
    geojson = gdf_display.__geo_interface__
    
    return geojson,center_lat, center_lon


def get_color_map(_gdf, column, cmap_name="Reds"):
    unique_vals = sorted(_gdf[column].unique())
    cmap = mpl.colormaps[cmap_name]  # Updated API
    n = len(unique_vals)
    val_to_color = {
        val: [int(c * 255) for c in cmap(i / max(n - 1, 1))[:3]] + [255]  # RGBA with alpha
        for i, val in enumerate(unique_vals)
    }
    return val_to_color


def create_map(_geojson, _center_lat, _center_lon, color):
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        _geojson,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        line_width_min_pixels=1,
        get_fill_color=f"properties.{color}",
        get_line_color=[0, 0, 0],
    )

    # Step 5: Map view
    view_state = pdk.ViewState(
        latitude=_center_lat,
        longitude=_center_lon,
        zoom=13,
        pitch=0,
    )  

    deck = pdk.Deck(
    layers=[polygon_layer],
    initial_view_state=view_state,
    tooltip={"html": """
            <b>Risk category:</b> {risk_cat}<br>
            <b>Predicted category:</b> {pred}<br>
            <b>Delta:</b> {delta}<br> 
            <b>District:</b> {district}<br>
            <b>Slope mean:</b> {slope_mean}<br>
            <b>Slope max:</b> {slope_max}<br>
            <b>Temporegime:</b> {temporegim}<br>
            <b>Trams:</b> {trams}<br>
            <b>Degree intersection:</b> {deg_inter}<br>
            <b>Bicycle lanes:</b> {bicycle_la}<br>
            <b>Traffic volume:</b> {traf_vol}<br>
            <b>Crossings:</b> {crossings}<br>
            <b>Lanes:</b> {lanes}<br>
            <b>Traffic signals:</b> {traffic_si}<br>
            <b>Lighting:</b> {lit}<br>
            <b>Road smoothness:</b> {road_smoot}<br>
            <b>Train station distance:</b> {train_stat}<br>
            <b>Office distance:</b> {office_dis}<br>
            <b>School distance:</b> {school__di}<br>
            <b>Water distance:</b> {water_dist}<br>
            <b>Pub/Bar distance:</b> {pub_distan}<br>
            <b>Bike parking distance:</b> {bikeparkin}           
            """},    
    map_style='light', map_provider='carto')
    
    return deck


# Load original data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(BASE_DIR, "data", "segment_model_comb.shp")
grid_file_path = os.path.join(BASE_DIR, "data", "grid_model_comb.shp")

seg_gdf = load_segments(shapefile_path)
grid_gdf = load_grid(grid_file_path)
seg_geojson, seg_center_lat, seg_center_lon = prep_geodf(seg_gdf)
grd_geojson, grd_center_lat, grd_center_lon = prep_geodf(grid_gdf)



tab1, tab2, = st.tabs(['Modelling', 'Accident data analysis'])

with tab1:
    st.title('Predicting bicycle accidents in Zurich')

    col1, col2 = st.columns(2)
    model = col1.selectbox(label='chose model', options=['Segment based', 'Grid based'])
    view = col2.selectbox(label='chose view', options=['Actual risk catecory', 'Predicted risk category', 'Delta'])

    selected_poly=None

    geojson, lat, lon = (seg_geojson, seg_center_lat, seg_center_lon) if model == 'Segment based' else (grd_geojson, grd_center_lat, grd_center_lon)
    color_column = {
        'Actual risk catecory': 'fill_color_rc',
        'Predicted risk category': 'fill_color_pred',
        'Delta': 'fill_color_delta'
    }[view]

    st.pydeck_chart(create_map(geojson, lat, lon, color_column))
    
    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    with st.expander("Comparing model prediction results"):
        
        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        
        subcol1,_, subcol2 = st.columns([.45,.1,.45])
    
        subcol1.image(os.path.join(BASE_DIR, "data", "cm_segment_model.png"))
        subcol2.image(os.path.join(BASE_DIR, "data", "grid_model_confusion_matrices.png"))
        
        
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

        subcol3, _, subcol4 = st.columns([.4, .2, .4])
        subcol3.image(os.path.join(BASE_DIR, "data", "RocAuc_segment.png"))
        subcol4.image(os.path.join(BASE_DIR, "data", "RocAuc_grid.png"))
        subcol3.image(os.path.join(BASE_DIR, "data", "segment_model_features.jpg"))
        subcol4.image(os.path.join(BASE_DIR, "data", "grid_model_features.jpg"))

        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

        st.image(os.path.join(BASE_DIR, "data", "percent_missclassified_per_dist.png"), use_container_width=False)
        st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
        st.image(os.path.join(BASE_DIR, "data", "mean_delta_by_district.png"))

with tab2:
    st.header('Visualisation of accident data')
    st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

    st.image(os.path.join(BASE_DIR, "data", "distribution_of_accident_severity.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "most_common_accident_types.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "severity_vs_accident_type_correlation.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accidents_per_month.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accidents_per_weekday.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "severity_by_type_and_weekday.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accident_distribution_by_weekday_and_time_heatmap.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accidents_per_districts.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accident_densitiy_tram_distance.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accident_severity_tram_proximity.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accidents_by_speedzone.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "avg_accident_speed_limit.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accidents_per_season.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "severity_vs_season_correlation.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accident_count_year.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accident_count_month.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    st.image(os.path.join(BASE_DIR, "data", "accidents_by_hour.png"))
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    
    
    
