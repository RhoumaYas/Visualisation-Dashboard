import streamlit as st 
import geopandas as gpd 
import pandas as pd 
import matplotlib as mpl
import pydeck as pdk



st.set_page_config(layout='wide')

#### functions to create and display map



@st.cache_data
def load_segments():
    gdf = gpd.read_file('../data/segment_model_final.shp') 
    gdf['geometry'] = gdf.geometry.simplify(tolerance=0.0001, preserve_topology=True)

    return gdf

@st.cache_data
def load_grid():
    gdf = gpd.read_file('../data/grid_model_final.shp') 
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
            <b>Delta:</b> {delta}             
            """},    
    map_style='light', map_provider='carto')
    
    return deck


# Load original data
seg_gdf = load_segments()
grid_gdf = load_grid()
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
        subcol1.image('./data/cm_segment_model.png')
        subcol2.image('./data/grid_model_confusion_matrices.png')
        
        
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)
        # st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        subcol3,_, subcol4 = st.columns([.4,.2,.4])
        subcol3.image('./data/RocAuc_segment.png')
        subcol4.image('./data/RocAuc_grid.png')
        
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)
        # st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        
        
        st.image('./data/percent_missclassified_per_dist.png', use_container_width=False)
        st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
        st.image('./data/mean_delta_by_district.png')
    
    
    
with tab2:
    st.header('Visualisation of accident data')
    st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)
        # st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        
        
    st.image('./data/accident_distribution_by_weekday_and_time_heatmap.png', use_container_width=False)
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    st.image('./data/accidents_per_districts.png')
    st.markdown("<div style='margin-bottom: 150px;'></div>", unsafe_allow_html=True)
    st.image('./data/accidents_per_season.png')
    
