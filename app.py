from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump
import streamlit as st
import pandas as pd
import duckdb

# Page setup
st.set_page_config(
    page_title="Brazil Tourism", page_icon="🇧🇷",
    layout="wide", initial_sidebar_state="auto"
)

# Load dataset and model from shared module
from data import load_data, get_model

DF = load_data()
model, scaler, model_mae = get_model(DF)

st.markdown("<h2><b>Brazil Tourism Explorer</b></h2>", unsafe_allow_html=True)
st.markdown("*Explore tourism trends in Brazil based on different conditions*")
st.divider()
st.subheader("Filter")
st.write("Select conditions to check statistics")

col1, col2, col3 = st.columns([5, 3, 2], gap="medium")

with col1:
    access_options = DF["Access_road"].unique().tolist()
    selected_access = st.multiselect(
        "**Road Access**",
        options=access_options,
        default=[x for x in access_options if x != "Unknown"]
    )

    min_active = int(DF["Active"].min())
    max_active = int(DF["Active"].max())
    active_range = st.slider("**Active Range**", min_active, max_active, (min_active, max_active), step=1)

    min_passive = int(DF["Passive"].min())
    max_passive = int(DF["Passive"].max())
    passive_range = st.slider("**Passive Range**", min_passive, max_passive, (min_passive, max_passive), step=1)
    
with col2:
    gender_options = DF["Sex"].unique().tolist()
    selected_gender = st.multiselect(
        "**Gender**",
        options=gender_options,
        default=[x for x in gender_options if x != "Unknown"]
    )

    min_cost = float(DF["Travel_cost"].min())
    max_cost = float(DF["Travel_cost"].max())
    travel_cost_range = st.slider("**Travel Cost Range**", min_cost, max_cost, (min_cost, max_cost))

with col3:
    min_age = int(DF["Age"].min())
    max_age = int(DF["Age"].max())
    age_range = st.slider("**Age Range**", min_age, max_age, (min_age, max_age))

    min_income = float(DF["Income"].min())
    max_income = float(DF["Income"].max())
    income_range = st.slider("**Income Range**", min_income, max_income, (min_income, max_income))


# Create the filtered dataframe whenever a filter is changed. Works as a query that applies all filters
filtered_df = DF[
        (DF["Age"].between(age_range[0], age_range[1])) &
        (DF["Income"].between(income_range[0], income_range[1])) &
        (DF["Travel_cost"].between(travel_cost_range[0], travel_cost_range[1])) &
        (DF["Active"].between(active_range[0], active_range[1])) &
        (DF["Passive"].between(passive_range[0], passive_range[1])) &
        (DF["Sex"].isin(selected_gender)) &
        (DF["Access_road"].isin(selected_access))
    ]

st.subheader("Results")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Average Income", f"{filtered_df['Income'].mean():.1f}")
with col2:
    st.metric("Average Travel Cost", f"{filtered_df['Travel_cost'].mean():.1f}")
with col3:
    st.metric("Average 'Active' Activities", f"{filtered_df['Active'].mean():.1f}")
with col4:
    st.metric("Rows Matching Filters", len(filtered_df))

st.markdown("\n")
with st.container(border=True, height=290):
    st.table(filtered_df)

st.divider()

st.subheader("Run SQL query on dataset")
with st.container(border=True, height=290, gap="small"):
    col1, col2, col3 = st.columns([2, 3, 5], gap="small")

    with col1:
        st.markdown("**Data preview**")
        st.dataframe(DF, use_container_width=False, height=210, hide_index=True)

    with col2:
        st.markdown("**Enter SQL here:**")
        query = st.text_area("", height=210, width=400, label_visibility="collapsed")

    # Use dataframe as a SQL table with duckdb
    with col3:
        st.markdown("**Queried dataset:**")
        if query:
            con = duckdb.connect()
            con.register('df', DF)
            try:
                result_df = con.execute(query).fetchdf()
                st.dataframe(result_df, use_container_width=True, height=210, hide_index=True)
            except Exception as e:
                st.error(e)
            con.close()

# Small status about the regression model
st.info(f"KNN regression model trained (MAE ≈ {model_mae:.2f}). To view charts, open the pages in the left sidebar (Trips vs Age, Income Pie).")