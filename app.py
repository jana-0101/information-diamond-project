from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from joblib import dump, load
import streamlit as st
import pandas as pd
import duckdb

# TODO
# Graphs, plots, etc of filtered dataframe...
# Test input values that will use the regression model
# More pages and info

# Page setup
st.set_page_config(
    page_title="Brazil Tourism", page_icon="🇧🇷",
    layout="wide", initial_sidebar_state="auto"
)

# Load dataset
@st.cache_data
def load_data():
    # Specify datatset path here
    df = pd.read_csv("dataset_190_braziltourism(1).csv")
    df.columns = df.columns.str.replace(" ", "_")

    # Classification cleaning: Categorize all misclassified data as Unknown
    def clean_categorical(col, valid_map):
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].replace(valid_map)
        df[col] = df[col].where(df[col].isin(valid_map.values()), "Unknown")
    clean_categorical("Sex", {"0": "Male", "1": "Female"})
    clean_categorical("Access_road", {"0": "Bad", "1": "Good"})

    # Regression cleaning: Convert numerical values to strings, remove eveyrthing thats not a number with regex, then convert back to numbers
    def clean_numeric(df, col):
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(r"[^0-9.\-]", "", regex=True)
        df[col] = df[col].replace("", "0")
        df[col] = df[col].astype(float)
        return df
    for col in ["Age", "Income", "Travel_cost", "Active", "Passive", "Logged_income", "Trips"]:
        df = clean_numeric(df, col)
    
    return df

# Initliaze basic perceptron regression model to predict Trips
@st.cache_resource
def perceptron():
    df_encoded = pd.get_dummies(df, columns=["Sex", "Access_road"], drop_first=False)

    Y = df_encoded["Trips"]
    X = df_encoded.drop('Trips', axis=1).copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_reg = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_train_scaled, y_train)
    y_pred = knn_reg.predict(X_test_scaled)

    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)

    dump(knn_reg, "knn_model.joblib")
    dump(scaler, "scaler.joblib")

    return knn_reg, scaler

df = load_data()
model = perceptron()

st.markdown("<h2><b>Brazil Tourism Explorer</b></h2>", unsafe_allow_html=True)
st.markdown("*Explore tourism trends in Brazil based on different conditions*")
st.divider()
st.subheader("Filter")
st.write("Select conditions to check statistics")

# Make 3 columns for all features
# Use st.slider for numerical features and st.multiselect for categorial features
# For options in st.multiselect get all unique values and convert them to a list first
# For range in st.slider get min and max value first 

col1, col2, col3 = st.columns([5, 3, 2], gap="medium")

with col1:
    access_options = df["Access_road"].unique().tolist()
    selected_access = st.multiselect(
        "**Road Access**",
        options=access_options,
        default=[x for x in access_options if x != "Unknown"]
    )

    min_active = int(df["Active"].min())
    max_active = int(df["Active"].max())
    active_range = st.slider("**Active Range**", min_active, max_active, (min_active, max_active), step=1)

    min_passive = int(df["Passive"].min())
    max_passive = int(df["Passive"].max())
    passive_range = st.slider("**Passive Range**", min_passive, max_passive, (min_passive, max_passive), step=1)
    
with col2:
    gender_options = df["Sex"].unique().tolist()
    selected_gender = st.multiselect(
        "**Gender**",
        options=gender_options,
        default=[x for x in gender_options if x != "Unknown"]
    )

    min_cost = float(df["Travel_cost"].min())
    max_cost = float(df["Travel_cost"].max())
    travel_cost_range = st.slider("**Travel Cost Range**", min_cost, max_cost, (min_cost, max_cost))

with col3:
    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())
    age_range = st.slider("**Age Range**", min_age, max_age, (min_age, max_age))

    min_income = float(df["Income"].min())
    max_income = float(df["Income"].max())
    income_range = st.slider("**Income Range**", min_income, max_income, (min_income, max_income))


# Create the filtered dataframe whenever a filter is changed. Works as a query that applies all filters
filtered_df = df[
        (df["Age"].between(age_range[0], age_range[1])) &
        (df["Income"].between(income_range[0], income_range[1])) &
        (df["Travel_cost"].between(travel_cost_range[0], travel_cost_range[1])) &
        (df["Active"].between(active_range[0], active_range[1])) &
        (df["Passive"].between(passive_range[0], passive_range[1])) &
        (df["Sex"].isin(selected_gender)) &
        (df["Access_road"].isin(selected_access))
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
        st.dataframe(df, use_container_width=False, height=210, hide_index=True)

    with col2:
        st.markdown("**Enter SQL here:**")
        query = st.text_area("", height=210, width=400, label_visibility="collapsed")

    # Use dataframe as a SQL table with duckdb
    with col3:
        st.markdown("**Queried dataset:**")
        if query:
            con = duckdb.connect()
            con.register('df', df)
            try:
                result_df = con.execute(query).fetchdf()
                st.dataframe(result_df, use_container_width=True, height=210, hide_index=True)
            except Exception as e:
                st.error(e)
            con.close()


