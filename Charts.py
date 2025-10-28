from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import os
import matplotlib.pyplot as plt
import plotly.express as px

# Specify datatset path here
FILE_PATH = "C:/Users/dalim/Desktop/streamlit_stuff/dataset_190_braziltourism.csv"
df = pd.read_csv(FILE_PATH)

# Page setup
st.set_page_config(
    page_title="Brazil Tourism", page_icon="🇧🇷",
    layout="wide", initial_sidebar_state="auto"
)
st.image(os.path.join(os.getcwd(),"static","brazil.jpg"), width=700)
# Load dataset
@st.cache_data
def clean_data(df):
    # Classification cleaning: 
    # Convert 0 to Male, 1 to Female using replace()
    # Categorize all misclassified data as Unknown. where() replaces the value with Unkown if its not in valid_map 
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
def perceptron(df):
    if os.path.exists("knn_model.joblib") and os.path.exists("scaler.joblib"):
        knn_reg = load("knn_model.joblib")
        scaler = load("scaler.joblib")

    df = df.copy()
    df = pd.get_dummies(df, columns=["Sex", "Access_road"], drop_first=False)

    Y = df["Trips"]
    X = df.drop(columns=["Trips"])

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_reg = KNeighborsRegressor(n_neighbors=1)
    knn_reg.fit(X_train_scaled, y_train)

    y_pred = knn_reg.predict(X_test_scaled)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)
    ACCURACY = accuracy_score(y_test, y_pred)
    PRECISION = precision_score(y_test, y_pred, average='macro')
    RECALL = recall_score(y_test, y_pred, average='macro')
    F1 = f1_score(y_test, y_pred, average='macro')

    dump(knn_reg, "knn_model.joblib")
    dump(scaler, "scaler.joblib")

    return (knn_reg, scaler, MAE, RMSE, ACCURACY, PRECISION, RECALL, F1)

df = clean_data(df)
model = perceptron(df)


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

access_options = df["Access_road"].unique().tolist()
gender_options = df["Sex"].unique().tolist()

with col1:
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
        (df["Sex"].isin(selected_gender)) &
        (df["Income"].between(income_range[0], income_range[1])) &
        (df["Travel_cost"].between(travel_cost_range[0], travel_cost_range[1])) &
        (df["Access_road"].isin(selected_access)) &
        (df["Active"].between(active_range[0], active_range[1])) &
        (df["Passive"].between(passive_range[0], passive_range[1]))
    ].copy()

# Fix integers for readability
for col in ["Age", "Active", "Passive", "Trips"]:
    if col in filtered_df.columns:
        filtered_df[col] = filtered_df[col].astype(int)

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

st.markdown("**Model Performance**")
with st.container(border=True, height=200, gap="small"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{model[4] * 100:.2f}%")

    with col2:
        st.metric("Precision", f"{model[5] * 100:.2f}%")

    with col3:
        st.metric("Recall", f"{model[6] * 100:.2f}%")

    with col4:
        st.metric("F1-Score", f"{model[7] * 100:.2f}%")

    col5, col6 = st.columns(2)

    with col5:
        st.metric("Mean Absolute Error (MAE)", f"{model[2]:.3f}")

    with col6:
        st.metric("Root Mean Squared Error (RMSE)", f"{model[3]:.3f}")

st.subheader("Predict amount of Trips")
st.caption("Using our trained regression model we will determine how many trips you would need to take based on provided info. *Note: Leaving a field blank counts as excluding the feature*")

col1, col2, col3 = st.columns([5, 3, 2], gap="medium")

with col1:
    selected_access_val = st.selectbox("**Road Access**", access_options, key="access_input")
    active_val = st.text_input("**Active value**", key="active_input")
    passive_val = st.text_input("**Passive value**", key="passive_input")

with col2:
    selected_gender_val = st.selectbox("**Gender**", gender_options, key="gender_input")
    travel_cost_val = st.text_input("**Travel Cost value**", key="travel_cost_input")

with col3:
    age_val = st.text_input("**Age value**", key="age_input")
    income_val = st.text_input("**Income value**", key="income_input")

income_num = float(income_val) if income_val else 0.0
logged_income_num = np.log(income_num) if income_num > 0 else 0.0

new_tourist = pd.DataFrame({
    "Age": [age_val if age_val else 0],
    "Income": [income_num],
    "Travel_cost": [float(travel_cost_val) if travel_cost_val else 0.0],
    "Active": [active_val if active_val else 0],
    "Passive": [passive_val if passive_val else 0],
    "Logged_income": [logged_income_num],
    "Sex_Female": [1 if selected_gender_val == "Female" else 0],
    "Sex_Male": [1 if selected_gender_val == "Male" else 0],
    "Sex_Unknown": [1 if selected_gender_val == "Unknown" else 0],
    "Access_road_Bad": [1 if selected_access_val == "Bad" else 0],
    "Access_road_Good": [1 if selected_access_val == "Good" else 0],
})

# Apply the same scaling used during training
new_tourist_scaled = model[1].transform(new_tourist)

# Predict Trips
predicted_trips = model[0].predict(new_tourist_scaled)
st.success(f"Predicted Trips: {predicted_trips[0]:.1f}")

st.divider()

st.subheader("Run SQL on dataset")
with st.container(border=True, height=290, gap="small"):
    col1, col2, col3 = st.columns([2, 3, 5], gap="small")

with col1:
    st.markdown("**Data preview**")
    st.dataframe(
        df, 
        use_container_width=True, # <--- CHANGE THIS TO TRUE
        height=210, 
        hide_index=True
    )

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


