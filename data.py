import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump

@st.cache_data
def load_data(path="dataset_190_braziltourism.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(" ", "_")

    # Classification cleaning: Categorize all misclassified data as Unknown
    def clean_categorical(col, valid_map):
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].replace(valid_map)
        df[col] = df[col].where(df[col].isin(valid_map.values()), "Unknown")
    clean_categorical("Sex", {"0": "Male", "1": "Female"})
    clean_categorical("Access_road", {"0": "Bad", "1": "Good"})

    # Regression cleaning: Convert numerical values to strings, remove everything that's not a number with regex, then convert back to numbers
    def clean_numeric(df_local, col):
        df_local[col] = df_local[col].astype(str)
        df_local[col] = df_local[col].str.replace(r"[^0-9.\-]", "", regex=True)
        df_local[col] = df_local[col].replace("", "0")
        df_local[col] = df_local[col].astype(float)
        return df_local
    for col in ["Age", "Income", "Travel_cost", "Active", "Passive", "Logged_income", "Trips"]:
        df = clean_numeric(df, col)
    
    return df

@st.cache_resource
def get_model(df):
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

    # Persist model and scaler (optional)
    dump(knn_reg, "knn_model.joblib")
    dump(scaler, "scaler.joblib")

    return knn_reg, scaler, MAE