from app import df
import plotly.express as px
import streamlit as st
import pandas as pd
import os

df = df.copy()

# --- Page setup ---
st.set_page_config(page_title="Brazil Data Pie Chart", layout="wide")
st.title("🥧 Proportion of Total Travel Cost by Income Group")

# --- Clean numeric columns ---
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
df["Travel_cost"] = pd.to_numeric(df["Travel_cost"], errors="coerce")
df = df.dropna(subset=["Income", "Travel_cost"])

# --- Group incomes into bins ---
df["Income_group"] = pd.cut(
    df["Income"],
    bins=[0, 500, 1000, 1500, 2000, df["Income"].max()],
    labels=["0–500", "500–1000", "1000–1500", "1500–2000", "2000+"]
)

# --- Aggregate total travel costs ---
agg = df.groupby("Income_group", as_index=False)["Travel_cost"].sum()

# --- Custom color palette (modern pastel gradient) ---
custom_colors = ["#0099C6", "#33B679", "#FFBB00", "#FF7043", "#AB47BC"]

# --- Create beautiful donut chart ---
fig = px.pie(
    agg,
    names="Income_group",
    values="Travel_cost",
    title="Share of Total Travel Cost by Income Group",
    hole=0.5,
    color_discrete_sequence=custom_colors
)

# --- Add better labels and layout tweaks ---
fig.update_traces(
    textposition="outside",
    textinfo="percent+label",
    pull=[0.03, 0.03, 0.03, 0.03, 0.03],  # slight pop-out for all slices
    marker=dict(line=dict(color="#1f1f1f", width=2))  # clean slice borders
)

fig.update_layout(
    paper_bgcolor="#0E1117",  # dark dashboard background
    plot_bgcolor="#0E1117",
    font=dict(color="#FAFAFA", size=14),
    title_font=dict(size=22, color="#00B4D8", family="Arial Black"),
    showlegend=True,
    legend_title_text="Income Range"
)

# --- Show chart ---
st.plotly_chart(fig, use_container_width=True)
