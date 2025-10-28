import streamlit as st
from data import load_data
import plotly.express as px

st.set_page_config(page_title="Income Pie", layout="wide")
st.title("🥧 Proportion of Total Travel Cost by Income Group")

DF = load_data().copy()

# --- Clean numeric columns (already cleaned, but ensure numeric) ---
DF["Income"] = pd.to_numeric(DF["Income"], errors="coerce")
DF["Travel_cost"] = pd.to_numeric(DF["Travel_cost"], errors="coerce")
DF = DF.dropna(subset=["Income", "Travel_cost"]) 

# --- Group incomes into bins ---
DF["Income_group"] = pd.cut(
    DF["Income"],
    bins=[0, 500, 1000, 1500, 2000, DF["Income"].max()],
    labels=["0–500", "500–1000", "1000–1500", "1500–2000", "2000+"]
)

# --- Aggregate total travel costs ---
agg = DF.groupby("Income_group", as_index=False)["Travel_cost"].sum()

# --- Custom color palette (modern pastel gradient) ---
custom_colors = ["#0099C6", "#33B679", "#FFBB00", "#FF7043", "#AB47BC"]

# --- Create donut chart ---
fig = px.pie(
    agg,
    names="Income_group",
    values="Travel_cost",
    title="Share of Total Travel Cost by Income Group",
    hole=0.5,
    color_discrete_sequence=custom_colors
)

fig.update_traces(
    textposition="outside",
    textinfo="percent+label",
    pull=[0.03] * len(agg),
    marker=dict(line=dict(color="#1f1f1f", width=2))
)

fig.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(color="#FAFAFA", size=14),
    title_font=dict(size=22, color="#00B4D8", family="Arial Black"),
    showlegend=True,
    legend_title_text="Income Range"
)

st.plotly_chart(fig, use_container_width=True)