import streamlit as st
from data import load_data

st.set_page_config(page_title="Trips vs Age", layout="wide")
st.title("✈️ Trips vs Age")

DF = load_data()

st.caption("This chart shows how the number of trips varies by tourist age. It helps identify travel patterns among different age groups.")

# Optional: filter out unrealistic ages (if dataset has noise)
DF_chart = DF[DF["Age"].between(10, 90)]

# Create the chart
st.scatter_chart(
    DF_chart,
    x="Age",
    y="Trips",
)

# Small interpretation summary
avg_trips_young = DF_chart[DF_chart["Age"] < 30]["Trips"].mean()
avg_trips_senior = DF_chart[DF_chart["Age"] > 60]["Trips"].mean()

st.write(f"🧭 *Average trips for tourists under 30:* **{avg_trips_young:.1f}**")
st.write(f"🏖️ *Average trips for tourists over 60:* **{avg_trips_senior:.1f}**")