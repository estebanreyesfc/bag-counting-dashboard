import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Bag Counting Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("results_with_videos.csv")

    for i in [1, 2, 3]:
        df[f"error_belt_{i}"] = np.where(
            df[f"belt_{i}_manual"].notna(),
            df[f"belt_{i}"].fillna(0) - df[f"belt_{i}_manual"],
            np.nan
        )

    df["total_error"] = df[
        ["error_belt_1", "error_belt_2", "error_belt_3"]
    ].abs().sum(axis=1, skipna=True)

    return df

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Filters")

df["camera"] = df["name"].str.extract(r'(Camera\d+)')

folders = st.sidebar.multiselect(
    "Folder",
    df["folder"].dropna().unique(),
    default=df["folder"].dropna().unique()
)

cameras = st.sidebar.multiselect(
    "Camera",
    df["camera"].dropna().unique(),
    default=df["camera"].dropna().unique()
)

df_filtered = df[
    (df["folder"].isin(folders)) &
    (df["camera"].isin(cameras))
]

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(df):

    correct, total = 0, 0
    errors = []
    overcount, undercount = 0, 0

    for _, row in df.iterrows():

        pred = [row.get("belt_1"), row.get("belt_2"), row.get("belt_3")]
        gt   = [row.get("belt_1_manual"), row.get("belt_2_manual"), row.get("belt_3_manual")]

        pred = [p for p in pred if pd.notna(p) and p != 0]
        gt   = [g for g in gt if pd.notna(g) and g != 0]

        if not pred or not gt:
            continue

        pred, gt = sorted(pred), sorted(gt)

        for i in range(min(len(pred), len(gt))):
            diff = pred[i] - gt[i]

            if diff == 0:
                correct += 1
            else:
                errors.append(diff)

            if diff > 0:
                overcount += diff
            elif diff < 0:
                undercount += abs(diff)

            total += 1

    accuracy = correct / total if total else np.nan
    return accuracy, overcount, undercount, total, errors


accuracy, overcount, undercount, total, errors = compute_metrics(df_filtered)

# -----------------------------
# KPIs
# -----------------------------
st.title("📦 Bag Counting — Production Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Global Accuracy", f"{accuracy:.2%}" if pd.notna(accuracy) else "N/A")
c2.metric("Comparisons", total)
c3.metric("Overcount", int(overcount))
c4.metric("Undercount", int(undercount))

# -----------------------------
# 📊 ERROR DISTRIBUTION (NEW)
# -----------------------------
st.subheader("📊 Error Distribution")

if errors:
    err_df = pd.DataFrame({"error": errors})
    fig = px.histogram(err_df, x="error", nbins=30)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No errors to display")

# -----------------------------
# 🔥 WORST CASES (CLICKABLE)
# -----------------------------
st.subheader("🚨 Worst Videos")

worst = df_filtered.sort_values("total_error", ascending=False).head(10)

selected_worst = st.selectbox(
    "Select worst video (auto loads below 👇)",
    worst["name"],
    key="worst_selector"
)

st.dataframe(
    worst[[
        "name", "total_error",
        "error_belt_1", "error_belt_2", "error_belt_3"
    ]],
    width="stretch"
)

# -----------------------------
# CLEAN TABLE
# -----------------------------
st.subheader("📋 Full Results (Only Real Belts)")

display_df = df_filtered.copy()

belt_cols = [
    "belt_1", "belt_2", "belt_3",
    "belt_1_manual", "belt_2_manual", "belt_3_manual"
]

for i in [1, 2, 3]:
    mask = display_df[f"belt_{i}_manual"].isna()
    display_df.loc[mask, [f"belt_{i}", f"belt_{i}_manual"]] = np.nan

for col in belt_cols:
    display_df[col] = display_df[col].replace(0, np.nan)

display_df = display_df.dropna(subset=belt_cols, how="all")

def highlight(row):
    if row["total_error"] > 5:
        return ["background-color: #5c1a1a"] * len(row)
    return [""] * len(row)

st.dataframe(display_df.style.apply(highlight, axis=1), width="stretch")

# -----------------------------
# 🎥 VIDEO INSPECTOR (AUTO-SYNC)
# -----------------------------
st.subheader("🎥 Video Inspector")

# 🔥 Use worst selection as default
video_options = df_filtered["name"].tolist()

default_index = video_options.index(selected_worst) if selected_worst in video_options else 0

selected_video = st.selectbox(
    "Select a video",
    video_options,
    index=default_index
)

video_row = df_filtered[df_filtered["name"] == selected_video].iloc[0]

col1, col2 = st.columns([2, 1])

with col1:
    if pd.notna(video_row["video_url"]):
        st.video(video_row["video_url"])
    else:
        st.warning("No video URL available")

with col2:
    st.markdown("### 📊 Counts")

    for i in [1, 2, 3]:
        manual = video_row[f"belt_{i}_manual"]
        pred = video_row[f"belt_{i}"]
        err = video_row[f"error_belt_{i}"]

        if pd.notna(manual) and manual > 0:

            # 🎯 COLOR LOGIC
            if err == 0:
                color = "🟢"
            elif err > 0:
                color = "🔴"  # overcount
            else:
                color = "🟡"  # undercount

            st.write(f"{color} **Belt {i}:** {int(pred)} / {int(manual)}")

    st.markdown("### ⚠️ Errors")

    for i in [1, 2, 3]:
        err = video_row[f"error_belt_{i}"]

        if pd.notna(err) and err != 0:
            if err > 0:
                st.write(f"🔴 B{i} Overcount: +{int(err)}")
            else:
                st.write(f"🟡 B{i} Undercount: {int(err)}")


# -----------------------------
# 🎯 FRAME-LEVEL DEBUG (NEXT LEVEL)
# -----------------------------
st.subheader("🎯 Frame-level Error Inspection")

if "frame_error" in df_filtered.columns:

    video_frames = df_filtered[df_filtered["name"] == selected_video]

    if not video_frames.empty:

        fig = px.line(
            video_frames,
            x="frame",
            y="frame_error",
            title="Error over time"
        )

        st.plotly_chart(fig, width="stretch")

        worst_frame = video_frames.loc[
            video_frames["frame_error"].idxmax()
        ]

        st.metric("Worst Frame Error", worst_frame["frame_error"])
        st.write(f"Frame: {int(worst_frame['frame'])}")

else:
    st.info("Frame-level data not available")