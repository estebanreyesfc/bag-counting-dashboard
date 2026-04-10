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

df["camera"] = df["name"].str.extract(r'(Camera\d+)')[0]

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
].copy()

df_filtered = df_filtered.reset_index(drop=True)

# Fix Drive URLs
df_filtered["video_url"] = df_filtered["video_url"].str.extract(r'id=([^&]+)')[0] \
    .apply(lambda x: f"https://drive.google.com/file/d/{x}/preview" if pd.notna(x) else np.nan)

# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(df):
    correct, total = 0, 0
    errors, abs_errors = [], []
    overcount, undercount = 0, 0
    within_1, within_2 = 0, 0
    big_errors = 0

    all_pred, all_gt = [], []

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

            errors.append(diff)
            abs_errors.append(abs(diff))
            all_pred.append(pred[i])
            all_gt.append(gt[i])

            if diff == 0:
                correct += 1
            if abs(diff) <= 1:
                within_1 += 1
            if abs(diff) <= 2:
                within_2 += 1
            if abs(diff) >= 3:
                big_errors += 1

            if diff > 0:
                overcount += diff
            elif diff < 0:
                undercount += abs(diff)

            total += 1

    return {
        "accuracy": correct / total if total else np.nan,
        "acc_1": within_1 / total if total else np.nan,
        "acc_2": within_2 / total if total else np.nan,
        "mae": np.mean(abs_errors) if abs_errors else np.nan,
        "bias": (overcount - undercount) / total if total else np.nan,
        "failure_rate": big_errors / total if total else np.nan,
        "overcount": overcount,
        "undercount": undercount,
        "total": total,
        "errors": errors,
        "all_pred": all_pred,
        "all_gt": all_gt,
        "correlation": np.corrcoef(all_gt, all_pred)[0, 1] if len(all_gt) > 1 else np.nan
    }


metrics = compute_metrics(df_filtered)

# -----------------------------
# KPIs
# -----------------------------
st.title("📦 Bag Counting — Production Dashboard")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Exact", f"{metrics['accuracy']:.1%}")
c2.metric("±1 Bag", f"{metrics['acc_1']:.1%}")
c3.metric("±2 Bags", f"{metrics['acc_2']:.1%}")
c4.metric("MAE", f"{metrics['mae']:.2f}")
c5.metric("Bias", f"{metrics['bias']:+.2f}")
c6.metric("Fail ≥3", f"{metrics['failure_rate']:.1%}")

c7, c8, c9 = st.columns(3)
c7.metric("Comparisons", metrics["total"])
c8.metric("Overcount", int(metrics["overcount"]))
c9.metric("Undercount", int(metrics["undercount"]))

# -----------------------------
# CORRELATION
# -----------------------------
st.subheader("🔗 Prediction vs Ground Truth")

if len(metrics["all_gt"]) > 0:
    corr_df = pd.DataFrame({
        "Ground Truth": metrics["all_gt"],
        "Prediction": metrics["all_pred"]
    })

    fig = px.scatter(
        corr_df,
        x="Ground Truth",
        y="Prediction",
        trendline="ols",
        title=f"Correlation: {metrics['correlation']:.3f}"
    )

    st.plotly_chart(fig, width="stretch")

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col_left, col_right = st.columns([1, 2])

# -----------------------------
# FILE BROWSER
# -----------------------------
with col_left:
    st.subheader("📂 Browser")

    def build_tree(df):
        tree = {}
        for idx, row in df.iterrows():
            folder_path = str(row["folder"]) if pd.notna(row["folder"]) else "root"
            parts = folder_path.split("/")

            current = tree
            for part in parts:
                current = current.setdefault(part, {})

            current.setdefault("_videos", []).append((idx, row["name"]))
        return tree

    def render_tree(tree):
        for key, value in tree.items():
            if key == "_videos":
                for idx, video in value:
                    is_selected = (
                        "selected_video_idx" in st.session_state and
                        st.session_state["selected_video_idx"] == idx
                    )

                    label = f"👉 {video}" if is_selected else f"🎥 {video}"

                    if st.button(label, key=f"tree_{idx}", width="stretch"):
                        st.session_state["selected_video_idx"] = idx
                continue

            with st.expander(f"📁 {key}"):
                render_tree(value)

    render_tree(build_tree(df_filtered))

# -----------------------------
# VIDEO INSPECTOR
# -----------------------------
with col_right:
    st.subheader("🎥 Video Inspector")

    if "selected_video_idx" not in st.session_state and len(df_filtered) > 0:
        st.session_state["selected_video_idx"] = df_filtered.index[0]

    selected_idx = st.session_state.get("selected_video_idx")

    if selected_idx not in df_filtered.index:
        st.warning("Select a video")
        st.stop()

    video_row = df_filtered.loc[selected_idx]

    col1, col2 = st.columns([2, 1])

    with col1:
        if pd.notna(video_row["video_url"]):
            st.iframe(video_row["video_url"], height=500)

    with col2:
        st.markdown("### 📊 Counts")

        for i in [1, 2, 3]:
            manual = video_row[f"belt_{i}_manual"]
            pred = video_row[f"belt_{i}"]
            err = video_row[f"error_belt_{i}"]

            if pd.notna(manual) and manual > 0:
                color = "🟢" if err == 0 else "🔴" if err > 0 else "🟡"
                st.write(f"{color} B{i}: {int(pred)} / {int(manual)}")

# -----------------------------
# ERROR DISTRIBUTION
# -----------------------------
st.subheader("📊 Error Distribution")

if metrics["errors"]:
    fig = px.histogram(pd.DataFrame({"error": metrics["errors"]}), x="error")
    st.plotly_chart(fig, width="stretch")

# -----------------------------
# WORST CASES
# -----------------------------
st.subheader("🚨 Worst Videos")

worst = df_filtered.sort_values("total_error", ascending=False).head(10)

selected_worst = st.selectbox("Select worst", worst["name"])

if selected_worst:
    idx = df_filtered[df_filtered["name"] == selected_worst].index[0]
    st.session_state["selected_video_idx"] = idx

st.dataframe(worst[["name", "total_error"]], width="stretch")
