
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Drug Review Analyzer",
    page_icon="💊",
    layout="wide"
)

# ---------- Styling ----------
st.markdown("""
<style>
.stApp { background-color: #f4f8ff; }
h1, h2, h3 { color: #12336b; }
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d9e7fa;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("💊 Patient Drug Review Analytics")
st.caption(
    "Analyze patient feedback, ratings and healthcare trends "
    "using data science."
)

# ---------- Dataset loading ----------
root = Path(__file__).resolve().parent

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose section",
        ["Overview", "Sentiment Analysis", "Rating Analysis",
         "Health Conditions", "Raw Data"]
    )
    st.divider()
    st.caption("Upload your own CSV or TSV dataset.")

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "tsv", "txt"]
    )

def load_file(file):
    if hasattr(file, "seek"):
        file.seek(0)
    return pd.read_csv(file, sep=None, engine="python")

df = None
source_name = ""

if uploaded is not None:
    try:
        df = load_file(uploaded)
        source_name = uploaded.name
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()
else:
    candidates = sorted(
        list(root.rglob("*.tsv")) + list(root.rglob("*.csv")),
        key=lambda p: (
            0 if "train" in p.name.lower() else 1,
            len(p.parts)
        )
    )

    if candidates:
        for path in candidates:
            try:
                temp = pd.read_csv(path, sep=None, engine="python")
                cols = {str(c).strip().lower() for c in temp.columns}
                if "rating" in cols and any(
                    "review" in c or "comment" in c for c in cols
                ):
                    df = temp
                    source_name = str(path.relative_to(root))
                    break
            except Exception:
                continue

if df is None:
    st.warning(
        "No suitable dataset found. Upload your drug review "
        "CSV or TSV file using the sidebar."
    )
    st.info(
        "Expected columns include rating and a review-text column. "
        "The DrugLib dataset may contain benefitsReview, "
        "sideEffectsReview and commentsReview."
    )
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
cols = {c.lower(): c for c in df.columns}

if "rating" not in cols:
    st.error("The dataset must contain a rating column.")
    st.stop()

rating_col = cols["rating"]
df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
df = df.dropna(subset=[rating_col]).copy()

review_cols = [
    c for c in df.columns
    if any(word in c.lower() for word in
           ["review", "comment", "benefit", "sideeffect"])
]

if not review_cols:
    st.error("No review-text column found in this dataset.")
    st.stop()

df["Review_Text"] = (
    df[review_cols].fillna("").astype(str)
    .agg(" ".join, axis=1).str.strip()
)

df = df[df["Review_Text"] != ""].copy()

if df.empty:
    st.error("No usable patient reviews were found.")
    st.stop()

# Rating-based sentiment, not NLP sentiment prediction
df["Sentiment"] = df[rating_col].apply(
    lambda x: "Positive" if x >= 7
    else "Neutral" if x >= 4
    else "Negative"
)

condition_col = cols.get("condition")
drug_col = cols.get("urldrugname") or cols.get("drugname")

# ---------- Sidebar filters ----------
with st.sidebar:
    st.subheader("Filters")

    sentiments = st.multiselect(
        "Sentiment",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )

    filtered = df[df["Sentiment"].isin(sentiments)].copy()

    if condition_col:
        conditions = sorted(
            filtered[condition_col].dropna().astype(str).unique()
        )
        selected = st.multiselect(
            "Health conditions",
            conditions,
            default=[]
        )
        if selected:
            filtered = filtered[
                filtered[condition_col].astype(str).isin(selected)
            ]

# ---------- Overview ----------
if page == "Overview":
    st.subheader("Healthcare Insights Dashboard")
    st.caption(f"Dataset: {source_name}")

    total = len(filtered)
    average = filtered[rating_col].mean()
    positive_pct = (
        100 * (filtered["Sentiment"] == "Positive").mean()
        if total else 0
    )
    negative_pct = (
        100 * (filtered["Sentiment"] == "Negative").mean()
        if total else 0
    )

    a, b, c, d = st.columns(4)
    a.metric("Total Reviews", f"{total:,}")
    b.metric("Average Rating", f"{average:.2f}/10" if total else "N/A")
    c.metric("Positive Reviews", f"{positive_pct:.1f}%")
    d.metric("Negative Reviews", f"{negative_pct:.1f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Distribution")
        counts = filtered["Sentiment"].value_counts().reindex(
            ["Positive", "Neutral", "Negative"], fill_value=0
        )
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of reviews")
        ax.set_xlabel("Sentiment")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots()
        filtered[rating_col].plot(
            kind="hist", bins=10, ax=ax, edgecolor="white"
        )
        ax.set_xlabel("Patient rating")
        ax.set_ylabel("Number of reviews")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if condition_col:
        st.subheader("Most Reviewed Health Conditions")
        top = filtered[condition_col].dropna().value_counts().head(10)
        if not top.empty:
            fig, ax = plt.subplots()
            top.sort_values().plot(kind="barh", ax=ax)
            ax.set_xlabel("Number of reviews")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Recent Patient Reviews")
    display_cols = ["Review_Text", rating_col, "Sentiment"]
    if condition_col:
        display_cols.append(condition_col)
    if drug_col:
        display_cols.append(drug_col)

    st.dataframe(
        filtered[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

# ---------- Sentiment Analysis ----------
elif page == "Sentiment Analysis":
    st.subheader("Patient Sentiment Analysis")
    st.write(
        "Sentiment categories are derived from numerical ratings: "
        "7–10 Positive, 4–6 Neutral, and 1–3 Negative."
    )

    st.dataframe(
        filtered[["Review_Text", rating_col, "Sentiment"]],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Analyzed Reviews",
        filtered.to_csv(index=False).encode("utf-8"),
        "analyzed_reviews.csv",
        "text/csv"
    )

# ---------- Rating Analysis ----------
elif page == "Rating Analysis":
    st.subheader("Patient Rating Analysis")
    st.write(filtered[rating_col].describe().to_frame("Rating Statistics"))

    fig, ax = plt.subplots()
    filtered[rating_col].plot(kind="hist", bins=10, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of reviews")
    st.pyplot(fig)
    plt.close(fig)

# ---------- Health Conditions ----------
elif page == "Health Conditions":
    st.subheader("Health Condition Analysis")

    if condition_col:
        top = filtered[condition_col].dropna().value_counts().head(15)
        st.dataframe(
            top.rename("Review Count").to_frame(),
            use_container_width=True
        )

        fig, ax = plt.subplots()
        top.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of reviews")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("This dataset does not contain a condition column.")

# ---------- Raw Data ----------
elif page == "Raw Data":
    st.subheader("Dataset Explorer")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.write("Original dataset columns:", df.columns.tolist())
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Processed Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "processed_drug_reviews.csv",
        "text/csv"
    )

st.divider()
st.caption(
    "Educational data analysis only. Sentiment is inferred from ratings "
    "and does not establish medicine safety or effectiveness."
)

