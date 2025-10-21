import os
import io
import time
import pathlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.preprocess import clean_dataframe
from utils.categorize import HybridCategorizer, ALLOWED_CATEGORIES
from utils.visualize import plot_category_bar, plot_category_pie



# ---------- Page Setup ----------
st.set_page_config(page_title="AI-Based Expense Classification Tool – Prototype", layout="wide")
st.title("AI-Based Expense Classification Tool – Prototype")
st.caption("By Anushka Goyal")

# ---------- Load Environment ----------
env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MINILM_MODEL = os.getenv("MINILM_MODEL", "all-MiniLM-L6-v2")

if not GEMINI_API_KEY:
    st.warning("⚠️ No OpenRouter API key found. Please add it in `.env` as `GEMINI_API_KEY=your_key_here`.")

# ---------- Sidebar Settings ----------
st.sidebar.header("Settings")

allowed_categories = st.sidebar.multiselect(
    "Allowed Categories (choose ≥ 5)",
    ALLOWED_CATEGORIES,
    default=ALLOWED_CATEGORIES[:10],
)

low_confidence_label = st.sidebar.text_input("Low-confidence label", "Uncategorized")
rule_confidence = st.sidebar.slider("Rule confidence", 0.0, 1.0, 0.95, 0.05)
llm_threshold = st.sidebar.slider("LLM acceptance threshold", 0.0, 1.0, 0.6, 0.05)

st.sidebar.divider()
use_llm = st.sidebar.checkbox("Use  MiniLLM fallback", value=True)
st.sidebar.write(f"Model: `{MINILM_MODEL}`")


# ---------- Session State ----------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "df_out" not in st.session_state:
    st.session_state.df_out = None


# ---------- Data Input ----------
st.subheader("Upload Transactions")
uploaded_file = st.file_uploader("Upload CSV file of Trasactions", type=["csv"])
sample_btn = st.button("Use sample dataset")

if sample_btn:
    st.session_state.raw_df = pd.read_csv("data/Sample Transactions.csv")
elif uploaded_file is not None:
    st.session_state.raw_df = pd.read_csv(uploaded_file)

raw_df = st.session_state.raw_df
print(raw_df)
if raw_df is not None:
    st.success(f"Loaded {len(raw_df)} transactions")
    st.dataframe(raw_df, use_container_width=True)

    if len(raw_df) < 30:
        st.warning("Please upload at least 30 transactions for a full test.")

    # ---------- Cleaning ----------
    st.subheader("Data Cleaning and Preprocessing")
    with st.spinner("Cleaning data..."):
        df = clean_dataframe(raw_df)
    st.dataframe(df.head(20), use_container_width=True)

    # ---------- Categorization ----------
    st.subheader("Categorisation of Transactions")

    if st.button("Run Categorisation"):
        st.session_state.df_out = None  # clear previous run
        with st.spinner("Categorising... please wait ⏳"):
            try:
                categorizer = HybridCategorizer(
                    # gemini_api_key=GEMINI_API_KEY, 
                    # gemini_api_url="https://api.gemini.ai/v1/chat/completions",
                    # model_name=MINILM_MODEL,
                )
                df_out = categorizer.categorize_df(df)
                st.session_state.df_out = df_out
                st.success("Categorisation complete")
            except Exception as e:
                st.error(f"Error during categorisation: {e}")


    # ---------- Display Output ----------
    if st.session_state.df_out is not None:
        df_out = st.session_state.df_out
        st.dataframe(df_out, use_container_width=True)

        # ---------- Summary ----------
        st.subheader("Summary & Insights")

        # Ensure debit column exists
        if "Debit" not in df_out.columns:
            st.warning("Could not find a 'Debit' column for spend summary.")
        else:
            df_out["Debit"] = pd.to_numeric(df_out["Debit"], errors="coerce").fillna(0)

            # Only consider rows where Debit > 0
            debit_df = df_out[df_out["Debit"] > 0].copy()

            if debit_df.empty:
                st.info("No debit transactions found for spend summary.")
            else:
                debit_df["Category"] = debit_df["Category"].fillna("Uncategorized")

                # Group spend by Category
                spend_by_cat = (
                    debit_df.groupby("Category")["Debit"]
                    .sum()
                    .abs()
                    .sort_values(ascending=False)
                )

                total_spend = spend_by_cat.sum()
                highest_category = spend_by_cat.idxmax() if not spend_by_cat.empty else "-"
                top_categories = spend_by_cat.head(3)

                # Display summary metrics
                st.metric("Highest Spend Category", highest_category)
                st.metric("Total Debit Spend (₹)", f"{total_spend:,.2f}")

                # Show top 3 categories in table
                with st.expander("📊 Top 3 Spending Categories"):
                    st.dataframe(
                        top_categories.reset_index().rename(
                            columns={"Debit": "Total Spend (₹)"}
                        )
                    )

                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_category_bar(spend_by_cat))
                with col2:
                    st.pyplot(plot_category_pie(spend_by_cat))

        # ---------- Export ----------
        st.subheader("Export Results")
        csv_buf = io.StringIO()
        df_out.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Categorized CSV",
            csv_buf.getvalue(),
            file_name="categorized_transactions.csv",
            mime="text/csv",
        )

    else:
        st.info("Upload a CSV file or click 'Use sample dataset' to begin.")
