
import streamlit as st
import joblib
import pandas as pd
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Spam Detector | AI Dashboard",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "spam_history.db"
METRICS_PATH = BASE_DIR / "model_metrics.json"

# ==================================================
# 2. DATABASE FUNCTIONS
# ==================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL
            )
        """)


def save_prediction(message, result, confidence):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions
            (message, result, confidence, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            message,
            result,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


def get_history():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )


def clear_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")


initialize_database()

# ==================================================
# 3. LOAD MACHINE LEARNING MODEL
# ==================================================

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "model.pkl")
    vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.info(
        "Make sure model.pkl and vectorizer.pkl "
        "are present in your project folder."
    )
    st.stop()

# ==================================================
# 4. LOAD MODEL METRICS
# ==================================================

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    metrics = load_metrics()
except FileNotFoundError:
    metrics = None
except (json.JSONDecodeError, KeyError, TypeError) as error:
    metrics = None
    st.warning(f"Metrics file could not be read: {error}")

# ==================================================
# 5. CUSTOM DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FC;
    color: #172033;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3, h4, p, label {
    color: #172033;
}

.main-title {
    text-align: center;
    color: #2563EB;
    font-size: 38px;
    font-weight: 800;
}

.sub-title {
    text-align: center;
    color: #64748B;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

div[data-testid="stMetricValue"] {
    color: #2563EB;
}

.stTextArea textarea {
    background: white !important;
    color: #172033 !important;
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
}

.stButton button {
    background: #2563EB !important;
    color: white !important;
    border-radius: 9px;
    font-weight: 600;
    border: none;
}

.stButton button:hover {
    background: #1D4ED8 !important;
}

[data-testid="stSidebar"] {
    background: #172033;
}

[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 6. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.title("📩 Spam AI")
    st.caption("Machine Learning Dashboard")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Detect Message",
            "📚 History",
            "📈 Model Evaluation",
            "ℹ️ About Project"
        ]
    )

    st.divider()
    st.caption("Powered by Python & Streamlit")

# ==================================================
# 7. HEADER AND STATISTICS
# ==================================================

st.markdown(
    '<div class="main-title">📩 Spam Message Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Advanced Machine Learning Message Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

history = get_history()
total = len(history)

spam_count = (
    int((history["result"] == "SPAM").sum())
    if total else 0
)

normal_count = total - spam_count

spam_percentage = (
    spam_count / total * 100
    if total else 0
)

# ==================================================
# 8. DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.subheader("📊 Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Messages", total)
    c2.metric("🚨 Spam Messages", spam_count)
    c3.metric("✅ Normal Messages", normal_count)
    c4.metric("Spam Percentage", f"{spam_percentage:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📈 Spam vs Normal")

        if total:
            chart_data = pd.DataFrame({
                "Category": ["Spam", "Normal"],
                "Messages": [spam_count, normal_count]
            }).set_index("Category")

            st.bar_chart(chart_data)
        else:
            st.info("Analytics will appear after your first prediction.")

    with right:
        st.subheader("📊 Message Distribution")

        if total:
            distribution = pd.DataFrame({
                "Message Type": ["Spam", "Normal"],
                "Count": [spam_count, normal_count]
            }).set_index("Message Type")

            st.bar_chart(distribution)
        else:
            st.info("No prediction data available yet.")

    st.divider()
    st.subheader("🤖 Machine Learning Performance")

    if metrics:
        best_name = metrics["best_model"]
        best_scores = metrics["models"][best_name]

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Best Model", best_name)
        m2.metric("Accuracy", f'{best_scores["accuracy"]:.2f}%')
        m3.metric("Precision", f'{best_scores["precision"]:.2f}%')
        m4.metric("F1 Score", f'{best_scores["f1_score"]:.2f}%')

        st.caption(
            "These scores are based on the held-out test dataset, "
            "not your live prediction history."
        )
    else:
        st.warning("Model metrics file is unavailable.")

    st.divider()
    st.subheader("🕒 Recent Predictions")

    if total:
        recent = history.head(5).copy()

        recent["confidence"] = recent["confidence"].apply(
            lambda value: f"{value:.2f}%"
            if pd.notna(value)
            else "Unavailable"
        )

        st.dataframe(
            recent[
                ["created_at", "message", "result", "confidence"]
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No predictions yet. Start by checking a message.")

# ==================================================
# 9. MESSAGE DETECTION
# ==================================================

elif page == "🔍 Detect Message":

    st.subheader("🔍 Check Your Message")

    with st.form("prediction_form"):
        message = st.text_area(
            "Enter your message:",
            placeholder="Example: Congratulations! You won a free prize...",
            height=160
        )

        submitted = st.form_submit_button(
            "🚀 Analyze Message",
            use_container_width=True
        )

    if submitted:

        if not message.strip():
            st.warning("Please enter a message first.")

        else:
            try:
                vector = vectorizer.transform([message])
                prediction = model.predict(vector)[0]

                # Training labels: 1 = SPAM, 0 = NORMAL
                is_spam = int(prediction) == 1
                label = "SPAM" if is_spam else "NORMAL"

                confidence = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(vector)[0]
                    classes = list(model.classes_)
                    predicted_index = classes.index(prediction)

                    confidence = float(
                        probabilities[predicted_index] * 100
                    )

                save_prediction(message, label, confidence)

                st.divider()
                st.subheader("🎯 Detection Result")

                if is_spam:
                    st.error("🚨 SPAM MESSAGE DETECTED!")
                    st.write(
                        "This message has been classified as spam."
                    )
                else:
                    st.success("✅ NORMAL MESSAGE")
                    st.write(
                        "This message has been classified as legitimate."
                    )

                if confidence is not None:
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.progress(
                        min(100, max(0, int(confidence)))
                    )
                else:
                    st.info(
                        "Probability scores are not supported by this model."
                    )

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not a guarantee of correctness."
                )

            except Exception as error:
                st.error(f"Prediction error: {error}")

# ==================================================
# 10. HISTORY AND CSV EXPORT
# ==================================================

elif page == "📚 History":

    st.subheader("📚 Complete Detection History")

    if total:
        search = st.text_input(
            "🔎 Search messages",
            placeholder="Type a word to search..."
        )

        filtered = history.copy()

        if search.strip():
            filtered = filtered[
                filtered["message"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.write(f"Showing {len(filtered)} records")

        display_df = filtered.copy()

        display_df["confidence"] = display_df["confidence"].apply(
            lambda value: f"{value:.2f}%"
            if pd.notna(value)
            else "Unavailable"
        )

        st.dataframe(
            display_df[
                ["id", "created_at", "message", "result", "confidence"]
            ],
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="spam_detection_history.csv",
            mime="text/csv"
        )

        st.divider()

        if st.button("🗑️ Delete All History"):
            clear_history()
            st.success("History cleared successfully!")
            st.rerun()

    else:
        st.info("No history available yet.")

# ==================================================
# 11. MODEL EVALUATION
# ==================================================

elif page == "📈 Model Evaluation":

    st.subheader("📈 Model Evaluation & Comparison")

    if metrics:

        st.write(
            "Performance comparison of machine learning models "
            "on the held-out test dataset."
        )

        best_name = metrics["best_model"]

        st.success(f"🏆 Best Performing Model: {best_name}")

        # Dataset information

        st.write("### 📁 Dataset Information")

        d1, d2, d3 = st.columns(3)

        d1.metric("Total Messages", metrics["dataset_size"])
        d2.metric("Training Samples", metrics["training_samples"])
        d3.metric("Testing Samples", metrics["testing_samples"])

        st.divider()

        # Model comparison

        st.write("### 📊 Model Performance Comparison")

        comparison_rows = []

        for name, scores in metrics["models"].items():
            comparison_rows.append({
                "Model": name,
                "Accuracy (%)": scores["accuracy"],
                "Precision (%)": scores["precision"],
                "Recall (%)": scores["recall"],
                "F1 Score (%)": scores["f1_score"]
            })

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.write("### Performance Metrics Chart")

        chart_df = comparison_df.set_index("Model")

        st.bar_chart(chart_df)

        st.divider()

        # Best model scores

        best_scores = metrics["models"][best_name]

        st.write(f"### 🏆 Best Model: {best_name}")

        a, b, c, d = st.columns(4)

        a.metric("Accuracy", f'{best_scores["accuracy"]:.2f}%')
        b.metric("Precision", f'{best_scores["precision"]:.2f}%')
        c.metric("Recall", f'{best_scores["recall"]:.2f}%')
        d.metric("F1 Score", f'{best_scores["f1_score"]:.2f}%')

        st.caption(
            "Metrics are calculated from the test dataset. "
            "They are not confidence scores of individual messages."
        )

        st.divider()

        # Confusion matrix

        st.write("### Confusion Matrix")

        model_names = list(metrics["models"].keys())

        selected_model = st.selectbox(
            "Choose Model",
            model_names,
            index=model_names.index(best_name)
        )

        selected_scores = metrics["models"][selected_model]
        cm = selected_scores["confusion_matrix"]

        confusion_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Spam"],
            columns=["Predicted Normal", "Predicted Spam"]
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

        st.caption(
            "Rows represent actual labels; "
            "columns represent predicted labels."
        )

        st.divider()

        # Classification report

        st.write("### Classification Report")

        report = selected_scores["classification_report"]
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )

    else:
        st.error(
            "model_metrics.json was not found or could not be loaded. "
            "Run python train.py to generate it."
        )

# ==================================================
# 12. ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### 📩 Spam Message Detector

    This application uses Machine Learning to classify
    text messages into Spam and Normal categories.

    **Technologies Used**

    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - Joblib
    - SQLite
    - TF-IDF Vectorization

    **Machine Learning Models Compared**

    - Multinomial Naive Bayes
    - Logistic Regression

    **How It Works**

    1. User enters a message.
    2. TF-IDF vectorizer converts text into numerical features.
    3. Trained model predicts its category.
    4. The prediction is displayed to the user.
    5. The result is saved in an SQLite database.
    6. Model performance is evaluated using a separate test dataset.
    """)

    if metrics:
        best_name = metrics["best_model"]
        best_f1 = metrics["models"][best_name]["f1_score"]

        st.success(
            f"Best model: {best_name} | F1 Score: {best_f1:.2f}%"
        )

    st.info(
        "This is a Machine Learning classification project. "
        "It does not require an external AI API."
    )

# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.markdown(
    "<p style='text-align:center;color:#64748B;'>"
    "Spam Message Detector | Python | Streamlit | "
    "Scikit-learn | SQLite"
    "</p>",
    unsafe_allow_html=True
)
