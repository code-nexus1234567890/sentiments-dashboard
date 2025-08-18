import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from passlib.hash import bcrypt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pymongo import MongoClient

# ---------- Optional BERT (only if you add transformers + torch to requirements) ----------
BERT_AVAILABLE = False
try:
    from transformers import pipeline  # type: ignore
    BERT_AVAILABLE = True
except Exception:
    BERT_AVAILABLE = False

# =================== App Config ===================
st.set_page_config(page_title="Sentiment Analytics (Login + Live)", page_icon="💬", layout="wide")

# Folders & files
DATA_DIR = "data"
LOG_PATH = os.path.join(DATA_DIR, "sentiment_log.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# =================== Database (MongoDB) ===================
MONGO_URI = "mongodb://localhost:27017"   # Change if Atlas or remote server
DB_NAME = "sentiment_app"
COLLECTION_NAME = "users"

def db_connect():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def init_db():
    db = db_connect()
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)

def user_exists(username: str) -> bool:
    db = db_connect()
    return db[COLLECTION_NAME].find_one({"username": username}) is not None

def create_user(username: str, password: str):
    pw_hash = bcrypt.hash(password)
    db = db_connect()
    db[COLLECTION_NAME].insert_one({
        "username": username,
        "password_hash": pw_hash,
        "created_at": datetime.utcnow().isoformat()
    })

def verify_user(username: str, password: str) -> bool:
    db = db_connect()
    user = db[COLLECTION_NAME].find_one({"username": username})
    if not user:
        return False
    return bcrypt.verify(password, user["password_hash"])

# Initialize DB on first run
init_db()

# =================== Sentiment Engines ===================
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_bert():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def infer_vader(text: str) -> Tuple[str, float]:
    analyzer = load_vader()
    sc = analyzer.polarity_scores(text)["compound"]
    if sc >= 0.05: lab = "POSITIVE"
    elif sc <= -0.05: lab = "NEGATIVE"
    else: lab = "NEUTRAL"
    return lab, float(sc)

def infer_bert(text: str) -> Tuple[str, float]:
    pipe = load_bert()
    out = pipe(text)[0]
    return out["label"].upper(), float(out["score"])

def analyze(text: str, engine: str) -> Tuple[str, float, str]:
    if engine == "Fast (VADER)":
        lab, sc = infer_vader(text)
        return lab, sc, "VADER"
    elif engine == "Accurate (BERT)" and BERT_AVAILABLE:
        lab, sc = infer_bert(text)
        return lab, sc, "BERT"
    else:
        lab, sc = infer_vader(text)
        return lab, sc, "VADER"

# =================== Logging ===================
def append_log(text: str, label: str, score: float, engine: str, username: Optional[str]):
    row = pd.DataFrame([[datetime.utcnow().isoformat(),
                         username or "",
                         text, label, score, engine]],
                       columns=["timestamp","user","text","label","score","engine"])
    if os.path.exists(LOG_PATH):
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)

def load_logs_df() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp","user","text","label","score","engine"])

# =================== Live Feed (Twitter via snscrape) ===================
def fetch_tweets(query: str, limit: int = 20) -> pd.DataFrame:
    try:
        import snscrape.modules.twitter as sntwitter  # type: ignore
    except Exception:
        st.error("snscrape not installed or failed to import. Ensure 'snscrape' is in requirements.txt.")
        return pd.DataFrame(columns=["text","date","user"])

    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        tweets.append({"text": tweet.content, "date": tweet.date, "user": str(tweet.user.username)})
    return pd.DataFrame(tweets)

# =================== Session State ===================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "session_results" not in st.session_state:
    st.session_state.session_results = []  # local (memory) results

# =================== UI Helpers ===================
def sentiment_color(label: str) -> str:
    return {"POSITIVE": "#27AE60", "NEGATIVE": "#C0392B"}.get(label, "#7F8C8D")

def result_card(label: str, score: float, engine: str):
    color = sentiment_color(label)
    st.markdown(
        f"<div style='padding:14px;border-radius:12px;background:{color};color:white;font-weight:600;'>"
        f"Sentiment: {label} &nbsp; | &nbsp; Score: {score:.3f} &nbsp; | &nbsp; Engine: {engine}"
        f"</div>", unsafe_allow_html=True
    )

# =================== Auth Screens ===================
def screen_login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_user(u, p):
            st.session_state.auth_user = u
            st.success(f"Welcome, {u}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

def screen_signup():
    st.subheader("📝 Create Account")
    u = st.text_input("Choose a username")
    p1 = st.text_input("Password", type="password")
    p2 = st.text_input("Confirm password", type="password")
    if st.button("Sign Up"):
        if not u or not p1:
            st.warning("Username and password required.")
        elif p1 != p2:
            st.warning("Passwords do not match.")
        elif user_exists(u):
            st.error("Username already taken.")
        else:
            create_user(u, p1)
            st.success("Account created. Please login from the sidebar.")

# =================== Main App (after login) ===================
def screen_dashboard():
    st.markdown("<h1 style='color:#2E86C1;margin-bottom:0'>Sentiment Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#5c6b7a'>Analyze text, batch CSVs, and live tweets. Logged in for personalized logs.</div>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Controls")
        opts = ["Fast (VADER)"]
        if BERT_AVAILABLE: opts.append("Accurate (BERT)")
        engine = st.radio("Engine", opts, index=0)
        do_log = st.checkbox("Save results to CSV", value=True)
        st.divider()
        st.write(f"👤 User: **{st.session_state.auth_user}**")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.experimental_rerun()

    tabs = st.tabs(["Single Text", "Batch CSV", "Live Feed (Twitter)", "Analytics & Logs"])

    # -------- Single Text --------
    with tabs[0]:
        colL, colR = st.columns([1.2, 1])
        with colL:
            st.subheader("✍️ Analyze a single text")
            txt = st.text_area("Enter text", height=140, placeholder="Type something people actually say…")
            if st.button("Analyze", key="an_single"):
                if txt.strip():
                    label, score, eng = analyze(txt, engine)
                    st.session_state.session_results.append(
                        {"timestamp": datetime.utcnow().isoformat(), "user": st.session_state.auth_user or "",
                         "text": txt, "label": label, "score": score, "engine": eng}
                    )
                    if do_log:
                        append_log(txt, label, score, eng, st.session_state.auth_user)
                    result_card(label, score, eng)
                else:
                    st.warning("Please enter some text.")
        with colR:
            st.subheader("Recent (this session)")
            if st.session_state.session_results:
                st.dataframe(pd.DataFrame(st.session_state.session_results)[["text","label","score","engine"]].tail(10), use_container_width=True)
            else:
                st.caption("No results yet.")

    # -------- Batch CSV --------
    with tabs[1]:
        st.subheader("📦 Upload CSV with a 'text' column")
        up = st.file_uploader("Choose file", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            if "text" not in df.columns:
                st.error("CSV must include a 'text' column.")
            else:
                results = []
                for t in df["text"].astype(str).tolist():
                    label, score, eng = analyze(t, engine)
                    results.append((label, score, eng))
                labels, scores, engines = zip(*results) if results else ([],[],[])
                out = pd.DataFrame({"text": df["text"], "label": labels, "score": scores, "engine": engines})
                st.dataframe(out, use_container_width=True)
                if do_log:
                    for _, r in out.iterrows():
                        append_log(str(r["text"]), r["label"], float(r["score"]), str(r["engine"]), st.session_state.auth_user)
                st.download_button("⬇️ Download results", out.to_csv(index=False), "results.csv", "text/csv")

    # -------- Live Feed (Twitter via snscrape) --------
    with tabs[2]:
        st.subheader("🛰️ Live Feed (Twitter, no API key)")
        q = st.text_input("Search query (e.g., iphone OR pixel -giveaway)", value="(happy OR sad) lang:en")
        n = st.slider("Number of tweets", min_value=5, max_value=100, value=20, step=5)
        if st.button("Fetch tweets", key="fetch"):
            tweets_df = fetch_tweets(q, n)
            if tweets_df.empty:
                st.warning("No tweets fetched (or snscrape blocked). Try different query or fewer tweets.")
            else:
                st.dataframe(tweets_df, use_container_width=True)
                analyzed_rows = []
                for _, row in tweets_df.iterrows():
                    label, score, eng = analyze(str(row["text"]), engine)
                    analyzed_rows.append({"text": row["text"], "date": row["date"], "user": row["user"],
                                          "label": label, "score": score, "engine": eng})
                    if do_log:
                        append_log(str(row["text"]), label, score, eng, st.session_state.auth_user)
                if analyzed_rows:
                    st.markdown("**Analyzed Tweets**")
                    st.dataframe(pd.DataFrame(analyzed_rows), use_container_width=True)

    # -------- Analytics & Logs --------
    with tabs[3]:
        st.subheader("📊 Analytics")
        logs_file = load_logs_df()
        session_df = pd.DataFrame(st.session_state.session_results)
        logs = pd.concat([logs_file, session_df], ignore_index=True) if not logs_file.empty else session_df.copy()

        if logs.empty:
            st.caption("No data yet. Run analyses first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.write("Distribution")
                counts = logs["label"].value_counts().reindex(["POSITIVE","NEGATIVE","NEUTRAL"]).fillna(0)
                fig1, ax1 = plt.subplots()
                counts.plot(kind="pie", autopct="%1.1f%%", ax=ax1, ylabel="")
                st.pyplot(fig1, use_container_width=True)

            with c2:
                st.write("Average score over time")
                tmp = logs.copy()
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
                tmp = tmp.dropna(subset=["timestamp"])
                if tmp.empty:
                    st.caption("No timestamps yet.")
                else:
                    ts = tmp.groupby(tmp["timestamp"].dt.floor("min"))["score"].mean()
                    fig2, ax2 = plt.subplots()
                    ts.plot(ax=ax2)
                    ax2.set_xlabel("time (minute)")
                    ax2.set_ylabel("avg score")
                    st.pyplot(fig2, use_container_width=True)

            st.divider()
            st.subheader("📥 Download full log")
            if not logs.empty:
                st.download_button("Download log CSV", logs.to_csv(index=False), "sentiment_log.csv", "text/csv")

# =================== Router ===================
with st.sidebar:
    st.header("Navigation")
    if st.session_state.auth_user:
        st.success(f"Logged in: {st.session_state.auth_user}")
        page = st.radio("Go to", ["Dashboard", "Logout"])
    else:
        page = st.radio("Go to", ["Login", "Sign Up", "About"])

if not st.session_state.auth_user:
    if page == "Login":
        screen_login()
    elif page == "Sign Up":
        screen_signup()
    else:
        st.title("Cloud-Based Real-Time Sentiment Analysis")
        st.write("Create an account or log in from the sidebar to use the dashboard.")
else:
    if page == "Logout":
        st.session_state.auth_user = None
        st.experimental_rerun()
    else:
        screen_dashboard()
