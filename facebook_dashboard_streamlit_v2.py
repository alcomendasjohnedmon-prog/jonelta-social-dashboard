import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None
try:
    from transformers import pipeline
except Exception:
    pipeline = None

st.set_page_config(
    page_title="Facebook Performance Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

GRAPH_API_BASE = "https://graph.facebook.com/v25.0"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_POST_LIMIT = 100
DEFAULT_COMMENT_LIMIT_PER_POST = 150

PAGE_ACCOUNT_FIELDS = "id,name,category,category_list,tasks,access_token"
PAGE_POST_FIELDS = (
    "id,message,created_time,permalink_url,status_type,full_picture,shares,"
    "reactions.summary(true),comments.summary(true),"
    "like_reactions:reactions.type(LIKE).summary(total_count).limit(0),"
    "love_reactions:reactions.type(LOVE).summary(total_count).limit(0),"
    "care_reactions:reactions.type(CARE).summary(total_count).limit(0),"
    "haha_reactions:reactions.type(HAHA).summary(total_count).limit(0),"
    "wow_reactions:reactions.type(WOW).summary(total_count).limit(0),"
    "sad_reactions:reactions.type(SAD).summary(total_count).limit(0),"
    "angry_reactions:reactions.type(ANGRY).summary(total_count).limit(0)"
)
COMMENT_FIELDS = (
    "id,message,created_time,from,comment_count,permalink_url,reactions.summary(true),"
    "like_reactions:reactions.type(LIKE).summary(total_count).limit(0),"
    "love_reactions:reactions.type(LOVE).summary(total_count).limit(0),"
    "care_reactions:reactions.type(CARE).summary(total_count).limit(0),"
    "haha_reactions:reactions.type(HAHA).summary(total_count).limit(0),"
    "wow_reactions:reactions.type(WOW).summary(total_count).limit(0),"
    "sad_reactions:reactions.type(SAD).summary(total_count).limit(0),"
    "angry_reactions:reactions.type(ANGRY).summary(total_count).limit(0)"
)
REACTION_COLUMNS = ["like_reactions", "love_reactions", "care_reactions", "haha_reactions", "wow_reactions", "sad_reactions", "angry_reactions"]

POSITIVE_TERMS = ["good", "great", "excellent", "amazing", "helpful", "thanks", "thank you", "love", "best", "recommend", "satisfied", "happy", "nice", "awesome", "legit", "salamat", "maganda", "mabait", "maayos", "sulit", "solid", "recommended"]
NEGATIVE_TERMS = ["bad", "worst", "poor", "terrible", "awful", "hate", "angry", "scam", "fake", "fraud", "delay", "late", "cancel", "refund", "complaint", "unresponsive", "disappointed", "pangit", "galit", "bwisit", "buwisit", "inis", "nakakainis", "bulok", "sayang", "hassle", "walang reply", "hindi sumasagot", "di sumasagot", "niloko", "nanloko"]
RISK_PATTERNS = {
    "legal/report threat": ["lawsuit", "sue", "legal action", "lawyer", "attorney", "report to", "complain to", "dti", "deped", "ched", "prc", "police", "barangay", "tulfo"],
    "refund/payment issue": ["refund", "chargeback", "scam", "nanloko", "niloko", "bayad", "payment", "paid already", "binayaran", "money back", "pera ko", "overcharged"],
    "service failure": ["no response", "unresponsive", "hindi sumasagot", "walang reply", "late", "delay", "cancel", "cancelled", "hindi dumating", "poor service", "bad service", "worst"],
    "anger/abuse": ["angry", "furious", "galit", "bwisit", "buwisit", "inis", "nakakainis", "terrible", "trash", "garbage", "stupid", "idiot", "hate", "pangit", "bulok"],
    "lead/sales opportunity": ["how much", "hm", "price", "pricing", "interested", "available", "avail", "pm", "dm", "inquire", "details", "location", "schedule", "enroll", "apply", "book"],
    "question needs reply": ["?", "how", "what", "when", "where", "why", "can i", "do you", "may i", "paano", "ano", "saan", "kailan", "pwede", "puwede", "magkano"],
}


def secret_value(name: str, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


def configured_pages() -> Dict[str, str]:
    pages = {}
    try:
        raw_pages = st.secrets.get("pages", {})
        for page_id, page_name in raw_pages.items():
            pages[str(page_id)] = str(page_name)
    except Exception:
        pass
    env_pages = os.getenv("FB_PAGE_IDS", "").strip()
    if env_pages:
        for item in env_pages.split(","):
            page_id = item.strip()
            if page_id:
                pages.setdefault(page_id, page_id)
    return pages


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def vader_model():
    if SentimentIntensityAnalyzer is None:
        return None
    return SentimentIntensityAnalyzer()


@st.cache_resource(show_spinner=False)
def transformer_model():
    if pipeline is None:
        return None
    try:
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment", top_k=None)
    except Exception:
        return None


def rules_score(text: str) -> float:
    lower_text = text.lower()
    positive_hits = sum(1 for word in POSITIVE_TERMS if word in lower_text)
    negative_hits = sum(1 for word in NEGATIVE_TERMS if word in lower_text)
    if positive_hits == 0 and negative_hits == 0:
        return 0.0
    return (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)


def transformer_sentiment(text: str) -> Tuple[Optional[str], Optional[float]]:
    model = transformer_model()
    if model is None or not text:
        return None, None
    try:
        result = model(text[:512])
        rows = result[0]
        if rows and isinstance(rows[0], list):
            rows = rows[0]
        scores = {str(row["label"]).lower(): float(row["score"]) for row in rows}
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        mapped = best_label.replace("label_0", "negative").replace("label_1", "neutral").replace("label_2", "positive")
        if "negative" in mapped:
            return "Negative", -best_score
        if "positive" in mapped:
            return "Positive", best_score
        return "Neutral", 0.0
    except Exception:
        return None, None


def hybrid_sentiment(text: Optional[str], use_transformer: bool) -> Dict[str, object]:
    cleaned = clean_text(text)
    if not cleaned:
        return {"sentiment": "Neutral", "sentiment_score": 0.0, "sentiment_engine": "empty"}
    if use_transformer:
        label, score = transformer_sentiment(cleaned)
        if label is not None:
            return {"sentiment": label, "sentiment_score": float(score), "sentiment_engine": "xlm-roberta"}
    scores = [rules_score(cleaned)]
    engines = ["rules"]
    analyzer = vader_model()
    if analyzer is not None:
        scores.append(float(analyzer.polarity_scores(cleaned)["compound"]))
        engines.append("vader")
    final_score = float(np.mean(scores))
    if final_score <= -0.25:
        label = "Negative"
    elif final_score >= 0.25:
        label = "Positive"
    else:
        label = "Neutral"
    return {"sentiment": label, "sentiment_score": final_score, "sentiment_engine": "+".join(engines)}


def classify_item(text: Optional[str], sentiment: str, sentiment_score: float, angry_count: int, sad_count: int) -> Dict[str, object]:
    lower_text = clean_text(text).lower()
    categories = []
    terms = []
    for category, needles in RISK_PATTERNS.items():
        hits = [needle for needle in needles if needle in lower_text]
        if hits:
            categories.append(category)
            terms.extend(hits[:3])
    severity = 0
    if sentiment == "Negative":
        severity += 25
    if sentiment_score <= -0.55:
        severity += 25
    if "legal/report threat" in categories:
        severity += 35
    if "refund/payment issue" in categories:
        severity += 25
    if "service failure" in categories:
        severity += 20
    if "anger/abuse" in categories:
        severity += 20
    if angry_count > 0:
        severity += min(20, angry_count * 4)
    if sad_count > 2:
        severity += 10
    if "question needs reply" in categories:
        severity += 10
    severity = min(int(severity), 100)
    if severity >= 70:
        priority = "Critical"
    elif severity >= 45:
        priority = "High"
    elif severity >= 20:
        priority = "Medium"
    else:
        priority = "Low"
    if "lead/sales opportunity" in categories and priority == "Low":
        priority = "Medium"
    needs_response = priority in ["Critical", "High"] or "question needs reply" in categories or "lead/sales opportunity" in categories
    return {"priority": priority, "severity_score": severity, "needs_response": needs_response, "risk_categories": ", ".join(categories) if categories else "none", "matched_terms": ", ".join(sorted(set(terms))[:8])}


def request_json(url: str, params: Optional[dict] = None) -> dict:
    response = requests.get(url, params=params, timeout=45)
    try:
        payload = response.json()
    except Exception:
        payload = {"error": {"message": response.text}}
    if response.status_code >= 400 and "error" not in payload:
        payload["error"] = {"message": "HTTP " + str(response.status_code)}
    return payload


def paginate(url: str, params: Optional[dict], max_items: int, sleep_seconds: float = 0.05) -> List[dict]:
    rows = []
    next_url = url
    next_params = params.copy() if params else None
    while next_url and len(rows) < max_items:
        payload = request_json(next_url, next_params)
        if "error" in payload:
            st.warning(payload["error"].get("message", "Graph API error"))
            break
        rows.extend(payload.get("data", []))
        paging = payload.get("paging", {})
        next_url = paging.get("next")
        next_params = None
        time.sleep(sleep_seconds)
    return rows[:max_items]


def reaction_count(item: dict, reaction_name: str) -> int:
    return int(item.get(reaction_name, {}).get("summary", {}).get("total_count", 0) or 0)


def normalize_post(post: dict, page_id: str, page_name: str, use_transformer: bool) -> dict:
    text = post.get("message", "")
    sentiment = hybrid_sentiment(text, use_transformer)
    reaction_values = {col: reaction_count(post, col) for col in REACTION_COLUMNS}
    risk = classify_item(text, sentiment["sentiment"], sentiment["sentiment_score"], reaction_values.get("angry_reactions", 0), reaction_values.get("sad_reactions", 0))
    row = {
        "item_type": "post", "page_id": page_id, "page_name": page_name, "post_id": post.get("id"), "item_id": post.get("id"),
        "parent_id": "", "author_name": page_name, "message": text, "created_time": post.get("created_time"),
        "permalink_url": post.get("permalink_url", ""), "status_type": post.get("status_type", ""),
        "comments_total": int(post.get("comments", {}).get("summary", {}).get("total_count", 0) or 0),
        "reactions_total": int(post.get("reactions", {}).get("summary", {}).get("total_count", 0) or 0),
        "shares_total": int(post.get("shares", {}).get("count", 0) or 0),
    }
    row.update(reaction_values)
    row.update(sentiment)
    row.update(risk)
    return row


def normalize_comment(comment: dict, post_id: str, page_id: str, page_name: str, use_transformer: bool) -> dict:
    text = comment.get("message", "")
    sentiment = hybrid_sentiment(text, use_transformer)
    reaction_values = {col: reaction_count(comment, col) for col in REACTION_COLUMNS}
    risk = classify_item(text, sentiment["sentiment"], sentiment["sentiment_score"], reaction_values.get("angry_reactions", 0), reaction_values.get("sad_reactions", 0))
    author = comment.get("from", {}).get("name", "") if isinstance(comment.get("from"), dict) else ""
    row = {
        "item_type": "comment", "page_id": page_id, "page_name": page_name, "post_id": post_id, "item_id": comment.get("id"),
        "parent_id": post_id, "author_name": author, "message": text, "created_time": comment.get("created_time"),
        "permalink_url": comment.get("permalink_url", ""), "status_type": "comment", "comments_total": int(comment.get("comment_count", 0) or 0),
        "reactions_total": int(comment.get("reactions", {}).get("summary", {}).get("total_count", 0) or 0), "shares_total": 0,
    }
    row.update(reaction_values)
    row.update(sentiment)
    row.update(risk)
    return row


def check_token_health(user_token: str):
    app_id = secret_value("APP_ID")
    app_secret = secret_value("APP_SECRET")
    if not app_id or not app_secret or not user_token:
        return None, None, None
    payload = request_json("https://graph.facebook.com/debug_token", {"input_token": user_token, "access_token": str(app_id) + "|" + str(app_secret)})
    if "error" in payload:
        return False, None, payload["error"].get("message")
    data = payload.get("data", {})
    is_valid = bool(data.get("is_valid"))
    expires_at = data.get("expires_at")
    expiry_date = datetime.fromtimestamp(expires_at) if expires_at else None
    return is_valid, expiry_date, None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_page_accounts(user_token: str) -> List[dict]:
    rows = paginate(GRAPH_API_BASE + "/me/accounts", {"fields": PAGE_ACCOUNT_FIELDS, "access_token": user_token, "limit": 100}, 500)
    return [{"page_id": row.get("id"), "page_name": row.get("name"), "category": row.get("category", ""), "tasks": row.get("tasks", []), "page_access_token": row.get("access_token")} for row in rows]


@st.cache_data(ttl=600, show_spinner=True)
def fetch_dashboard_data(user_token: str, selected_page_ids: List[str], since_iso: str, until_iso: str, post_limit: int, comment_limit: int, include_comments: bool, use_transformer: bool) -> pd.DataFrame:
    account_rows = fetch_page_accounts(user_token)
    account_lookup = {str(row["page_id"]): row for row in account_rows if row.get("page_id")}
    configured = configured_pages()
    if not selected_page_ids:
        selected_page_ids = list(account_lookup.keys())[:7]
    all_rows = []
    progress = st.progress(0, text="Fetching Facebook data...")
    total_pages = max(len(selected_page_ids), 1)
    for page_index, page_id in enumerate(selected_page_ids):
        account = account_lookup.get(str(page_id), {})
        page_name = account.get("page_name") or configured.get(str(page_id), str(page_id))
        page_token = account.get("page_access_token") or user_token
        post_url = GRAPH_API_BASE + "/" + str(page_id) + "/posts"
        posts = paginate(post_url, {"fields": PAGE_POST_FIELDS, "access_token": page_token, "since": since_iso, "until": until_iso, "limit": 50}, post_limit)
        for post in posts:
            post_id = post.get("id")
            all_rows.append(normalize_post(post, str(page_id), page_name, use_transformer))
            if include_comments and post_id:
                comment_url = GRAPH_API_BASE + "/" + post_id + "/comments"
                comments = paginate(comment_url, {"fields": COMMENT_FIELDS, "access_token": page_token, "filter": "stream", "limit": 100}, comment_limit)
                for comment in comments:
                    all_rows.append(normalize_comment(comment, post_id, str(page_id), page_name, use_transformer))
        progress.progress((page_index + 1) / total_pages, text="Fetched " + page_name)
    progress.empty()
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["created_time"])
        df["date"] = df["created_time"].dt.date
        df["hour"] = df["created_time"].dt.hour
        df["weekday"] = df["created_time"].dt.day_name()
        df["engagement_total"] = df["reactions_total"] + df["comments_total"] + df["shares_total"]
        df["message_short"] = df["message"].fillna("").str.slice(0, 160)
    return df


def style_priority(value):
    colors = {"Critical": "background-color:#fee2e2;color:#991b1b;font-weight:700", "High": "background-color:#ffedd5;color:#9a3412;font-weight:700", "Medium": "background-color:#fef9c3;color:#854d0e", "Low": "background-color:#dcfce7;color:#166534"}
    return colors.get(value, "")


st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
[data-testid="stMetricValue"] {font-size: 2rem;}
.alert-card {border:1px solid #fecaca; background:#fff1f2; color:#7f1d1d; padding:1rem; border-radius:14px; margin-bottom:.75rem;}
.good-card {border:1px solid #bbf7d0; background:#f0fdf4; color:#14532d; padding:1rem; border-radius:14px; margin-bottom:.75rem;}
.small-muted {color:#64748b; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("Facebook Performance Command Center")
st.caption("Dynamic dashboard for multi-page engagement, sentiment analytics, and critical interaction flagging.")

user_token = secret_value("FB_TOKEN")
if not user_token:
    st.error("Missing FB_TOKEN. Add it to .streamlit/secrets.toml or environment variables.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    is_valid, expiry_date, token_error = check_token_health(user_token)
    if is_valid is True:
        if expiry_date:
            days_left = (expiry_date - datetime.now()).days
            st.success("Token valid - " + str(days_left) + " days left")
        else:
            st.success("Token valid")
    elif is_valid is False:
        st.error("Token invalid")
        if token_error:
            st.caption(token_error)
    else:
        st.warning("Token health unavailable")

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    accounts = fetch_page_accounts(user_token)
    account_options = {row["page_name"] + " - " + row["page_id"]: row["page_id"] for row in accounts if row.get("page_id")}
    configured = configured_pages()
    for page_id, page_name in configured.items():
        account_options.setdefault(page_name + " - " + page_id, page_id)
    default_labels = list(account_options.keys())[:7]
    selected_labels = st.multiselect("Pages", list(account_options.keys()), default=default_labels)
    selected_page_ids = [account_options[label] for label in selected_labels]
    lookback_days = st.slider("Lookback days", 1, 180, DEFAULT_LOOKBACK_DAYS)
    post_limit = st.slider("Max posts per page", 10, 500, DEFAULT_POST_LIMIT, step=10)
    include_comments = st.toggle("Fetch comments", value=True)
    comment_limit = st.slider("Max comments per post", 25, 500, DEFAULT_COMMENT_LIMIT_PER_POST, step=25, disabled=not include_comments)
    use_transformer = st.toggle("Use transformer sentiment", value=False, help="More accurate for social media and multilingual text, but slower and requires transformers + torch.")

until_dt = datetime.now(timezone.utc)
since_dt = until_dt - timedelta(days=lookback_days)
df = fetch_dashboard_data(user_token, selected_page_ids, since_dt.isoformat(), until_dt.isoformat(), post_limit, comment_limit, include_comments, use_transformer)

if df.empty:
    st.warning("No data returned. Check page permissions, selected pages, or date range.")
    st.stop()

post_df = df[df["item_type"] == "post"].copy()
comment_df = df[df["item_type"] == "comment"].copy()
flagged_df = df[df["priority"].isin(["Critical", "High"])].copy().sort_values(["severity_score", "created_time"], ascending=[False, False])
needs_reply_df = df[df["needs_response"] == True].copy().sort_values("created_time", ascending=False)

kpi_cols = st.columns(6)
kpi_cols[0].metric("Pages", df["page_id"].nunique())
kpi_cols[1].metric("Posts", len(post_df))
kpi_cols[2].metric("Comments", len(comment_df))
kpi_cols[3].metric("Engagement", int(post_df["engagement_total"].sum()))
kpi_cols[4].metric("Critical/High", len(flagged_df))
kpi_cols[5].metric("Needs reply", len(needs_reply_df))

if len(flagged_df) > 0:
    top_alert = flagged_df.iloc[0]
    st.markdown("<div class='alert-card'><b>Top active flag</b><br>" + str(top_alert["priority"]) + " - severity " + str(int(top_alert["severity_score"])) + " - " + str(top_alert["page_name"]) + "<br><span class='small-muted'>" + str(top_alert["message_short"]) + "</span></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='good-card'><b>No Critical or High priority items detected.</b><br><span class='small-muted'>Still review Medium items for sales leads and unanswered questions.</span></div>", unsafe_allow_html=True)

tab_overview, tab_sentiment, tab_posts, tab_interactions, tab_explorer, tab_export = st.tabs(["Overview", "Sentiment + Risk", "Posts", "Interactions", "Explorer", "Export"])

with tab_overview:
    left, right = st.columns([1.3, 1])
    daily = df.groupby(["date", "item_type"], as_index=False).size().rename(columns={"size": "count"})
    with left:
        st.subheader("Activity trend")
        st.plotly_chart(px.area(daily, x="date", y="count", color="item_type", color_discrete_map={"post": "#1877F2", "comment": "#10B981"}), use_container_width=True)
    page_summary = post_df.groupby("page_name", as_index=False).agg(posts=("item_id", "count"), reactions=("reactions_total", "sum"), comments=("comments_total", "sum"), shares=("shares_total", "sum"), engagement=("engagement_total", "sum"), avg_severity=("severity_score", "mean"))
    with right:
        st.subheader("Page leaderboard")
        st.plotly_chart(px.bar(page_summary.sort_values("engagement", ascending=True), x="engagement", y="page_name", orientation="h", color="avg_severity", color_continuous_scale="OrRd"), use_container_width=True)
    reaction_mix = post_df.groupby("page_name", as_index=False)[REACTION_COLUMNS].sum()
    reaction_long = reaction_mix.melt(id_vars="page_name", value_vars=REACTION_COLUMNS, var_name="reaction", value_name="count")
    st.subheader("Reaction mix")
    st.plotly_chart(px.bar(reaction_long, x="page_name", y="count", color="reaction", barmode="stack"), use_container_width=True)

with tab_sentiment:
    left, right = st.columns(2)
    sentiment_counts = df.groupby(["page_name", "sentiment"], as_index=False).size().rename(columns={"size": "count"})
    with left:
        st.subheader("Sentiment by page")
        st.plotly_chart(px.bar(sentiment_counts, x="page_name", y="count", color="sentiment", barmode="stack", color_discrete_map={"Positive": "#16a34a", "Neutral": "#64748b", "Negative": "#dc2626"}), use_container_width=True)
    priority_counts = df.groupby(["priority", "item_type"], as_index=False).size().rename(columns={"size": "count"})
    with right:
        st.subheader("Priority distribution")
        st.plotly_chart(px.bar(priority_counts, x="priority", y="count", color="item_type", category_orders={"priority": ["Critical", "High", "Medium", "Low"]}), use_container_width=True)
    heat = df.pivot_table(index="weekday", columns="hour", values="item_id", aggfunc="count", fill_value=0)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = heat.reindex([day for day in weekday_order if day in heat.index])
    st.subheader("Activity heatmap")
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues"), use_container_width=True)

with tab_posts:
    st.subheader("Top posts")
    top_posts = post_df.sort_values("engagement_total", ascending=False).head(25)
    cols = ["page_name", "created_time", "engagement_total", "reactions_total", "comments_total", "shares_total", "sentiment", "priority", "message_short", "permalink_url"]
    st.dataframe(top_posts[cols].style.applymap(style_priority, subset=["priority"]), use_container_width=True, hide_index=True)
    st.plotly_chart(px.scatter(post_df, x="created_time", y="engagement_total", color="sentiment", size="comments_total", hover_data=["page_name", "message_short"], color_discrete_map={"Positive": "#16a34a", "Neutral": "#64748b", "Negative": "#dc2626"}), use_container_width=True)

with tab_interactions:
    st.subheader("Critical and high-priority queue")
    display_cols = ["priority", "severity_score", "needs_response", "page_name", "item_type", "created_time", "risk_categories", "matched_terms", "author_name", "message_short", "permalink_url"]
    if flagged_df.empty:
        st.success("No Critical or High items in the current window.")
    else:
        st.dataframe(flagged_df[display_cols].style.applymap(style_priority, subset=["priority"]), use_container_width=True, hide_index=True)
    st.subheader("Needs-response queue")
    st.dataframe(needs_reply_df[display_cols].head(200).style.applymap(style_priority, subset=["priority"]), use_container_width=True, hide_index=True)

with tab_explorer:
    st.subheader("Searchable dataset")
    query = st.text_input("Search messages, authors, risk categories")
    priority_filter = st.multiselect("Priority filter", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
    sentiment_filter = st.multiselect("Sentiment filter", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    filtered = df[df["priority"].isin(priority_filter) & df["sentiment"].isin(sentiment_filter)].copy()
    if query:
        q = query.lower()
        mask = filtered["message"].fillna("").str.lower().str.contains(q, regex=False) | filtered["author_name"].fillna("").str.lower().str.contains(q, regex=False) | filtered["risk_categories"].fillna("").str.lower().str.contains(q, regex=False)
        filtered = filtered[mask]
    st.caption(str(len(filtered)) + " matching rows")
    st.dataframe(filtered[["page_name", "item_type", "created_time", "sentiment", "sentiment_score", "priority", "severity_score", "needs_response", "risk_categories", "author_name", "message", "permalink_url"]].head(1000).style.applymap(style_priority, subset=["priority"]), use_container_width=True, hide_index=True)

with tab_export:
    st.subheader("Exports")
    csv_all = df.to_csv(index=False).encode("utf-8")
    csv_flags = flagged_df.to_csv(index=False).encode("utf-8")
    csv_replies = needs_reply_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full dataset CSV", csv_all, "facebook_dashboard_full_dataset.csv", "text/csv", use_container_width=True)
    st.download_button("Download critical/high flags CSV", csv_flags, "facebook_dashboard_flags.csv", "text/csv", use_container_width=True)
    st.download_button("Download needs-response queue CSV", csv_replies, "facebook_dashboard_needs_response.csv", "text/csv", use_container_width=True)
    st.info("Tip: use the exports for weekly reporting, escalation logs, and response QA.")

st.caption("Last refresh: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))