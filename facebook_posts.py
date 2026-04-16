import requests
import pandas as pd
from textblob import TextBlob
import streamlit as st
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="UPHS-JONELTA Social Media Dashboard-prototype",
    page_icon="📊",
    layout="wide"
)

PAGE_ID = "111665584503975"
PAGE_NAME = "Statistician for Undergraduate/ Graduate Level Thesis"

GRAPH_API_BASE = "https://graph.facebook.com/v25.0"
TOKEN = st.secrets["FB_TOKEN"]

PAGE_ACCOUNT_FIELDS = "id,name,category,category_list,tasks,access_token"

PAGE_POST_FIELDS = (
    "id,message,created_time,reactions.summary(true),"
    "like_reactions:reactions.type(LIKE).summary(total_count).limit(0),"
    "love_reactions:reactions.type(LOVE).summary(total_count).limit(0),"
    "care_reactions:reactions.type(CARE).summary(total_count).limit(0),"
    "haha_reactions:reactions.type(HAHA).summary(total_count).limit(0),"
    "wow_reactions:reactions.type(WOW).summary(total_count).limit(0),"
    "sad_reactions:reactions.type(SAD).summary(total_count).limit(0),"
    "angry_reactions:reactions.type(ANGRY).summary(total_count).limit(0),"
    "comments.summary(true)"
)

REACTION_COLUMNS = [
    "like_reactions", "love_reactions", "care_reactions",
    "haha_reactions", "wow_reactions", "sad_reactions", "angry_reactions"
]

PAGE_ACCOUNTS_DATA = []
PAGE_ACCOUNTS_RAW = {}
PAGE_POSTS_RAW = {}

# ---------------- TOKEN HEALTH CHECK ----------------
def check_token_health(user_token):
    APP_ID = st.secrets["APP_ID"]
    APP_SECRET = st.secrets["APP_SECRET"]

    app_token = f"{APP_ID}|{APP_SECRET}"

    url = "https://graph.facebook.com/debug_token"

    params = {
        "input_token": user_token,
        "access_token": app_token
    }

    try:
        res = requests.get(url, params=params).json()

        if "error" in res:
            return None, None, False

        data = res.get("data", {})
        expires_at = data.get("expires_at")
        is_valid = data.get("is_valid", False)

        if not expires_at:
            return None, None, is_valid

        expiry_date = datetime.fromtimestamp(expires_at)
        days_left = (expiry_date - datetime.now()).days

        return days_left, expiry_date, is_valid

    except:
        return None, None, False

# ---------------- UI HEADER ----------------
st.title("UPHS-JONELTA Social Media Engagement Dashboard-prototype")

st.markdown("""
### A centralized, polished view of page engagement, sentiment, and reaction trends.
Explore top-performing posts, emoji reaction mix, and safety flags
""")

# ---------------- TOKEN STATUS DISPLAY ----------------
days_left, expiry_date, is_valid = check_token_health(TOKEN)

if not is_valid:
    st.error("❌ Token INVALID — refresh immediately")
elif days_left is not None:
    if days_left < 5:
        st.error(f"🚨 Token expiring VERY SOON ({days_left} days)")
    elif days_left < 15:
        st.warning(f"⚠️ Token expires in {days_left} days")
    else:
        st.success(f"✅ Token valid ({days_left} days left)")

    st.caption(f"Expiry date: {expiry_date}")
else:
    st.warning("⚠️ Unable to determine token status")

# Sidebar visibility
if days_left:
    st.sidebar.metric("Token Days Left", days_left)

# ---------------- FUNCTIONS ----------------
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    flag = "Safe"
    if sentiment.polarity < -0.1 and sentiment.subjectivity > 0.5:
        flag = "Flag for Review"
    return sentiment.polarity, sentiment.subjectivity, flag


@st.cache_data
def fetch_page_accounts():
    global PAGE_ACCOUNTS_RAW
    url = f"{GRAPH_API_BASE}/me/accounts?fields={PAGE_ACCOUNT_FIELDS}&access_token={TOKEN}"
    data = requests.get(url).json()
    PAGE_ACCOUNTS_RAW = data

    if "error" in data:
        st.error(f"❌ Facebook API Error: {data['error']['message']}")
        return []

    page_accounts = []
    for account in data.get("data", []):
        page_accounts.append({
            "page_id": account.get("id"),
            "page_name": account.get("name"),
            "category": account.get("category"),
            "tasks": account.get("tasks", []),
            "page_access_token": account.get("access_token")
        })

    return page_accounts


def fetch_comments(post_id, access_token):
    url = f"{GRAPH_API_BASE}/{post_id}/comments?fields=id,message,created_time,reactions.summary(true)&access_token={access_token}"
    data = requests.get(url).json()

    comments = []
    if "data" in data:
        for comment in data["data"]:
            polarity, subjectivity, flag = analyze_sentiment(comment.get("message", ""))
            comments.append({
                "comment_id": comment.get("id"),
                "message": comment.get("message", ""),
                "created_time": comment.get("created_time"),
                "polarity": polarity,
                "subjectivity": subjectivity,
                "flag": flag,
                "reactions_total": comment.get("reactions", {}).get("summary", {}).get("total_count", 0)
            })
    return comments


@st.cache_data
def fetch_data():
    global PAGE_ACCOUNTS_DATA, PAGE_POSTS_RAW

    all_data = []
    page_accounts = fetch_page_accounts()
    PAGE_ACCOUNTS_DATA = page_accounts

    if not page_accounts:
        return pd.DataFrame()

    page = next((p for p in page_accounts if p["page_id"] == PAGE_ID), page_accounts[0])
    page_token = page["page_access_token"]
    page_id = page["page_id"]
    page_name = page["page_name"]

    url = f"{GRAPH_API_BASE}/{page_id}/posts?fields={PAGE_POST_FIELDS}&access_token={page_token}"
    data = requests.get(url).json()
    PAGE_POSTS_RAW = data

    if "error" in data:
        st.error(f"❌ Posts Fetch Error: {data['error']['message']}")
        return pd.DataFrame()

    if "data" not in data:
        st.warning("No posts returned.")
        return pd.DataFrame()

    for post in data["data"]:
        polarity, subjectivity, flag = analyze_sentiment(post.get("message", ""))

        post_data = {
            "type": "post",
            "page_name": page_name,
            "page_id": page_id,
            "id": post.get("id"),
            "message": post.get("message", ""),
            "created_time": post.get("created_time"),
            "polarity": polarity,
            "subjectivity": subjectivity,
            "flag": flag,
            "reactions_total": post.get("reactions", {}).get("summary", {}).get("total_count", 0),
            "comments_total": post.get("comments", {}).get("summary", {}).get("total_count", 0),
        }

        for col in REACTION_COLUMNS:
            post_data[col] = post.get(col, {}).get("summary", {}).get("total_count", 0)

        all_data.append(post_data)

        comments = fetch_comments(post.get("id"), page_token)
        for c in comments:
            c["type"] = "comment"
            c["page_name"] = page_name
            c["post_id"] = post.get("id")
            all_data.append(c)

    df = pd.DataFrame(all_data)

    if not df.empty:
        df["created_time"] = pd.to_datetime(df["created_time"])
        df["date"] = df["created_time"].dt.date
        df["hour"] = df["created_time"].dt.hour
        df["day"] = df["created_time"].dt.day_name()

    return df


# ---------------- UI CONTROLS ----------------
st.sidebar.header("Controls")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = fetch_data()

# ---------------- DISPLAY ----------------
if df.empty:
    st.error("No data fetched.")
else:
    flagged = df[df['flag'] == 'Flag for Review']

    if not flagged.empty:
        st.warning(f"🚨 {len(flagged)} flagged items detected")
        st.dataframe(flagged[['page_name', 'type', 'message']])
    else:
        st.success("No flagged content detected.")

    traffic = df.groupby(['date', 'day', 'hour']).size().reset_index(name='count')
    st.subheader("Traffic")
    st.dataframe(traffic.head(10))
    st.plotly_chart(px.line(traffic, x='date', y='count'))

    engagements = df.groupby('page_name')[['reactions_total', 'comments_total']].sum().reset_index()
    engagements['total'] = engagements['reactions_total'] + engagements['comments_total']

    st.subheader("Engagements")
    st.dataframe(engagements)

    st.plotly_chart(px.bar(engagements, x='page_name', y='total'))

# ---------------- LAST REFRESH ----------------
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
