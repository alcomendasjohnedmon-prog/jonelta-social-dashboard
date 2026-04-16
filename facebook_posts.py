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

PAGE_ID = "111665584503975"  # Replace with your actual page ID if needed
PAGE_NAME = "Statistician for Undergraduate/ Graduate Level Thesis"

GRAPH_API_BASE = "https://graph.facebook.com/v25.0"
TOKEN = "EAAeRoPHMTT8BREPqnucG28ak1QftCXT47ujoZAVQ5PuI0PKlsraNAUe3GnGuFSdAgUZAPRPOr4sijZAZAbAZB6GuXDfDhQ8bggqky9wLThk6HP4cIt0a1PP9PXDOQfanACNW3k1Ic6oBWuEHqbgK2mlJHyQww1e0rDyRH3pJJtBD0U3iUdLraK9cmk0n51xxVOjiBKpFlkmroTchhUQJ8h7qnZCbtAYqVw6dQq1jy60BAqHk3dXT4xVKv20L5L8ZCC4OmWOzzIomB4ZD"
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
    "like_reactions",
    "love_reactions",
    "care_reactions",
    "haha_reactions",
    "wow_reactions",
    "sad_reactions",
    "angry_reactions"
]
PAGE_ACCOUNTS_DATA = []
PAGE_ACCOUNTS_RAW = {}
PAGE_POSTS_RAW = {}

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0b3d91 0%, #1f5c99 35%, #4c8cd7 100%);
        color: #f0f4fb;
    }
    .css-1d391kg, .css-18e3th9 {background: rgba(255, 255, 255, 0.08) !important;}
    .stButton>button {
        background-color: #1f77b4;
        color: #ffffff;
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        color: #ffffff;
    }
    section[data-testid='stSidebar'] {
        background: rgba(8, 27, 58, 0.98);
        color: #f0f4fb;
    }
    [data-testid='metric-container'] {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 18px !important;
        padding: 1rem !important;
        box-shadow: 0 18px 45px rgba(0,0,0,0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    # Flag as potentially malicious if polarity is negative and subjectivity high
    flag = "Safe"
    if sentiment.polarity < -0.1 and sentiment.subjectivity > 0.5:
        flag = "Flag for Review"
    return sentiment.polarity, sentiment.subjectivity, flag

@st.cache_data
def fetch_page_accounts():
    global PAGE_ACCOUNTS_RAW
    url = f"{GRAPH_API_BASE}/me/accounts?fields={PAGE_ACCOUNT_FIELDS}&access_token={TOKEN}"
    response = requests.get(url)
    data = response.json()
    PAGE_ACCOUNTS_RAW = data
    page_accounts = []
    if "data" in data:
        for account in data["data"]:
            page_accounts.append({
                "page_id": account.get("id"),
                "page_name": account.get("name"),
                "category": account.get("category"),
                "category_list": account.get("category_list", []),
                "tasks": account.get("tasks", []),
                "page_access_token": account.get("access_token")
            })
    return page_accounts


def fetch_comments(post_id, access_token):
    url = f"{GRAPH_API_BASE}/{post_id}/comments?fields=id,message,created_time,reactions.summary(true)&access_token={access_token}"
    response = requests.get(url)
    data = response.json()
    comments = []
    if "data" in data:
        for comment in data["data"]:
            message = comment.get("message", "")
            polarity, subjectivity, flag = analyze_sentiment(message)
            reactions_total = comment.get("reactions", {}).get("summary", {}).get("total_count", 0)
            comments.append({
                "comment_id": comment.get("id"),
                "message": message,
                "created_time": comment.get("created_time"),
                "polarity": polarity,
                "subjectivity": subjectivity,
                "flag": flag,
                "reactions_total": reactions_total
            })
    return comments

@st.cache_data
def fetch_data():
    global PAGE_ACCOUNTS_DATA, PAGE_POSTS_RAW
    all_data = []
    page_accounts = fetch_page_accounts()
    PAGE_ACCOUNTS_DATA = page_accounts

    if not page_accounts:
        st.error("No page accounts returned from /me/accounts. Please check your user access token and permissions.")
        return pd.DataFrame(all_data)

    page = next((p for p in page_accounts if p["page_id"] == PAGE_ID), page_accounts[0])
    page_name = page.get("page_name", PAGE_NAME)
    page_id = page.get("page_id", PAGE_ID)
    page_token = page.get("page_access_token", TOKEN)

    url = f"{GRAPH_API_BASE}/{page_id}/posts?fields={PAGE_POST_FIELDS}&access_token={page_token}"
    response = requests.get(url)
    data = response.json()
    PAGE_POSTS_RAW = data

    if "data" not in data:
        st.error(f"Error fetching posts for {page_name}: {data}")
        return pd.DataFrame(all_data)

    for post in data["data"]:
        message = post.get("message", "")
        polarity, subjectivity, flag = analyze_sentiment(message)
        reactions_total = post.get("reactions", {}).get("summary", {}).get("total_count", 0)
        comments_total = post.get("comments", {}).get("summary", {}).get("total_count", 0)
        created_time = post.get("created_time")
        post_data = {
            "type": "post",
            "page_name": page_name,
            "page_id": page_id,
            "id": post.get("id"),
            "message": message,
            "created_time": created_time,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "flag": flag,
            "reactions_total": reactions_total,
            "comments_total": comments_total,
            "like_reactions": post.get("like_reactions", {}).get("summary", {}).get("total_count", 0),
            "love_reactions": post.get("love_reactions", {}).get("summary", {}).get("total_count", 0),
            "care_reactions": post.get("care_reactions", {}).get("summary", {}).get("total_count", 0),
            "haha_reactions": post.get("haha_reactions", {}).get("summary", {}).get("total_count", 0),
            "wow_reactions": post.get("wow_reactions", {}).get("summary", {}).get("total_count", 0),
            "sad_reactions": post.get("sad_reactions", {}).get("summary", {}).get("total_count", 0),
            "angry_reactions": post.get("angry_reactions", {}).get("summary", {}).get("total_count", 0)
        }
        all_data.append(post_data)

        # Fetch comments for this post
        comments = fetch_comments(post.get("id"), page_token)
        for comment in comments:
            comment["post_id"] = post.get("id")
            comment["page_name"] = page_name
            comment["type"] = "comment"
            all_data.append(comment)

    df = pd.DataFrame(all_data)
    if not df.empty:
        for col in REACTION_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df['created_time'] = pd.to_datetime(df['created_time'])
        df['date'] = df['created_time'].dt.date
        df['hour'] = df['created_time'].dt.hour
        df['day'] = df['created_time'].dt.day_name()
    return df

# Streamlit App
st.title("UPHS-JONELTA Social Media Engagement Dashboard")
st.markdown(
    """
    ### A centralized, polished view of page engagement, sentiment, and reaction trends.
    Explore top-performing posts, emoji reaction mix, and safety flags"""
)

st.sidebar.header("Controls")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = fetch_data()

if PAGE_ACCOUNTS_DATA:
    page_df = pd.DataFrame([{
        "page_name": p["page_name"],
        "page_id": p["page_id"],
        "category": p["category"],
        "tasks": ", ".join(p.get("tasks", []))
    } for p in PAGE_ACCOUNTS_DATA if p["page_id"] == PAGE_ID])
    if page_df.empty:
        page_df = pd.DataFrame([{
            "page_name": PAGE_NAME,
            "page_id": PAGE_ID,
            "category": "",
            "tasks": ""
        }])
    st.subheader("Connected Facebook Page")
    st.dataframe(page_df)

with st.expander("Raw API Response Preview"):
    st.subheader("Page Accounts API Response")
    st.json(PAGE_ACCOUNTS_RAW)
    st.subheader("Page Posts API Response")
    st.json(PAGE_POSTS_RAW)

if df.empty:
    st.error("No data fetched. Please check your token and page ID.")
else:
    # Flagging System
    flagged = df[df['flag'] == 'Flag for Review']
    if not flagged.empty:
        st.warning(f"🚨 Alert: {len(flagged)} potentially harmful posts/comments detected!")
        st.subheader("Flagged Content")
        st.dataframe(flagged[['page_name', 'type', 'message', 'polarity', 'subjectivity', 'flag']])
    else:
        st.success("No flagged content detected.")

    # Highest Traffic
    st.subheader("Highest Social Media Traffic")
    traffic = df.groupby(['date', 'day', 'hour']).size().reset_index(name='count')
    traffic = traffic.sort_values('count', ascending=False)
    st.dataframe(traffic.head(10))
    
    # Plot traffic over time
    fig = px.line(traffic, x='date', y='count', title='Traffic Over Time')
    st.plotly_chart(fig)

    # Most Frequent Engagements
    st.subheader("Most Frequent Reactions and Engagements")
    engagements = df.groupby('page_name').agg({
        'reactions_total': 'sum',
        'comments_total': 'sum'
    }).reset_index()
    engagements['total_engagements'] = engagements['reactions_total'] + engagements['comments_total']
    engagements = engagements.sort_values('total_engagements', ascending=False)
    st.dataframe(engagements)
    
    fig2 = px.bar(
        engagements,
        x='page_name',
        y='total_engagements',
        title='Total Engagements for Page',
        color='page_name',
        template='plotly_dark'
    )
    fig2.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2)

    # Reaction Type Metrics
    st.subheader("Reaction Type Trends")
    reaction_sums = df[df['type'] == 'post'][REACTION_COLUMNS].sum()
    reaction_sums = reaction_sums[reaction_sums > 0].sort_values(ascending=False)
    if not reaction_sums.empty:
        reaction_labels = [col.replace('_reactions', '').title() for col in reaction_sums.index]
        reaction_fig = px.bar(
            x=reaction_labels,
            y=reaction_sums.values,
            labels={'x': 'Reaction Type', 'y': 'Count'},
            title='Reaction Type Distribution Across Posts',
            color=reaction_labels,
            template='plotly_dark'
        )
        reaction_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(reaction_fig)

        top_reaction_type = reaction_labels[0]
        top_reaction_count = int(reaction_sums.iloc[0])
        unique_reaction_types = len(reaction_sums)
    else:
        top_reaction_type = 'None yet'
        top_reaction_count = 0
        unique_reaction_types = 0

    st.subheader("Overall Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Posts/Comments", len(df))
    with col2:
        st.metric("Flagged Items", len(flagged))
    with col3:
        avg_polarity = df['polarity'].mean()
        st.metric("Average Sentiment", f"{avg_polarity:.2f}")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Top Reaction Type", top_reaction_type)
    with col5:
        st.metric("Top Reaction Count", top_reaction_count)
    with col6:
        st.metric("Reaction Types Seen", unique_reaction_types)
