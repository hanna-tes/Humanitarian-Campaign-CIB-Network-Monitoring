# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from itertools import combinations
import re
from io import StringIO
import tldextract

# --- Set Page Config ---
st.set_page_config(page_title="Humanitarian Campaign Monitor", layout="wide")
st.title("🕊️ Humanitarian Campaign Monitoring Dashboard")

# --- Add Background Image ---
def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://images.unsplash.com/photo-1587614382346-4ec70e388b28?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            color: white;
        }
        .css-1d391kg, .stMarkdown h1, h2, h3 {
            color: white !important;
            text-shadow: 2px 2px 4px #000;
        }
        .stSidebar {
            background-color: rgba(255, 255, 255, 0.95);
        }
        .stDataFrame, .stTable {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# --- Define the 6 key phrases to track ---
PHRASES_TO_TRACK = [
    "If you're scrolling, PLEASE leave a dot",
    "I'm so hungry, I'm not ashamed to say that",
    "3 replies — even dots — can break the algorithm",
    "My body is slowly falling apart from malnutrition, dizziness, and weight loss",
    "Good bye. If we die, don't forget us",
    "If you see this reply with a dot"
]

# --- Helper Functions ---
def infer_platform_from_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    if "tiktok.com" in url: return "TikTok"
    elif "facebook.com" in url or "fb.watch" in url: return "Facebook"
    elif "twitter.com" in url or "x.com" in url: return "X"
    elif "youtube.com" in url or "youtu.be" in url: return "YouTube"
    elif "instagram.com" in url: return "Instagram"
    elif "telegram.me" in url or "t.me" in url: return "Telegram"
    elif url.startswith("https://") or url.startswith("http://"):
        media_domains = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"]
        if any(domain in url for domain in media_domains): return "News/Media"
        return "Media"
    else: return "Unknown"

def extract_original_text(text):
    if pd.isna(text) or not isinstance(text, str): return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

def parse_timestamp_robust(timestamp):
    if pd.isna(timestamp): return None
    if isinstance(timestamp, (int, float)):
        if 0 < timestamp < 253402300800: return int(timestamp)
        else: return None
    date_formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']
    try:
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed): return int(parsed.timestamp())
    except: pass
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed): return int(parsed.timestamp())
        except (ValueError, TypeError): continue
    return None

def extract_all_urls(text):
    if pd.isna(text) or not isinstance(text, str): return []
    return re.findall(r'https?://\S+', text)

# --- Combine Datasets with Flexible Column Mapping ---
def combine_social_media_data(
    meltwater_df=None,
    civicsignals_df=None,
    openmeasure_df=None,
    meltwater_object_col='hit sentence',
    civicsignals_object_col='title',
    openmeasure_object_col='text'
):
    """
    Combines datasets from Meltwater, CivicSignals, and Open-Measure (optional).
    Allows specification of which column to use as 'object_id' for coordination analysis.
    Returns timestamp as UNIX integer.
    """
    combined_dfs = []

    def get_specific_col(df, col_name_lower):
        if col_name_lower in df.columns:
            return df[col_name_lower]
        return pd.Series([np.nan] * len(df), index=df.index)

    # Process Meltwater
    if meltwater_df is not None and not meltwater_df.empty:
        meltwater_df.columns = meltwater_df.columns.str.lower()
        mw = pd.DataFrame()
        mw['account_id'] = get_specific_col(meltwater_df, 'influencer')
        mw['content_id'] = get_specific_col(meltwater_df, 'tweet id')
        mw['object_id'] = get_specific_col(meltwater_df, meltwater_object_col.lower())
        mw['URL'] = get_specific_col(meltwater_df, 'url')
        mw['timestamp_share'] = get_specific_col(meltwater_df, 'date')
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)

    # Process CivicSignals
    if civicsignals_df is not None and not civicsignals_df.empty:
        civicsignals_df.columns = civicsignals_df.columns.str.lower()
        cs = pd.DataFrame()
        cs['account_id'] = get_specific_col(civicsignals_df, 'media_name')
        cs['content_id'] = get_specific_col(civicsignals_df, 'stories_id')
        cs['object_id'] = get_specific_col(civicsignals_df, civicsignals_object_col.lower())
        cs['URL'] = get_specific_col(civicsignals_df, 'url')
        cs['timestamp_share'] = get_specific_col(civicsignals_df, 'publish_date')
        cs['source_dataset'] = 'CivicSignals'
        combined_dfs.append(cs)

    # Process Open-Measure
    if openmeasure_df is not None and not openmeasure_df.empty:
        openmeasure_df.columns = openmeasure_df.columns.str.lower()
        om = pd.DataFrame()
        om['account_id'] = get_specific_col(openmeasure_df, 'actor_username')
        om['content_id'] = get_specific_col(openmeasure_df, 'id')
        om['object_id'] = get_specific_col(openmeasure_df, openmeasure_object_col.lower())
        om['URL'] = get_specific_col(openmeasure_df, 'url')
        om['timestamp_share'] = get_specific_col(openmeasure_df, 'created_at')
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)

    if not combined_dfs:
        return pd.DataFrame()

    combined = pd.concat(combined_dfs, ignore_index=True)
    combined = combined.dropna(subset=['account_id', 'content_id', 'timestamp_share', 'object_id']).copy()
    combined['account_id'] = combined['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    combined['content_id'] = combined['content_id'].astype(str).replace('nan', '').fillna('')
    combined['URL'] = combined['URL'].astype(str).replace('nan', '').fillna('')
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')
    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined = combined[combined['object_id'].str.strip() != ""].copy()
    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    if df.empty:
        return pd.DataFrame()

    df_processed = df.copy()
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    df_processed['timestamp_share'] = df_processed['timestamp_share'].astype('Int64')
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed['extracted_urls'] = df_processed['object_id'].apply(extract_all_urls)

    def assign_phrase(text):
        for p in PHRASES_TO_TRACK:
            if p.lower() in text:
                return p
        return "Other"
    df_processed['assigned_phrase'] = df_processed['original_text'].apply(assign_phrase)

    return df_processed

# --- Analysis Functions ---
def cluster_texts(df, eps, min_samples, max_features):
    if 'original_text' not in df.columns or df['original_text'].nunique() <= 1:
        df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
    texts = df['original_text'].astype(str).tolist()
    if not texts or all(t.strip() == "" for t in texts):
        df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    try:
        tfidf = vectorizer.fit_transform(texts)
    except: df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
    eps = max(0.01, min(0.99, eps))
    labels = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit_predict(tfidf)
    df_copy = df.copy()
    df_copy['cluster'] = labels
    return df_copy

def find_coordinated_groups(df, threshold, max_features):
    if 'cluster' not in df.columns or df.empty:
        return []
    groups = []
    for cluster_id, group in df[df['cluster'] != -1].groupby('cluster'):
        if len(group) < 2 or group['account_id'].nunique() < 2:
            continue
        clean = group[['account_id', 'timestamp_share', 'original_text']].copy()
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3,5), max_features=max_features)
            X = vectorizer.fit_transform(clean['original_text'])
            sim = cosine_similarity(X)
            high_sim = np.where(sim >= threshold)
            pairs = [(i, j) for i, j in zip(*high_sim) if i < j]
            if len(pairs) > 0:
                accounts = clean.iloc[[i for i, j in pairs]]['account_id'].unique()
                groups.append({
                    "posts": clean.rename(columns={'account_id': 'Account ID', 'timestamp_share': 'Timestamp', 'original_text': 'Text'}).to_dict('records'),
                    "num_posts": len(pairs),
                    "num_accounts": len(accounts),
                    "max_similarity_score": round(sim.max(), 3),
                    "cluster_id": cluster_id
                })
        except: continue
    return sorted(groups, key=lambda x: x['num_posts'], reverse=True)

def build_user_interaction_graph(df, coordination_type="text"):
    G = nx.Graph()
    influencer_column = 'account_id'
    if coordination_type == "text":
        if 'cluster' not in df.columns:
            return G, {}, {}
        for cluster_id, group in df[df['cluster'] != -1].groupby('cluster'):
            users = group[influencer_column].dropna().unique()
            if len(users) < 2: continue
            for u1, u2 in combinations(users, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)
    elif coordination_type == "url":
        if 'URL' not in df.columns:
            return G, {}, {}
        url_groups = df.groupby('URL')
        for url_shared, group in url_groups:
            if pd.isna(url_shared) or url_shared.strip() == "":
                continue
            users_sharing_url = group[influencer_column].dropna().unique().tolist()
            if len(users_sharing_url) < 2:
                for user in users_sharing_url:
                    if user not in G: G.add_node(user)
                continue
            for u1, u2 in combinations(users_sharing_url, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)

    all_influencers = df[influencer_column].dropna().unique().tolist()
    influencer_platform_map = df.groupby(influencer_column)['Platform'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    ).to_dict()

    for inf in all_influencers:
        if inf not in G: G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')

    if coordination_type == "text":
        cluster_map = df.groupby(influencer_column)['cluster'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else -2
        ).to_dict()
        for inf in G.nodes():
            G.nodes[inf]['cluster'] = cluster_map.get(inf, -2)
    elif coordination_type == "url":
        url_map = df.groupby(influencer_column)['URL'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        ).to_dict()
        for inf in G.nodes():
            G.nodes[inf]['url'] = url_map.get(inf, 'Unknown')

    if not G.nodes(): return G, {}, {}

    pos = nx.kamada_kawai_layout(G)
    color_map = {}
    for node in G.nodes():
        if coordination_type == "text":
            color_map[node] = G.nodes[node].get('cluster', -2)
        else:
            color_map[node] = hash(G.nodes[node].get('url', 'Unknown')) % 100

    return G, pos, color_map

# --- Caching ---
@st.cache_data
def cached_clustering(_df, eps, min_samples, max_features): return cluster_texts(_df, eps, min_samples, max_features)
@st.cache_data
def cached_find_coordinated_groups(_df, threshold, max_features): return find_coordinated_groups(_df, threshold, max_features)
@st.cache_data
def cached_network_graph(_df, coordination_type): return build_user_interaction_graph(_df, coordination_type)

# --- Auto-Encoding CSV Reader ---
def read_uploaded_file(uploaded_file, file_name):
    if not uploaded_file:
        return pd.DataFrame()
    bytes_data = uploaded_file.getvalue()
    encodings = ['utf-16le', 'utf-8-sig', 'utf-16be', 'latin1', 'cp1252']
    decoded_content = None
    detected_enc = None
    for enc in encodings:
        try:
            decoded_content = bytes_data.decode(enc)
            detected_enc = enc
            st.sidebar.info(f"✅ {file_name}: Decoded using '{enc}'")
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    if decoded_content is None:
        st.error(f"❌ Failed to read {file_name} CSV: Could not decode with any supported encoding.")
        return pd.DataFrame()
    sample_line = decoded_content.strip().splitlines()[0]
    sep = '\t' if '\t' in sample_line else ' ' if ' ' in sample_line else ','
    try:
        df = pd.read_csv(StringIO(decoded_content), sep=sep, on_bad_lines='skip', low_memory=False)
        st.sidebar.success(f"✅ {file_name}: Loaded {len(df)} rows (sep='{sep}', enc='{detected_enc}')")
        return df
    except Exception as e:
        st.error(f"❌ Failed to parse {file_name} CSV after decoding: {e}")
        return pd.DataFrame()

# --- Sidebar: Data Source Selector ---
st.sidebar.header("📥 Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Upload Files", "Use Default Dataset"])

# --- Load Default Dataset ---
if data_source == "Use Default Dataset":
    st.sidebar.info("Using default humanitarian dataset from GitHub.")
    with st.spinner("📥 Loading default dataset..."):
        base_url = "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/"
        default_url = f"{base_url}3_replies_%E2%80%94_even_dots_%E2%80%94_can_break_the_algorithm_AN%20-%20Aug%2013%2C%202025%20-%2010%2037%2011%20AM.csv"  # Replace with your actual humanitarian dataset
        try:
            meltwater_df_default = pd.read_csv(default_url, sep=' ')
            st.sidebar.success(f"✅ Default dataset loaded: {len(meltwater_df_default)} rows")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load default dataset: {e}")
            meltwater_df_default = pd.DataFrame()
    uploaded_meltwater_files = None
else:
    uploaded_meltwater_files = st.sidebar.file_uploader("Upload Meltwater CSV(s)", type="csv", accept_multiple_files=True, key="meltwater_uploads")

uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type="csv", key="civicsignals_upload")
uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type="csv", key="openmeasure_upload")

# Coordination mode selector
st.sidebar.header("🎯 Coordination Analysis Mode")
coordination_mode = st.sidebar.radio(
    "Analyze coordination by:",
    ("Text Content", "Shared URLs"),
    help="Choose what defines a coordinated action: similar text messages or sharing the same external link."
)

# --- Read and Combine ---
if 'combined_raw_df' not in st.session_state:
    st.session_state.combined_raw_df = pd.DataFrame()

if st.sidebar.button("🔄 Load & Combine Data"):
    with st.spinner("🔄 Loading and combining datasets..."):
        dfs = []

        if data_source == "Use Default Dataset" and 'meltwater_df_default' in locals():
            dfs.append(meltwater_df_default)

        if uploaded_meltwater_files:
            for i, file in enumerate(uploaded_meltwater_files):
                file.seek(0)
                df_temp = read_uploaded_file(file, f"Meltwater_{i+1}")
                if not df_temp.empty:
                    dfs.append(df_temp)

        if uploaded_civicsignals:
            uploaded_civicsignals.seek(0)
            df_temp = read_uploaded_file(uploaded_civicsignals, "CivicSignals")
            if not df_temp.empty:
                dfs.append(df_temp)

        if uploaded_openmeasure:
            uploaded_openmeasure.seek(0)
            df_temp = read_uploaded_file(uploaded_openmeasure, "Open-Measure")
            if not df_temp.empty:
                dfs.append(df_temp)

        if not dfs:
            st.error("❌ No files uploaded or all failed to load.")
        else:
            obj_map = {
                "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
                "civicsignals": "title" if coordination_mode == "Text Content" else "url",
                "openmeasure": "text" if coordination_mode == "Text Content" else "url"
            }
            st.session_state.combined_raw_df = combine_social_media_data(
                meltwater_df=pd.concat([d for d in dfs if 'influencer' in d.columns], ignore_index=True) if any('influencer' in d.columns for d in dfs) else None,
                civicsignals_df=next((d for d in dfs if 'media_name' in d.columns), None),
                openmeasure_df=next((d for d in dfs if 'actor_username' in d.columns), None),
                meltwater_object_col=obj_map["meltwater"],
                civicsignals_object_col=obj_map["civicsignals"],
                openmeasure_object_col=obj_map["openmeasure"]
            )
            if st.session_state.combined_raw_df.empty:
                st.error("❌ No valid data after combining. Check column names.")
            else:
                st.sidebar.success(f"✅ Combined {len(st.session_state.combined_raw_df):,} posts.")

# --- Preprocess ---
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if not st.session_state.combined_raw_df.empty and st.sidebar.button("🧹 Preprocess Data"):
    with st.spinner("🧹 Preprocessing data..."):
        st.session_state.df = final_preprocess_and_map_columns(st.session_state.combined_raw_df, coordination_mode=coordination_mode)
    if st.session_state.df.empty:
        st.error("❌ No valid data after preprocessing.")
    else:
        st.sidebar.success(f"✅ Processed {len(st.session_state.df):,} valid posts.")

# --- Filters ---
if not st.session_state.df.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")

    min_date = pd.to_datetime(st.session_state.df['timestamp_share'].min(), unit='s').date()
    max_date = pd.to_datetime(st.session_state.df['timestamp_share'].max(), unit='s').date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    start_ts = int(pd.Timestamp(date_range[0], tz='UTC').timestamp())
    end_ts = int(pd.Timestamp(date_range[1], tz='UTC').timestamp()) + 86399
    filtered_df = st.session_state.df[
        (st.session_state.df['timestamp_share'] >= start_ts) &
        (st.session_state.df['timestamp_share'] <= end_ts)
    ].copy()

    max_sample = st.sidebar.number_input("Max Posts for Analysis", 0, 100_000, 5000)
    if max_sample > 0 and len(filtered_df) > max_sample:
        df_for_analysis = filtered_df.sample(n=max_sample, random_state=42).copy()
        st.sidebar.warning(f"📊 Analyzing {len(df_for_analysis):,} sampled posts.")
    else:
        df_for_analysis = filtered_df.copy()
        st.sidebar.info(f"📊 Analyzing {len(df_for_analysis):,} posts.")
else:
    filtered_df = pd.DataFrame()
    df_for_analysis = pd.DataFrame()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Humanitarian Campaign Overview")
    st.markdown(f"**Coordination Mode:** `{coordination_mode}` | **Total Posts:** `{len(st.session_state.df):,}`")

    if not st.session_state.df.empty:
        # Sample data
        st.markdown("### 📋 Raw Data Sample (First 10 Rows)")
        display_df = st.session_state.df.copy()
        display_df['date_utc'] = pd.to_datetime(display_df['timestamp_share'], unit='s', utc=True)
        st.dataframe(display_df[['account_id', 'content_id', 'object_id', 'date_utc']].head(10), use_container_width=True)

        # Top Influencers
        st.markdown("---")
        top_influencers = st.session_state.df['account_id'].value_counts().head(10)
        fig_influencers = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Number of Posts', 'index': 'Account'})
        st.plotly_chart(fig_influencers, use_container_width=True)

        # Emotional Phrases
        st.markdown("---")
        def contains_phrase(text):
            if pd.isna(text): return "Other"
            text = str(text).lower()
            for p in PHRASES_TO_TRACK:
                if p.lower() in text:
                    return p
            return "Other"
        st.session_state.df['emotional_phrase'] = st.session_state.df['object_id'].apply(contains_phrase)
        phrase_counts = st.session_state.df['emotional_phrase'].value_counts()
        if "Other" in phrase_counts.index:
            phrase_counts = phrase_counts.drop("Other")
        if not phrase_counts.empty:
            fig_phrases = px.bar(phrase_counts, title="Posts by Emotional Phrase", labels={'value': 'Count', 'index': 'Phrase'})
            st.plotly_chart(fig_phrases, use_container_width=True)
        else:
            st.info("No emotional phrases detected.")

        # Daily Volume
        st.markdown("---")
        plot_df = st.session_state.df.copy()
        plot_df['date'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True).dt.date
        time_series = plot_df.groupby('date').size()
        fig_ts = px.line(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'index': 'Date'}, markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)

        # Platform Distribution
        st.markdown("---")
        if 'Platform' in st.session_state.df.columns:
            platform_counts = st.session_state.df['Platform'].value_counts()
            fig_platform = px.pie(platform_counts, names=platform_counts.index, values='Platform', title="Posts by Platform")
            st.plotly_chart(fig_platform, use_container_width=True)

# ==================== TAB 2: Analysis ====================
with tab2:
    st.subheader("🔍 Fundraising & Coordination Analysis")

    if coordination_mode == "Text Content":
        st.markdown("### 🔁 Similar Message Pairs Detected")
        eps = st.slider("DBSCAN eps", 0.1, 1.0, 0.3, 0.05, key="eps_tab2")
        min_samples = st.slider("Min Samples", 2, 10, 2, key="min_samples_tab2")
        max_features = st.slider("Max TF-IDF Features", 1000, 5000, 2000, 500, key="max_features_tab2")
        threshold = st.slider("Pairwise Similarity Threshold", 0.8, 0.99, 0.9, 0.01, key="threshold_tab2")

        if st.button("🔍 Run Similarity Analysis") and not df_for_analysis.empty:
            clustered = cached_clustering(df_for_analysis, eps, min_samples, max_features)
            groups = cached_find_coordinated_groups(clustered, threshold, max_features)
            if not groups:
                st.warning("No similar messages found above the threshold.")
            else:
                all_pairs = []
                for g in groups:
                    posts = g["posts"]
                    for i in range(len(posts)):
                        for j in range(i+1, len(posts)):
                            sim_score = cosine_similarity(
                                TfidfVectorizer(stop_words='english').fit_transform([posts[i]['Text'], posts[j]['Text']])
                            )[0,1]
                            if sim_score >= threshold:
                                all_pairs.append({
                                    "Similarity Score": round(sim_score, 3),
                                    "Account 1": posts[i]['Account ID'],
                                    "Post 1 URL": posts[i]['URL'],
                                    "Account 2": posts[j]['Account ID'],
                                    "Post 2 URL": posts[j]['URL'],
                                    "Shared Text": posts[i]['Text'][:200] + "..."
                                })
                pairs_df = pd.DataFrame(all_pairs).sort_values("Similarity Score", ascending=False)
                st.dataframe(pairs_df, use_container_width=True)

                @st.cache_data
                def convert_df(_df):
                    return _df.to_csv(index=False).encode('utf-8')
                csv = convert_df(pairs_df)
                st.download_button("📥 Download Similar Pairs Report", csv, "similar_pairs.csv", "text/csv")

    # --- Fundraising Analysis ---
    st.markdown("---")
    st.subheader("💰 Suspicious Fundraising Links")

    if 'extracted_urls' in df_for_analysis.columns:
        all_urls = df_for_analysis.explode('extracted_urls').dropna(subset=['extracted_urls'])
        fundraising_domains = ['gofundme.com', 'paypal.me', 'donorbox.org']
        all_urls['domain'] = all_urls['extracted_urls'].apply(lambda x: tldextract.extract(x).registered_domain)
        fundraising_links = all_urls[all_urls['domain'].isin(fundraising_domains)]

        if not fundraising_links.empty:
            summary = fundraising_links.groupby('extracted_urls').agg(
                Num_Posts=('content_id', 'size'),
                Num_Unique_Accounts=('account_id', 'nunique'),
                First_Shared=('timestamp_share', 'min'),
                Last_Shared=('timestamp_share', 'max')
            ).reset_index()
            summary['First_Shared'] = pd.to_datetime(summary['First_Shared'], unit='s', utc=True)
            summary['Last_Shared'] = pd.to_datetime(summary['Last_Shared'], unit='s', utc=True)
            summary['Coordination_Score'] = (summary['Num_Posts'] / summary['Num_Unique_Accounts']).round(2)
            summary['Risk_Flag'] = 'Medium'
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No known fundraising links detected.")
    else:
        st.info("No URL data available.")

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🌐 Network of Coordinated Accounts")

    if 'max_nodes_to_display' not in st.session_state:
        st.session_state.max_nodes_to_display = 40
    st.session_state.max_nodes_to_display = st.slider(
        "Maximum Nodes to Display",
        min_value=10, max_value=200, value=st.session_state.max_nodes_to_display, step=10,
        help="Limit the graph to the top N most connected accounts."
    )

    if st.button("Build Network Graph") and not df_for_analysis.empty:
        with st.spinner("Building network..."):
            G, pos, color_map = cached_network_graph(df_for_analysis, coordination_type="text" if coordination_mode == "Text Content" else "url")

        if not G.nodes():
            st.warning("No coordinated activity detected.")
        else:
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:st.session_state.max_nodes_to_display]
            subgraph = G.subgraph(top_nodes)
            pos_filtered = {node: pos[node] for node in top_nodes}

            edge_x, edge_y = [], []
            for u, v in subgraph.edges():
                x0, y0 = pos_filtered[u]; x1, y1 = pos_filtered[v]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

            node_x, node_y, node_text, node_color = [], [], [], []
            for node in subgraph.nodes():
                x, y = pos_filtered[node]; node_x.append(x); node_y.append(y)
                deg = subgraph.degree(node)
                platform = subgraph.nodes[node].get('platform', 'Unknown')
                node_text.append(f"{node}<br>deg: {deg} • {platform}")
                node_color.append(color_map.get(node, -2))

            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                                  marker=dict(size=[subgraph.degree(n)*3 + 10 for n in subgraph.nodes()],
                                              color=node_color, colorscale='Viridis', showscale=True))
            fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="CIB Network", showlegend=False, hovermode='closest'))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🚨 Risk Assessment: Top Central Accounts")
            degree_centrality = nx.degree_centrality(subgraph)
            risk_df = pd.DataFrame(degree_centrality.items(), columns=['Account', 'Centrality']).sort_values('Centrality', ascending=False).head(20)
            risk_df['Risk Score'] = (risk_df['Centrality'] * 100).round(2)
            st.dataframe(risk_df, use_container_width=True)
