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

# --- Add Professional Background Image ---
def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1587614382346-4ec70e388b28?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
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
    Combines datasets from Meltwater, CivicSignals, and Open-Measure.
    Maps original column names to core analysis columns.
    Handles case-insensitive column matching.
    """
    combined_dfs = []

    def get_col(df, possible_names, default=None):
        if df is None or df.empty: return pd.Series([], dtype='object')
        for name in possible_names:
            if name.lower() in df.columns.str.lower():
                col = df.columns[df.columns.str.lower() == name.lower()][0]
                return df[col]
        return pd.Series([np.nan] * len(df), index=df.index) if default is None else pd.Series([default] * len(df), index=df.index)

    # Process Meltwater
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer', 'Author', 'author', 'User'], 'Unknown_User')
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'Post ID', 'post_id', 'id'], 'Unknown_ID')
        mw['object_id'] = get_col(meltwater_df, [meltwater_object_col, 'Content', 'content', 'Text'], '')
        mw['URL'] = get_col(meltwater_df, ['url', 'URL', 'Link', 'link'], '')
        mw['timestamp_share'] = get_col(meltwater_df, ['date', 'Published Date', 'publish_date', 'Date', 'created_at'], None)
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)

    # Process CivicSignals
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'Outlet', 'outlet', 'Publisher'], 'Unknown_Outlet')
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'story_id', 'id'], 'Unknown_ID')
        cs['object_id'] = get_col(civicsignals_df, [civicsignals_object_col, 'title', 'Title'], '')
        cs['URL'] = get_col(civicsignals_df, ['url', 'URL', 'Story URL'], '')
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'date', 'Date'], None)
        cs['source_dataset'] = 'CivicSignals'
        combined_dfs.append(cs)

    # Process Open-Measure
    if openmeasure_df is not None and not openmeasure_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasure_df, ['actor_username', 'username', 'user'], 'Unknown_User')
        om['content_id'] = get_col(openmeasure_df, ['id', 'post_id'], 'Unknown_ID')
        om['object_id'] = get_col(openmeasure_df, [openmeasure_object_col, 'text', 'content'], '')
        om['URL'] = get_col(openmeasure_df, ['url', 'link'], '')
        om['timestamp_share'] = get_col(openmeasure_df, ['created_at', 'date', 'timestamp'], None)
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
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share'], ignore_index=True)
    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df):
    if df.empty:
        return pd.DataFrame()

    df_processed = df.copy()
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    df_processed['timestamp_share'] = df_processed['timestamp_share'].astype('Int64')
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed['extracted_urls'] = df_processed['object_id'].apply(extract_all_urls)

    # Assign phrase
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

def build_user_interaction_graph(df):
    G = nx.Graph()
    if 'cluster' not in df.columns:
        return G, {}, {}
    for cluster_id, group in df[df['cluster'] != -1].groupby('cluster'):
        users = group['account_id'].dropna().unique()
        if len(users) < 2: continue
        for u1, u2 in combinations(users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]['weight'] += 1
            else:
                G.add_edge(u1, u2, weight=1)
    for user in df['account_id'].dropna().unique():
        if user not in G: G.add_node(user)
    if not G.nodes(): return G, {}, {}
    pos = nx.kamada_kawai_layout(G)
    cluster_map = df.groupby('account_id')['cluster'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else -2).to_dict()
    return G, pos, cluster_map

# --- Caching ---
@st.cache_data
def cached_clustering(_df, eps, min_samples, max_features): return cluster_texts(_df, eps, min_samples, max_features)
@st.cache_data
def cached_find_coordinated_groups(_df, threshold, max_features): return find_coordinated_groups(_df, threshold, max_features)
@st.cache_data
def cached_network_graph(_df): return build_user_interaction_graph(_df)

# --- Auto-Encoding CSV Reader ---
def read_uploaded_file_with_encoding_detection(uploaded_file, file_name):
    if not uploaded_file:
        return pd.DataFrame()
    bytes_data = uploaded_file.getvalue()
    encodings = ['utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin1', 'cp1252']
    for enc in encodings:
        try:
            decoded = bytes_data.decode(enc)
            st.sidebar.info(f"✅ {file_name}: Decoded with `{enc}`")
            sample = decoded.splitlines()[0] if decoded.splitlines() else ""
            sep = '\t' if '\t' in sample else ','
            df = pd.read_csv(StringIO(decoded), sep=sep, on_bad_lines='skip', low_memory=False)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to parse {file_name}: {e}")
            continue
    st.sidebar.error(f"❌ Failed to decode '{file_name}'.")
    return pd.DataFrame()

# --- Sidebar: Upload Multiple Files ---
st.sidebar.header("📤 Upload Your Data")
st.sidebar.info("Upload one or more CSV files. You can upload multiple Meltwater files.")

uploaded_meltwater_files = st.sidebar.file_uploader(
    "Upload Meltwater CSV(s)", type="csv", accept_multiple_files=True, key="meltwater_uploads"
)
uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type="csv", key="civicsignals_upload")
uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type="csv", key="openmeasure_upload")

# --- Read and Combine ---
if 'combined_raw_df' not in st.session_state:
    st.session_state.combined_raw_df = pd.DataFrame()

# Coordination mode selector
coordination_mode = st.sidebar.radio(
    "Select Coordination Mode",
    options=["Text Content", "URL"],
    help="Choose whether to analyze coordination based on text similarity or shared URLs."
)

if st.sidebar.button("🔄 Load & Combine Data"):
    with st.spinner("🔄 Loading and combining datasets..."):
        dfs = []

        if uploaded_meltwater_files:
            for i, file in enumerate(uploaded_meltwater_files):
                file.seek(0)
                df_temp = read_uploaded_file_with_encoding_detection(file, f"Meltwater_{i+1}")
                if not df_temp.empty:
                    dfs.append(df_temp)

        if uploaded_civicsignals:
            uploaded_civicsignals.seek(0)
            df_temp = read_uploaded_file_with_encoding_detection(uploaded_civicsignals, "CivicSignals")
            if not df_temp.empty:
                dfs.append(df_temp)

        if uploaded_openmeasure:
            uploaded_openmeasure.seek(0)
            df_temp = read_uploaded_file_with_encoding_detection(uploaded_openmeasure, "Open-Measure")
            if not df_temp.empty:
                dfs.append(df_temp)

        if not dfs:
            st.error("❌ No files uploaded or all failed to load.")
        else:
            # Map object column based on mode
            obj_map = {
                "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
                "civicsignals": "title" if coordination_mode == "Text Content" else "url",
                "openmeasure": "text" if coordination_mode == "Text Content" else "url"
            }
            st.session_state.combined_raw_df = combine_social_media_data(
                meltwater_df=pd.concat([d for d in dfs if 'influencer' in d.columns or any(kw in [c.lower() for c in d.columns] for kw in ['author', 'content'])], ignore_index=True) if any('influencer' in d.columns or any(kw in [c.lower() for c in d.columns] for kw in ['author', 'content']) for d in dfs) else None,
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
        st.session_state.df = final_preprocess_and_map_columns(st.session_state.combined_raw_df)
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
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Similarity & Coordination", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Overview of Combined Data")

    if not st.session_state.df.empty:
        display_df = st.session_state.df.copy()
        display_df['date_utc'] = pd.to_datetime(display_df['timestamp_share'], unit='s', utc=True)

        st.markdown("### 📋 Raw Data Sample (First 10 Rows)")
        st.dataframe(display_df[['account_id', 'content_id', 'object_id', 'date_utc']].head(10), use_container_width=True)

        top_influencers = st.session_state.df['account_id'].value_counts().head(10)
        fig_influencers = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Number of Posts', 'index': 'Account'})
        st.plotly_chart(fig_influencers, use_container_width=True)

        phrase_counts = st.session_state.df['assigned_phrase'].value_counts()
        if "Other" in phrase_counts.index:
            phrase_counts = phrase_counts.drop("Other")
        fig_phrases = px.bar(phrase_counts, title="Posts by Emotional Phrase", labels={'value': 'Count', 'index': 'Phrase'})
        st.plotly_chart(fig_phrases, use_container_width=True)

        plot_df = st.session_state.df.copy()
        plot_df['date'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True).dt.date
        time_series = plot_df.groupby('date').size()
        fig_ts = px.line(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'index': 'Date'})
        st.plotly_chart(fig_ts, use_container_width=True)

        if 'Platform' in st.session_state.df.columns:
            platform_counts = st.session_state.df['Platform'].value_counts()
            fig_platform = px.bar(platform_counts, title="Posts by Platform", labels={'value': 'Count', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True)
    else:
        st.info("Upload and preprocess data to see overview.")

# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🔍 Detection of Similar Messages")

    eps = st.slider("DBSCAN eps", 0.1, 1.0, 0.3, 0.05, key="eps_tab2")
    min_samples = st.slider("Min Samples", 2, 10, 2, key="min_samples_tab2")
    max_features = st.slider("Max TF-IDF Features", 1000, 5000, 2000, 500, key="max_features_tab2")
    threshold = st.slider("Pairwise Similarity Threshold", 0.8, 0.99, 0.9, 0.01, key="threshold_tab2")

    if st.button("🔍 Run Similarity Analysis") and not df_for_analysis.empty:
        with st.spinner("Clustering..."):
            clustered = cached_clustering(df_for_analysis, eps, min_samples, max_features)
        with st.spinner("Finding groups..."):
            groups = cached_find_coordinated_groups(clustered, threshold, max_features)

        if not groups:
            st.warning("No similar messages found above the threshold.")
        else:
            st.success(f"✅ Found {len(groups)} coordinated groups.")

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

            st.markdown("### 🔁 Similar Message Pairs Detected")
            st.dataframe(pairs_df, use_container_width=True)

            @st.cache_data
            def convert_df(_df):
                return _df.to_csv(index=False).encode('utf-8')

            csv = convert_df(pairs_df)
            st.download_button("📥 Download Similar Pairs Report", csv, "similar_pairs.csv", "text/csv")

    # ==================== NEW: Suspicious Fundraising Section ====================
    st.markdown("---")
    st.subheader("💰 Suspicious Fundraising Links Detected")

    if 'extracted_urls' not in df_for_analysis.columns:
        st.info("No URL data available.")
    else:
        # Extract all URLs
        all_urls = df_for_analysis.explode('extracted_urls')
        all_urls = all_urls.dropna(subset=['extracted_urls'])
        all_urls['clean_url'] = all_urls['extracted_urls'].str.strip().str.lower()

        # Define domains
        trusted_domains = {
            'gofundme.com', 'paypal.me', 'unicef.org', 'icrc.org', 'doctorswithoutborders.org',
            'redcross.org', 'globalgiving.org', 'charitywater.org', 'donorbox.org'
        }
        suspicious_keywords = ['fund', 'donate', 'give', 'support', 'relief']

        def get_domain(url):
            from urllib.parse import urlparse
            try:
                return urlparse(url).netloc.lower().strip()
            except:
                return ""

        all_urls['domain'] = all_urls['clean_url'].apply(get_domain)
        all_urls['is_fundraising'] = all_urls['domain'].apply(
            lambda d: 'Trusted' if d in trusted_domains
            else 'Suspicious' if d and (any(k in d for k in suspicious_keywords) or '.me' in d)
            else 'Other'
        )

        fundraising_df = all_urls[all_urls['is_fundraising'].isin(['Trusted', 'Suspicious'])].copy()

        if fundraising_df.empty:
            st.info("No fundraising links found in the data.")
        else:
            st.success(f"✅ Found {len(fundraising_df)} fundraising link shares from {fundraising_df['account_id'].nunique()} accounts.")

            # Summarize campaigns
            campaign_summary = fundraising_df.groupby('clean_url').agg(
                Num_Posts=('content_id', 'size'),
                Num_Unique_Accounts=('account_id', 'nunique'),
                Top_Accounts=('account_id', lambda x: ', '.join(x.value_counts().head(3).index)),
                First_Shared=('timestamp_share', 'min'),
                Last_Shared=('timestamp_share', 'max'),
                Campaign_Type=('is_fundraising', 'first')
            ).reset_index()

            campaign_summary['First_Shared'] = pd.to_datetime(campaign_summary['First_Shared'], unit='s', utc=True)
            campaign_summary['Last_Shared'] = pd.to_datetime(campaign_summary['Last_Shared'], unit='s', utc=True)

            # Add risk score
            campaign_summary['Risk_Score'] = (
                (campaign_summary['Num_Posts'] / campaign_summary['Num_Unique_Accounts']) *
                np.where(campaign_summary['Campaign_Type'] == 'Suspicious', 2.0, 1.0)
            )
            campaign_summary['Risk_Flag'] = np.where(
                campaign_summary['Risk_Score'] > campaign_summary['Risk_Score'].quantile(0.8), 'High',
                np.where(campaign_summary['Campaign_Type'] == 'Suspicious', 'Medium', 'Low')
            )

            st.markdown("### 🚩 Detected Fundraising Campaigns")
            st.dataframe(
                campaign_summary[[
                    'clean_url', 'Campaign_Type', 'Num_Posts', 'Num_Unique_Accounts',
                    'Top_Accounts', 'First_Shared', 'Last_Shared', 'Risk_Flag'
                ]].sort_values('Risk_Score', ascending=False),
                use_container_width=True
            )

            @st.cache_data
            def convert_df(_df):
                return _df.to_csv(index=False).encode('utf-8')

            csv = convert_df(campaign_summary)
            st.download_button(
                "📥 Download Fundraising Report",
                csv,
                "fundraising_campaigns.csv",
                "text/csv"
            )
# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🌐 Network of Coordinated Accounts")

    if 'max_nodes_to_display' not in st.session_state:
        st.session_state.max_nodes_to_display = 40

    st.session_state.max_nodes_to_display = st.slider(
        "Maximum Nodes to Display in Graph",
        min_value=10, max_value=200, value=st.session_state.max_nodes_to_display, step=10,
        help="Limit the graph to the top N most connected accounts."
    )

    if st.button("Build Network Graph") and not df_for_analysis.empty:
        with st.spinner("Building network..."):
            clustered = cached_clustering(df_for_analysis, eps, min_samples, max_features)
            G, pos, cluster_map = cached_network_graph(clustered)

        if not G.nodes():
            st.warning("No coordinated activity detected to build a network graph.")
        else:
            # Filter top nodes by degree
            node_degrees = dict(G.degree())
            top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:st.session_state.max_nodes_to_display]
            subgraph = G.subgraph(top_nodes)
            pos_filtered = {node: pos[node] for node in top_nodes}

            edge_x, edge_y = [], []
            for u, v in subgraph.edges():
                x0, y0 = pos_filtered[u]
                x1, y1 = pos_filtered[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

            node_x, node_y, node_text, node_color = [], [], [], []
            for node in subgraph.nodes():
                x, y = pos_filtered[node]
                node_x.append(x); node_y.append(y)
                deg = subgraph.degree(node)
                platform = subgraph.nodes[node].get('platform', 'Unknown')
                node_text.append(f"{node}<br>deg: {deg} • {platform}")
                node_color.append(cluster_map.get(node, -2))

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                marker=dict(
                    size=[subgraph.degree(n)*3 + 10 for n in subgraph.nodes()],
                    color=node_color, colorscale='Viridis', showscale=True
                )
            )

            fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="CIB Network", showlegend=False, hovermode='closest'))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🚨 Risk Assessment: Top Central Accounts")
            degree_centrality = nx.degree_centrality(subgraph)
            risk_df = pd.DataFrame(degree_centrality.items(), columns=['Account', 'Centrality']).sort_values('Centrality', ascending=False).head(20)
            risk_df['Risk Score'] = (risk_df['Centrality'] * 100).round(2)
            st.dataframe(risk_df, use_container_width=True)
    else:
        st.info("Upload data and click 'Build Network Graph' to begin.")
