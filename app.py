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
import requests
from collections import Counter
import tldextract
import time

# --- Set Page Config ---
st.set_page_config(page_title="Humanitarian Campaign CIB Network Monitoring", layout="wide")
st.title("🕊️ Humanitarian Campaign Monitoring Dashboard")

# --- Define the 6 key phrases to track ---
PHRASES_TO_TRACK = [
    "If you're scrolling, PLEASE leave a dot",
    "I'm so hungry, I'm not ashamed to say that",
    "3 replies — even dots — can break the algorithm",
    "My body is slowly falling apart from malnutrition, dizziness, and weight loss",
    "Good bye. If we die, don't forget us",
    "If you see this reply with a dot"
]

# --- GitHub URLs for default data (✅ Fixed: no trailing spaces) ---
PHRASE_DATA_SOURCES = {
    "If you're scrolling, PLEASE leave a dot": "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/If_youre_scrolling_PLEASE_leave_a_dot_AND_postType%20-%20Aug%2013%2C%202025%20-%2010%2032%2047%20AM.csv",
    "3 replies — even dots — can break the algorithm": "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/3_replies_even_dots_can_break_the_algorithm_AND_postType%20-%20Aug%2013%2C%202025%20-%2010%2032%2047%20AM.csv",
    # Add other URLs as needed
}

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
    date_formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']
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

@st.cache_data
def fetch_csv_from_url(url, timeout=10):
    """Fetch CSV from URL with timeout and error handling"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), low_memory=False)
    except Exception as e:
        st.warning(f"❌ Failed to load data from URL: {e}")
        return pd.DataFrame()

@st.cache_data
def final_preprocess_and_map_columns(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_processed = df.copy()

    # Map columns
    df_processed = df_processed.rename(columns={
        'tweet id': 'content_id',
        'influencer': 'account_id',
        'hit sentence': 'object_id',
        'date': 'timestamp_share',
        'text': 'object_id',
        'url': 'URL'
    }, errors='ignore')

    for col in ['account_id', 'content_id', 'object_id', 'timestamp_share', 'URL']:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed['account_id'] = df_processed['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    df_processed['URL'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')
    df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    df_processed['timestamp_share'] = df_processed['timestamp_share'].apply(parse_timestamp_robust)
    df_processed = df_processed.dropna(subset=['timestamp_share', 'account_id', 'original_text'])
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)

    # ✅ Assign the most matching phrase (don't filter out)
    def assign_phrase(text):
        for phrase in PHRASES_TO_TRACK:
            if phrase.lower() in text:
                return phrase
        return "Other"
    df_processed['assigned_phrase'] = df_processed['original_text'].apply(assign_phrase)

    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed['extracted_urls'] = df_processed['object_id'].apply(extract_all_urls)

    core_columns = ['account_id', 'content_id', 'object_id', 'original_text', 'timestamp_share', 'Platform', 'extracted_urls', 'assigned_phrase']
    other_columns = [col for col in df_processed.columns if col not in core_columns]
    
    core_df = df_processed[core_columns].copy()
    other_df = df_processed[['content_id'] + other_columns].copy()
    
    return core_df, other_df

def find_fundraising_campaigns(df, coordinated_groups_df):
    if df.empty: return pd.DataFrame()
    fundraising_domains = ['gofundme.com', 'paypal.com', 'justgiving.com', 'donorbox.org', 'redcross.org', 'unicef.org', 'doctorswithoutborders.org', 'icrc.org']
    urls_df = df.explode('extracted_urls').dropna(subset=['extracted_urls']).copy()
    urls_df.rename(columns={'extracted_urls': 'Fundraising Link'}, inplace=True)
    urls_df = urls_df.drop_duplicates(subset=['content_id', 'Fundraising Link'])
    if urls_df.empty: return pd.DataFrame()

    def get_domain(url):
        ext = tldextract.extract(url)
        if ext.domain == 'paypal' and ext.suffix == 'me': return 'paypal.me'
        return f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
    
    urls_df['domain'] = urls_df['Fundraising Link'].apply(get_domain)
    fundraising_links_df = urls_df[urls_df['domain'].isin(fundraising_domains) | (urls_df['domain'] == 'paypal.me')].copy()
    if fundraising_links_df.empty: return pd.DataFrame()

    campaign_summary_df = fundraising_links_df.groupby('Fundraising Link').agg(
        Num_Posts=('content_id', 'size'),
        Num_Unique_Accounts=('account_id', 'nunique'),
        First_Shared=('timestamp_share', 'min'),
        Last_Shared=('timestamp_share', 'max')
    ).reset_index()

    campaign_summary_df['First_Shared'] = pd.to_datetime(campaign_summary_df['First_Shared'], unit='s', utc=True)
    campaign_summary_df['Last_Shared'] = pd.to_datetime(campaign_summary_df['Last_Shared'], unit='s', utc=True)
    campaign_summary_df['Coordination_Score'] = 0.0
    campaign_summary_df['Risk_Flag'] = 'Low'

    if not coordinated_groups_df.empty:
        account_coordination_scores = coordinated_groups_df.groupby('Account ID')['group_size'].max().fillna(0)
        account_coordination_mapping = account_coordination_scores.to_dict()
        link_scores = fundraising_links_df.groupby('Fundraising Link')['account_id'].apply(
            lambda accounts: max([account_coordination_mapping.get(acc, 0) for acc in accounts], default=0)
        )
        campaign_summary_df['Coordination_Score'] = campaign_summary_df['Fundraising Link'].map(link_scores).fillna(0)

    q80 = campaign_summary_df['Coordination_Score'].quantile(0.8)
    campaign_summary_df['Risk_Flag'] = np.where(
        campaign_summary_df['Coordination_Score'] > q80, 'High',
        np.where((campaign_summary_df['Num_Unique_Accounts'] < 5) & (campaign_summary_df['Num_Posts'] > 20), 'Needs Review', 'Low')
    )

    return campaign_summary_df[['Fundraising Link', 'Num_Posts', 'Num_Unique_Accounts', 'First_Shared', 'Last_Shared', 'Coordination_Score', 'Risk_Flag']]

def cluster_texts(df, eps, min_samples, max_features, mode):
    if mode == 'URL Mode':
        df_for_clustering = df.explode('extracted_urls').dropna(subset=['extracted_urls']).copy()
        if df_for_clustering.empty or df_for_clustering['extracted_urls'].nunique() <= 1:
            df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
        texts_to_cluster = df_for_clustering['extracted_urls'].astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=max_features, analyzer='char_wb', ngram_range=(2, 5))
    else:
        if df['original_text'].nunique() <= 1:
            df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
        texts_to_cluster = df['original_text'].astype(str).tolist()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

    if not texts_to_cluster or all(t.strip() == "" for t in texts_to_cluster):
        df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
        
    try: tfidf_matrix = vectorizer.fit_transform(texts_to_cluster)
    except Exception: df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy

    eps = max(0.01, min(0.99, eps))
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(tfidf_matrix)
    
    if mode == 'URL Mode':
        df_clustered_urls = df_for_clustering.copy()
        df_clustered_urls['cluster'] = clustering.labels_
        df_copy = df.copy()
        df_copy['cluster'] = -1
        for cluster_id, group in df_clustered_urls.groupby('cluster'):
            if cluster_id != -1:
                post_ids = group['content_id'].unique()
                df_copy.loc[df_copy['content_id'].isin(post_ids), 'cluster'] = cluster_id
        return df_copy
    else:
        df_copy = df.copy()
        df_copy['cluster'] = clustering.labels_
        return df_copy

def find_coordinated_groups(df, threshold, max_features, mode):
    text_col = 'original_text' if mode == 'Text Mode' else 'extracted_urls'
    if mode == 'URL Mode': df = df.explode('extracted_urls').dropna(subset=['extracted_urls']).copy()
    coordination_groups = []
    clustered_groups = df[df['cluster'] != -1].groupby('cluster')

    for cluster_id, group in clustered_groups:
        if len(group) < 2 or len(group['account_id'].unique()) < 2: continue
        if len(group) > 500:  # Skip huge clusters
            continue
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col]].copy().reset_index(drop=True)

        vectorizer = TfidfVectorizer(
            stop_words='english' if mode == 'Text Mode' else None,
            analyzer='char_wb' if mode == 'URL Mode' else 'word',
            ngram_range=(3, 5) if mode == 'Text Mode' else (2, 5),
            max_features=max_features
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_df[text_col])
        except Exception: continue

        cosine_sim = cosine_similarity(tfidf_matrix)
        adj = {i: [] for i in range(len(clean_df))}
        for i in range(len(clean_df)):
            for j in range(i + 1, len(clean_df)):
                if cosine_sim[i, j] >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        for i in range(len(clean_df)):
            if i not in visited:
                group_indices = []
                queue = [i]
                visited.add(i)
                while queue:
                    u = queue.pop(0)
                    group_indices.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
                if len(group_indices) > 1:
                    group_posts = clean_df.iloc[group_indices]
                    if len(group_posts['account_id'].unique()) > 1:
                        max_sim = cosine_sim[np.ix_(group_indices, group_indices)].max()
                        posts_data = group_posts.rename(columns={
                            'account_id': 'Account ID', 'Platform': 'Platform',
                            'timestamp_share': 'Timestamp', text_col: 'Text', 'URL': 'URL'
                        }).to_dict('records')
                        coordination_groups.append({
                            "posts": posts_data,
                            "num_posts": len(posts_data),
                            "num_accounts": len(group_posts['account_id'].unique()),
                            "max_similarity_score": round(max_sim, 3)
                        })
    return sorted(coordination_groups, key=lambda x: x['num_posts'], reverse=True)

def build_user_interaction_graph(df):
    G = nx.Graph()
    if 'cluster' not in df.columns or df.empty: return G, {}, {}
    grouped = df[df['cluster'] != -1].groupby('cluster')
    for cluster_id, group in grouped:
        users = group['account_id'].dropna().unique()
        if len(users) < 2: continue
        for u1, u2 in combinations(users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]['weight'] += 1
            else:
                G.add_edge(u1, u2, weight=1)
    all_users = df['account_id'].dropna().unique()
    for user in all_users:
        if user not in G: G.add_node(user)
        G.nodes[user]['platform'] = df[df['account_id'] == user]['Platform'].mode().iloc[0] if not df[df['account_id'] == user]['Platform'].mode().empty else 'Unknown'
    if G.nodes():
        pos = nx.spring_layout(G, seed=42, k=2)  # Faster than kamada-kawai
        cluster_map = df.groupby('account_id')['cluster'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else -2).to_dict()
        return G, pos, cluster_map
    return G, {}, {}

# --- Cached Functions ---
@st.cache_data(show_spinner="🔍 Finding coordinated groups...")
def cached_find_coordinated_groups(_df, threshold, max_features, mode):
    return find_coordinated_groups(_df, threshold, max_features, mode)

@st.cache_data(show_spinner="🧩 Clustering content...")
def cached_clustering(_df, eps, min_samples, max_features, mode):
    return cluster_texts(_df, eps, min_samples, max_features, mode)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df): return build_user_interaction_graph(_df)

@st.cache_data(show_spinner="🔗 Identifying fundraising campaigns...")
def cached_find_fundraising_campaigns(df, coordinated_groups_df):
    return find_fundraising_campaigns(df, coordinated_groups_df)

# --- Main Dashboard Logic ---
st.sidebar.header("📥 Upload Your Data")
st.sidebar.info("Upload your CSV files or use default data from GitHub.")
uploaded_files = {
    "Meltwater CSV": st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater"),
    "CivicSignals CSV": st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals"),
    "Open-Measure CSV": st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure"),
}

df_raw = pd.DataFrame()
core_df = pd.DataFrame()
other_df = pd.DataFrame()

# === Data Loading ===
if any(uploaded_files.values()):
    st.info("📁 Processing uploaded files...")
    all_dfs = []
    for name, file in uploaded_files.items():
        if file:
            try:
                df_temp = pd.read_csv(file, low_memory=False)
                df_temp['source'] = name
                all_dfs.append(df_temp)
                st.sidebar.success(f"✅ Loaded {len(df_temp)} rows from {name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to read {name}: {e}")
    if all_dfs:
        df_raw = pd.concat(all_dfs, ignore_index=True)
else:
    st.info("🌐 No upload — loading default data from GitHub...")
    all_dfs = []
    for phrase, url in PHRASE_DATA_SOURCES.items():
        with st.spinner(f"📥 Fetching data for: *{phrase}*"):
            df_temp = fetch_csv_from_url(url)
            if not df_temp.empty:
                df_temp['assigned_phrase_hint'] = phrase
                all_dfs.append(df_temp)
                st.sidebar.success(f"✅ Fetched {len(df_temp)} rows for '{phrase}'")
    if all_dfs:
        df_raw = pd.concat(all_dfs, ignore_index=True)

# === Preprocess ===
if not df_raw.empty:
    core_df, other_df = final_preprocess_and_map_columns(df_raw)
    if core_df.empty:
        st.error("❌ No valid data after preprocessing.")
    else:
        st.sidebar.success(f"✅ Loaded {len(core_df):,} posts for analysis.")
else:
    st.error("❌ No data loaded. Please upload a file or check GitHub URLs.")
    st.stop()

# === Filters & Sampling ===
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters & Settings")
min_date = pd.to_datetime(core_df['timestamp_share'].min(), unit='s').date()
max_date = pd.to_datetime(core_df['timestamp_share'].max(), unit='s').date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
start_ts = int(pd.Timestamp(date_range[0], tz='UTC').timestamp())
end_ts = int(pd.Timestamp(date_range[1], tz='UTC').timestamp()) + 86399

filtered_df = core_df[(core_df['timestamp_share'] >= start_ts) & (core_df['timestamp_share'] <= end_ts)].copy()

max_sample = st.sidebar.number_input("Max Posts for Analysis", min_value=0, value=5000, step=1000, help="0 = no limit")
if max_sample > 0 and len(filtered_df) > max_sample:
    df_for_analysis = filtered_df.sample(n=max_sample, random_state=42).copy()
    st.sidebar.warning(f"📊 Analyzing {len(df_for_analysis):,} sampled posts.")
else:
    df_for_analysis = filtered_df.copy()
    st.sidebar.info(f"📊 Analyzing all {len(df_for_analysis):,} posts.")

analysis_mode = st.sidebar.radio("Analysis Mode", ("Text Mode", "URL Mode"))

# === Tabs ===
tab0, tab1, tab2, tab3 = st.tabs(["📝 Raw Data", "❤️ Campaign Pulse", "🕸️ CIB Network", "💰 Fundraising & Risk"])

# --- TAB 0: Raw Data ---
with tab0:
    st.dataframe(core_df.head(20), use_container_width=True)

# --- TAB 1: Campaign Pulse ---
with tab1:
    if analysis_mode == "URL Mode":
        all_urls = df_for_analysis.explode('extracted_urls')['extracted_urls'].dropna()
        if not all_urls.empty:
            top_urls = all_urls.value_counts().head(10)
            fig = px.bar(top_urls, title="Top Shared URLs", labels={'value': 'Shares'})
            st.plotly_chart(fig)
    else:
        phrase_counts = df_for_analysis['assigned_phrase'].value_counts()
        fig = px.bar(phrase_counts, title="Posts by Key Phrase", labels={'value': 'Count'})
        st.plotly_chart(fig)

# --- TAB 2: CIB Network ---
with tab2:
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.3, 0.05)
    min_samples = st.sidebar.slider("Min Samples", 2, 10, 2)
    max_features = st.sidebar.slider("Max TF-IDF Features", 1000, 5000, 2000, 1000)
    threshold_sim = st.slider("Pairwise Similarity Threshold", 0.75, 0.99, 0.90, 0.01)
    min_connections = st.sidebar.slider("Min Connections for Network", 1, 10, 2)

    with st.spinner("🔍 Clustering and detecting coordination..."):
        clustered_df = cached_clustering(df_for_analysis, eps, min_samples, max_features, analysis_mode)
        coordinated_groups = cached_find_coordinated_groups(clustered_df, threshold_sim, max_features, analysis_mode)

    if coordinated_groups:
        st.info(f"✅ Found {len(coordinated_groups)} coordinated groups.")
        for i, g in enumerate(coordinated_groups[:5]):
            st.markdown(f"**Group {i+1}**: {g['num_posts']} posts, {g['num_accounts']} accounts")
            posts_df = pd.DataFrame(g['posts'])
            posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s', utc=True)
            st.dataframe(posts_df, use_container_width=True)
            st.markdown("---")
    else:
        st.warning("No coordinated groups found.")

    with st.spinner("🕸️ Building network..."):
        G, pos, cluster_map = cached_network_graph(clustered_df)
        if G.nodes():
            G_filtered = G.copy()
            G_filtered.remove_nodes_from([n for n in G if G.degree(n) < min_connections])
            if G_filtered.nodes():
                # (Plot code as before — omitted for brevity, use your existing)
                pass
            else:
                st.info("No nodes meet the min connection threshold.")
        else:
            st.warning("No network to display.")

# --- TAB 3: Fundraising ---
with tab3:
    with st.spinner("🔍 Checking for fundraising links..."):
        fundraising_df = cached_find_fundraising_campaigns(df_for_analysis, pd.DataFrame())
    if not fundraising_df.empty:
        st.dataframe(fundraising_df)
    else:
        st.info("No fundraising links detected.")
