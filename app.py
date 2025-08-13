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
import requests
from collections import Counter
import tldextract

# --- Set Page Config ---
st.set_page_config(page_title="Humanitarian Campaign Monitor", layout="wide")
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

# --- GitHub URLs for default data (✅ NO TRAILING SPACES) ---
PHRASE_DATA_SOURCES = {
    "If you're scrolling, PLEASE leave a dot": "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/If_youre_scrolling_PLEASE_leave_a_dot_AND_postType%20-%20Aug%2013%2C%202025%20-%2010%2032%2047%20AM.csv",
    "3 replies — even dots — can break the algorithm": "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/3_replies_even_dots_can_break_the_algorithm_AND_postType%20-%20Aug%2013%2C%202025%20-%2010%2032%2047%20AM.csv",
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

# --- Combine Multiple Datasets ---
def combine_social_media_data(
    meltwater_df=None,
    civicsignals_df=None,
    openmeasure_df=None,
    meltwater_object_col='hit sentence',
    civicsignals_object_col='title',
    openmeasure_object_col='text'
):
    combined_dfs = []
    def get_col(df, col):
        return df[col] if col in df.columns else pd.Series([np.nan]*len(df), index=df.index)

    for df, name, obj_col, id_col, ts_col, url_col in [
        (meltwater_df, 'Meltwater', meltwater_object_col, 'tweet id', 'date', 'url'),
        (civicsignals_df, 'CivicSignals', civicsignals_object_col, 'stories_id', 'publish_date', 'url'),
        (openmeasure_df, 'OpenMeasure', openmeasure_object_col, 'id', 'created_at', 'url')
    ]:
        if df is not None and not df.empty:
            df = df.copy()
            df.columns = df.columns.str.lower()
            new_df = pd.DataFrame()
            new_df['account_id'] = get_col(df, 'influencer' if name == 'Meltwater' else 'media_name' if name == 'CivicSignals' else 'actor_username')
            new_df['content_id'] = get_col(df, id_col)
            new_df['object_id'] = get_col(df, obj_col)
            new_df['URL'] = get_col(df, url_col)
            new_df['timestamp_share'] = get_col(df, ts_col)
            new_df['source_dataset'] = name
            combined_dfs.append(new_df)

    if not combined_dfs:
        return pd.DataFrame()
    combined = pd.concat(combined_dfs, ignore_index=True)
    combined = combined.dropna(subset=['account_id', 'content_id', 'timestamp_share', 'object_id'])
    combined['account_id'] = combined['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    combined['content_id'] = combined['content_id'].astype(str).replace('nan', '').fillna('')
    combined['URL'] = combined['URL'].astype(str).replace('nan', '').fillna('')
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')
    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined = combined[combined['object_id'].str.strip() != ""]
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
    
    # Limit nodes by degree
    node_degrees = dict(G.degree())
    top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:st.session_state.max_nodes_to_display]
    subgraph = G.subgraph(top_nodes)
    pos = nx.kamada_kawai_layout(subgraph)
    cluster_map = df.groupby('account_id')['cluster'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else -2).to_dict()
    return subgraph, pos, cluster_map

def find_fundraising_campaigns(df, _):
    if df.empty: return pd.DataFrame()
    domains = ['gofundme.com', 'paypal.com', 'paypal.me', 'unicef.org', 'icrc.org']
    urls = df.explode('extracted_urls').dropna(subset=['extracted_urls'])
    urls['domain'] = urls['extracted_urls'].apply(lambda x: tldextract.extract(x).registered_domain)
    filtered = urls[urls['domain'].isin(domains)]
    if filtered.empty: return pd.DataFrame()
    summary = filtered.groupby('extracted_urls').agg(
        Num_Posts=('content_id', 'size'),
        Num_Unique_Accounts=('account_id', 'nunique'),
        First_Shared=('timestamp_share', 'min'),
        Last_Shared=('timestamp_share', 'max')
    ).reset_index()
    summary.rename(columns={'extracted_urls': 'Fundraising Link'}, inplace=True)
    summary['First_Shared'] = pd.to_datetime(summary['First_Shared'], unit='s', utc=True)
    summary['Last_Shared'] = pd.to_datetime(summary['Last_Shared'], unit='s', utc=True)
    summary['Coordination_Score'] = 0.0
    summary['Risk_Flag'] = 'Low'
    return summary[['Fundraising Link', 'Num_Posts', 'Num_Unique_Accounts', 'First_Shared', 'Last_Shared', 'Coordination_Score', 'Risk_Flag']]

# --- Caching ---
@st.cache_data
def cached_clustering(_df, eps, min_samples, max_features): return cluster_texts(_df, eps, min_samples, max_features)
@st.cache_data
def cached_find_coordinated_groups(_df, threshold, max_features): return find_coordinated_groups(_df, threshold, max_features)
@st.cache_data
def cached_network_graph(_df): return build_user_interaction_graph(_df)
@st.cache_data
def cached_find_fundraising_campaigns(df, _): return find_fundraising_campaigns(df, _)

# --- Data Loading ---
def fetch_csv_from_url(url):
    try:
        r = requests.get(url.strip(), timeout=10)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception as e:
        st.warning(f"❌ Failed to load {url}: {e}")
        return pd.DataFrame()

st.sidebar.header("📥 Data Source")
data_source = st.sidebar.radio("Choose data source:", ("Use Default Datasets", "Upload CSV Files"))

# Clear cache on change
if 'last_data_source' not in st.session_state or st.session_state.last_data_source != data_source:
    st.cache_data.clear()
    st.session_state.last_data_source = data_source

if 'max_nodes_to_display' not in st.session_state:
    st.session_state.max_nodes_to_display = 40

combined_raw_df = pd.DataFrame()

if data_source == "Use Default Datasets":
    st.sidebar.info("Using default datasets from GitHub.")
    with st.spinner("📥 Loading default data..."):
        dfs = []
        for phrase, url in PHRASE_DATA_SOURCES.items():
            df_temp = fetch_csv_from_url(url)
            if not df_temp.empty:
                df_temp['assigned_phrase_hint'] = phrase
                dfs.append(df_temp)
                st.sidebar.success(f"✅ Loaded {len(df_temp)} rows for '{phrase}'")
        if dfs:
            combined_raw_df = combine_social_media_data(*dfs)
    st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts.")

elif data_source == "Upload CSV Files":
    uploaded_files = {
        "Meltwater": st.sidebar.file_uploader("Upload Meltwater CSV", type="csv"),
        "CivicSignals": st.sidebar.file_uploader("Upload CivicSignals CSV", type="csv"),
        "Open-Measure": st.sidebar.file_uploader("Upload Open-Measure CSV", type="csv"),
    }
    dfs = []
    for name, file in uploaded_files.items():
        if file:
            try:
                df = pd.read_csv(file)
                st.sidebar.success(f"✅ {name}: {len(df)} rows")
                dfs.append(df)
            except Exception as e:
                st.sidebar.error(f"❌ {name}: {e}")
    if dfs:
        combined_raw_df = combine_social_media_data(*dfs)

# --- Preprocess ---
if combined_raw_df.empty:
    st.warning("❌ No data loaded. Please upload a file or check URLs.")
    st.stop()

df = final_preprocess_and_map_columns(combined_raw_df)
if df.empty:
    st.warning("❌ No valid data after preprocessing.")
    st.stop()

# --- Filters ---
st.sidebar.header("🔍 Filters")
min_date = pd.to_datetime(df['timestamp_share'].min(), unit='s').date()
max_date = pd.to_datetime(df['timestamp_share'].max(), unit='s').date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
start_ts = int(pd.Timestamp(date_range[0], tz='UTC').timestamp())
end_ts = int(pd.Timestamp(date_range[1], tz='UTC').timestamp()) + 86399
filtered_df = df[(df['timestamp_share'] >= start_ts) & (df['timestamp_share'] <= end_ts)].copy()

max_sample = st.sidebar.number_input("Max Posts for Analysis", 0, 100000, 5000)
df_for_analysis = filtered_df.sample(n=min(max_sample, len(filtered_df)), random_state=42) if max_sample and len(filtered_df) > max_sample else filtered_df.copy()

# --- Tabs ---
tab0, tab1, tab2, tab3 = st.tabs(["📝 Raw Data", "📊 Pulse", "🕸️ CIB", "💰 Fundraising"])

# ==================== TAB 0: Raw Data ====================
with tab0:
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)
    with st.expander("🔍 Full Columns"):
        st.dataframe(df, use_container_width=True)

# ==================== TAB 1: Campaign Pulse ====================
with tab1:
    st.subheader("📊 Campaign Pulse")
    phrase_counts = df['assigned_phrase'].value_counts()
    fig_phrase = px.bar(phrase_counts, title="Posts by Key Phrase", labels={'value': 'Count', 'index': 'Phrase'})
    st.plotly_chart(fig_phrase, use_container_width=True)

    if 'Platform' in df.columns:
        platform_counts = df['Platform'].value_counts()
        fig_platform = px.bar(platform_counts, title="Posts by Platform", labels={'value': 'Count', 'index': 'Platform'})
        st.plotly_chart(fig_platform, use_container_width=True)

    # Time series
    plot_df = df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True).dt.date
    time_series = plot_df.groupby('date').size()
    fig_ts = px.line(time_series, title="Daily Post Volume", labels={'value': 'Posts', 'index': 'Date'})
    st.plotly_chart(fig_ts, use_container_width=True)

# ==================== TAB 2: CIB Network ====================
with tab2:
    st.subheader("🕸️ Coordinated Inauthentic Behavior (CIB) Detection")
    st.markdown("Detects coordinated amplification using text similarity and network analysis.")

    # Sidebar controls
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.3, 0.05)
    min_samples = st.sidebar.slider("Min Samples", 2, 10, 2, 1)
    max_features = st.sidebar.slider("Max TF-IDF Features", 1000, 5000, 2000, 500)
    threshold = st.slider("Pairwise Similarity Threshold", 0.8, 0.99, 0.9, 0.01)
    st.session_state.max_nodes_to_display = st.sidebar.slider("Max Nodes in Graph", 10, 200, st.session_state.max_nodes_to_display)

    if st.button("🔍 Run CIB Detection"):
        with st.spinner("🧩 Clustering posts..."):
            clustered = cached_clustering(df_for_analysis, eps, min_samples, max_features)
        st.success(f"✅ Clustering completed. Found {len(clustered[clustered['cluster'] != -1])} posts in clusters.")

        with st.spinner("🕵️ Finding coordinated groups..."):
            groups = cached_find_coordinated_groups(clustered, threshold, max_features)
        if groups:
            st.info(f"✅ Found **{len(groups)}** coordinated groups.")
            for i, g in enumerate(groups[:10]):
                st.markdown(f"**Group {i+1}** | Posts: `{g['num_posts']}` | Accounts: `{g['num_accounts']}` | Max Sim: `{g['max_similarity_score']}`")
                posts_df = pd.DataFrame(g['posts'])
                posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s', utc=True)
                st.dataframe(posts_df, use_container_width=True)
                st.markdown("---")
        else:
            st.warning("No coordinated groups found above threshold.")

        # Network Graph
        with st.spinner("🌐 Building network graph..."):
            G, pos, cluster_map = cached_network_graph(clustered)
        if not G.nodes():
            st.warning("No network to display.")
        else:
            # Build edge trace
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

            # Build node trace
            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x); node_y.append(y)
                deg = G.degree(node)
                platform = G.nodes[node].get('platform', 'Unknown')
                cluster_id = cluster_map.get(node, -2)
                node_text.append(f"User: {node}<br>Platform: {platform}<br>Degree: {deg}<br>Cluster: {cluster_id}")
                node_color.append(cluster_id)
                node_size.append(deg * 5 + 10)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cluster"),
                    line_width=2,
                    opacity=0.8
                )
            )

            fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                title=f"Network of Coordinated Accounts (Top {st.session_state.max_nodes_to_display} Nodes)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700
            ))
            st.plotly_chart(fig_net, use_container_width=True)

# ==================== TAB 3: Fundraising & Risk ====================
with tab3:
    st.subheader("💰 Fundraising Links & Risk Detection")
    st.markdown("Scans posts for links to known humanitarian fundraising platforms.")

    if st.button("🔍 Scan for Fundraising Campaigns"):
        with st.spinner("🔎 Analyzing outbound links..."):
            funds = cached_find_fundraising_campaigns(df_for_analysis, pd.DataFrame())
        if not funds.empty:
            st.success(f"✅ Found **{len(funds)}** fundraising campaigns.")
            st.dataframe(funds, use_container_width=True)

            st.markdown("### Risk Flags")
            st.markdown("""
            - **High**: Linked to coordinated network
            - **Needs Review**: Few accounts, many posts
            - **Low**: Organic sharing
            """)
        else:
            st.info("No fundraising links detected in the current dataset.")

    # Optional: Show top shared URLs
    with st.expander("🔗 Top Shared URLs"):
        all_urls = df.explode('extracted_urls')['extracted_urls'].dropna()
        if not all_urls.empty:
            top_urls = all_urls.value_counts().head(10)
            fig_urls = px.bar(top_urls, title="Top 10 Shared URLs", labels={'value': 'Shares'})
            st.plotly_chart(fig_urls)
        else:
            st.info("No URLs extracted.")
