# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from itertools import combinations
import re
from io import StringIO
import requests
import tldextract
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# Page & Title
# ---------------------------------------------------------
st.set_page_config(page_title="Humanitarian Campaign CIB Network Monitoring", layout="wide")
st.title("🕊️ Humanitarian Campaign Monitoring Dashboard")

# ---------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------
PHRASES_TO_TRACK = [
    "If you're scrolling, PLEASE leave a dot",
    "I'm so hungry, I'm not ashamed to say that",
    "3 replies — even dots — can break the algorithm",
    "My body is slowly falling apart from malnutrition, dizziness, and weight loss",
    "Good bye. If we die, don't forget us",
    "If you see this reply with a dot"
]

PHRASE_DATA_SOURCES = {
    "If you're scrolling, PLEASE leave a dot": "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/If_youre_scrolling_PLEASE_leave_a_dot_AND_postType%20-%20Aug%2013%2C%202025%20-%2010%2032%2047%20AM.csv",
    "3 replies — even dots — can break the algorithm": "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/3_replies_even_dots_can_break_the_algorithm_AND_postType%20-%20Aug%2013%2C%202025%20-%2010%2032%2047%20AM.csv",
    # Add more if needed
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def infer_platform_from_url(url: str) -> str:
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    u = url.lower()
    if "tiktok.com" in u: return "TikTok"
    if "facebook.com" in u or "fb.watch" in u: return "Facebook"
    if "twitter.com" in u or "x.com" in u: return "X"
    if "youtube.com" in u or "youtu.be" in u: return "YouTube"
    if "instagram.com" in u: return "Instagram"
    if "telegram.me" in u or "t.me" in u: return "Telegram"
    return "Media"

def extract_original_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str): return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

def parse_timestamp_robust(ts):
    try:
        parsed = pd.to_datetime(ts, errors='coerce', utc=True)
        if pd.notna(parsed):
            return int(parsed.timestamp())
    except Exception:
        return None
    return None

def extract_all_urls(text: str):
    if pd.isna(text) or not isinstance(text, str): return []
    return re.findall(r'https?://\S+', text)

# ---------------------------------------------------------
# Data Loading (Parallel GitHub fetch)
# ---------------------------------------------------------
def fetch_csv_from_url(url: str, timeout=8) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text), low_memory=False)
    except Exception as e:
        st.warning(f"❌ Failed to load {url}: {e}")
        return pd.DataFrame()

def parallel_fetch_csvs(sources: dict) -> list[pd.DataFrame]:
    dfs = []
    with ThreadPoolExecutor(max_workers=min(8, len(sources) or 1)) as ex:
        fut = {ex.submit(fetch_csv_from_url, url): phrase for phrase, url in sources.items()}
        for f in as_completed(fut):
            phrase = fut[f]
            df = f.result()
            if not df.empty:
                df["assigned_phrase_hint"] = phrase
                dfs.append(df)
                st.sidebar.success(f"✅ Fetched {len(df):,} rows for '{phrase}'")
    return dfs

# ---------------------------------------------------------
# Preprocess & Column Mapping
# ---------------------------------------------------------
def map_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Map columns. Note: keep URL only for platform inference & other_df; not in core columns.
    df = df.rename(columns={
        'tweet id': 'content_id',
        'influencer': 'account_id',
        'hit sentence': 'object_id',
        'date': 'timestamp_share',
        'text': 'object_id',
        'url': 'URL'
    }, errors='ignore')

    # Ensure expected cols exist
    for col in ['account_id', 'content_id', 'object_id', 'timestamp_share']:
        if col not in df.columns:
            df[col] = np.nan

    # Clean & parse
    df['object_id'] = df['object_id'].astype(str).replace('nan', '').fillna('')
    df['account_id'] = df['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    if 'URL' not in df.columns:  # optional URL column
        df['URL'] = ""

    df['original_text'] = df['object_id'].apply(extract_original_text)
    df['timestamp_share'] = df['timestamp_share'].apply(parse_timestamp_robust)

    df = df.dropna(subset=['timestamp_share', 'account_id', 'original_text'])
    df = df[df['original_text'].str.strip() != ""].reset_index(drop=True)

    # Assign phrases (non-filtering)
    def assign_phrase(text):
        for p in PHRASES_TO_TRACK:
            if p.lower() in text:
                return p
        return "Other"
    df['assigned_phrase'] = df['original_text'].apply(assign_phrase)

    # Platform detection (from URL if present; if no URL, leave Unknown/Media)
    df['Platform'] = df['URL'].apply(infer_platform_from_url)
    # Extract outbound links from the text
    df['extracted_urls'] = df['object_id'].apply(extract_all_urls)
    return df

def split_core_other(df: pd.DataFrame):
    # Per your note: URL must NOT be in core columns
    core_columns = [
        'account_id', 'content_id', 'object_id', 'original_text',
        'timestamp_share', 'Platform', 'extracted_urls', 'assigned_phrase'
    ]
    for c in core_columns:
        if c not in df.columns:
            df[c] = np.nan
    core_df = df[core_columns].copy()
    # Other columns include everything else (including URL)
    other_columns = [c for c in df.columns if c not in core_columns]
    if 'content_id' not in other_columns:
        other_columns = ['content_id'] + other_columns
    other_df = df[['content_id'] + [c for c in other_columns if c != 'content_id']].copy()
    return core_df, other_df

# ---------------------------------------------------------
# Caching for heavy functions
# ---------------------------------------------------------
@st.cache_data(show_spinner="🧩 Clustering content...", ttl=3600)
def cached_clustering(_df: pd.DataFrame, eps: float, min_samples: int, max_features: int, mode: str) -> pd.DataFrame:
    return cluster_texts(_df, eps, min_samples, max_features, mode)

@st.cache_data(show_spinner="🔍 Finding coordinated groups...", ttl=3600)
def cached_find_coordinated_groups(_df: pd.DataFrame, threshold: float, max_features: int, mode: str):
    return find_coordinated_groups(_df, threshold, max_features, mode)

@st.cache_data(show_spinner="🕸️ Building network graph...", ttl=3600)
def cached_network_graph(_df: pd.DataFrame):
    return build_user_interaction_graph(_df)

@st.cache_data(show_spinner="🔗 Identifying fundraising campaigns...", ttl=3600)
def cached_find_fundraising_campaigns(df: pd.DataFrame, coordinated_groups_df: pd.DataFrame):
    return find_fundraising_campaigns(df, coordinated_groups_df)

# ---------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------
def cluster_texts(df: pd.DataFrame, eps: float, min_samples: int, max_features: int, mode: str) -> pd.DataFrame:
    df = df.copy()
    if mode == 'URL Mode':
        df_urls = df.explode('extracted_urls').dropna(subset=['extracted_urls'])
        if df_urls.empty or df_urls['extracted_urls'].nunique() <= 1:
            df['cluster'] = -1
            return df
        texts = df_urls['extracted_urls'].astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=max_features, analyzer='char_wb', ngram_range=(2, 5))
        tfidf_matrix = vectorizer.fit_transform(texts)
        clustering = DBSCAN(metric='cosine', eps=max(0.01, min(0.99, eps)), min_samples=min_samples).fit(tfidf_matrix)

        df_urls['cluster'] = clustering.labels_
        df['cluster'] = -1
        for cid, group in df_urls.groupby('cluster'):
            if cid == -1: 
                continue
            post_ids = group['content_id'].unique()
            df.loc[df['content_id'].isin(post_ids), 'cluster'] = cid
        return df
    else:
        if df['original_text'].nunique() <= 1:
            df['cluster'] = -1
            return df
        texts = df['original_text'].astype(str).tolist()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(texts)
        clustering = DBSCAN(metric='cosine', eps=max(0.01, min(0.99, eps)), min_samples=min_samples).fit(tfidf_matrix)
        df['cluster'] = clustering.labels_
        return df

def find_coordinated_groups(df: pd.DataFrame, threshold: float, max_features: int, mode: str):
    text_col = 'original_text' if mode == 'Text Mode' else 'extracted_urls'
    working = df.copy()
    if mode == 'URL Mode':
        working = working.explode('extracted_urls').dropna(subset=['extracted_urls'])
        if working.empty:
            return []

    coordination_groups = []
    clustered = working[working['cluster'] != -1].groupby('cluster', sort=False)

    for cluster_id, group in clustered:
        if len(group) < 2 or len(group['account_id'].unique()) < 2:
            continue
        if len(group) > 500:  # skip huge clusters for speed
            continue

        clean_df = group[['account_id', 'timestamp_share', 'Platform', text_col]].copy().reset_index(drop=True)

        vectorizer = TfidfVectorizer(
            stop_words='english' if mode == 'Text Mode' else None,
            analyzer='char_wb' if mode == 'URL Mode' else 'word',
            ngram_range=(3, 5) if mode == 'Text Mode' else (2, 5),
            max_features=max_features
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_df[text_col].astype(str).tolist())
        except Exception:
            continue

        cosine_sim = cosine_similarity(tfidf_matrix)
        # Build adjacency by threshold
        adj = {i: [] for i in range(len(clean_df))}
        for i in range(len(clean_df)):
            for j in range(i + 1, len(clean_df)):
                if cosine_sim[i, j] >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        for i in range(len(clean_df)):
            if i in visited:
                continue
            queue = [i]
            visited.add(i)
            group_indices = []
            while queue:
                u = queue.pop(0)
                group_indices.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            if len(group_indices) > 1 and clean_df.iloc[group_indices]['account_id'].nunique() > 1:
                sub = cosine_sim[np.ix_(group_indices, group_indices)].copy()
                np.fill_diagonal(sub, 0.0)
                max_sim = float(sub.max()) if sub.size else 0.0
                posts_data = clean_df.rename(columns={
                    'account_id': 'Account ID',
                    'timestamp_share': 'Timestamp',
                    text_col: 'Text'
                }).iloc[group_indices]
                posts_records = posts_data.to_dict('records')
                coordination_groups.append({
                    "posts": posts_records,
                    "num_posts": len(posts_records),
                    "num_accounts": posts_data['Account ID'].nunique(),
                    "max_similarity_score": round(max_sim, 3),
                    "cluster_id": int(cluster_id)
                })

    # Sort by size then score
    coordination_groups.sort(key=lambda x: (x['num_posts'], x['max_similarity_score']), reverse=True)
    return coordination_groups

def build_user_interaction_graph(df: pd.DataFrame):
    G = nx.Graph()
    if 'cluster' not in df.columns or df.empty:
        return G, {}, {}

    # Add edges between accounts that co-appear in a cluster
    for cluster_id, group in df[df['cluster'] != -1].groupby('cluster', sort=False):
        users = group['account_id'].dropna().unique()
        if len(users) < 2:
            continue
        for u1, u2 in combinations(users, 2):
            if G.has_edge(u1, u2):
                G[u1][u2]['weight'] += 1
            else:
                G.add_edge(u1, u2, weight=1)

    # Ensure singletons are present
    for user in df['account_id'].dropna().unique():
        if user not in G:
            G.add_node(user)

    # Node attributes
    for user in G.nodes():
        sub = df[df['account_id'] == user]
        platform = sub['Platform'].mode().iloc[0] if not sub['Platform'].mode().empty else 'Unknown'
        G.nodes[user]['platform'] = platform

    if len(G.nodes()) == 0:
        return G, {}, {}

    # Positions (spring is faster than KK for larger graphs)
    pos = nx.spring_layout(G, seed=42, k=2)
    # Cluster map by account (most frequent cluster label, excluding -1)
    def mode_excluding_minus1(x):
        s = x[x != -1]
        return s.mode().iloc[0] if not s.mode().empty else -2
    cluster_map = df.groupby('account_id')['cluster'].apply(mode_excluding_minus1).to_dict()
    return G, pos, cluster_map

def find_fundraising_campaigns(df: pd.DataFrame, coordinated_groups_df: pd.DataFrame) -> pd.DataFrame:
    # Look for fundraising domains in outbound URLs found in text
    if df.empty:
        return pd.DataFrame()
    fundraising_domains = {
        'gofundme.com', 'paypal.com', 'justgiving.com', 'donorbox.org',
        'redcross.org', 'unicef.org', 'doctorswithoutborders.org', 'icrc.org', 'paypal.me'
    }
    urls_df = df.explode('extracted_urls').dropna(subset=['extracted_urls']).copy()
    if urls_df.empty:
        return pd.DataFrame()
    urls_df.rename(columns={'extracted_urls': 'Fundraising Link'}, inplace=True)
    urls_df = urls_df.drop_duplicates(subset=['content_id', 'Fundraising Link'])

    def get_domain(url):
        try:
            ext = tldextract.extract(url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
            if domain == "paypal.me":
                return "paypal.me"
            return domain
        except Exception:
            return ""

    urls_df['domain'] = urls_df['Fundraising Link'].apply(get_domain)
    fundraising_links_df = urls_df[urls_df['domain'].isin(fundraising_domains)].copy()
    if fundraising_links_df.empty:
        return pd.DataFrame()

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

    # If you want to use coordinated group info to score campaigns, you can pass a df here.
    if not coordinated_groups_df.empty and 'Account ID' in coordinated_groups_df.columns:
        # Example: score link by max group size of participating accounts
        account_group_score = coordinated_groups_df.groupby('Account ID')['group_size'].max().fillna(0).to_dict()
        link_scores = fundraising_links_df.groupby('Fundraising Link')['account_id'].apply(
            lambda accounts: max([account_group_score.get(acc, 0) for acc in accounts], default=0)
        )
        campaign_summary_df['Coordination_Score'] = campaign_summary_df['Fundraising Link'].map(link_scores).fillna(0)

    q80 = campaign_summary_df['Coordination_Score'].quantile(0.8)
    campaign_summary_df['Risk_Flag'] = np.where(
        campaign_summary_df['Coordination_Score'] > q80, 'High',
        np.where((campaign_summary_df['Num_Unique_Accounts'] < 5) & (campaign_summary_df['Num_Posts'] > 20), 'Needs Review', 'Low')
    )

    return campaign_summary_df[['Fundraising Link', 'Num_Posts', 'Num_Unique_Accounts',
                                'First_Shared', 'Last_Shared', 'Coordination_Score', 'Risk_Flag']]

# ---------------------------------------------------------
# Sidebar: Upload & Controls
# ---------------------------------------------------------
st.sidebar.header("📥 Upload Your Data")
uploaded_files = {
    "Meltwater CSV": st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater"),
    "CivicSignals CSV": st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals"),
    "Open-Measure CSV": st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure"),
}

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
df_raw = pd.DataFrame()
if any(uploaded_files.values()):
    st.info("📁 Processing uploaded files...")
    all_dfs = []
    for name, file in uploaded_files.items():
        if file:
            try:
                df_temp = pd.read_csv(file, low_memory=False)
                df_temp['source'] = name
                all_dfs.append(df_temp)
                st.sidebar.success(f"✅ Loaded {len(df_temp):,} rows from {name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to read {name}: {e}")
    if all_dfs:
        df_raw = pd.concat(all_dfs, ignore_index=True)
else:
    st.info("🌐 No upload — loading default data from GitHub in parallel...")
    t_load = time.time()
    dfs = parallel_fetch_csvs(PHRASE_DATA_SOURCES)
    st.write(f"⏱ Default fetch finished in {time.time() - t_load:.2f}s")
    if dfs:
        df_raw = pd.concat(dfs, ignore_index=True)

if df_raw.empty:
    st.error("❌ No data loaded. Please upload a CSV or verify the GitHub URLs.")
    st.stop()

# ---------------------------------------------------------
# Preprocess & Split
# ---------------------------------------------------------
t0 = time.time()
df_processed = map_and_prepare(df_raw)
core_df, other_df = split_core_other(df_processed)
st.sidebar.success(f"✅ {len(core_df):,} valid posts after preprocessing.")
st.caption(f"Preprocessing time: {time.time() - t0:.2f}s")

# ---------------------------------------------------------
# Filters & Sampling
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters & Settings")

if core_df.empty:
    st.error("❌ No valid rows after preprocessing.")
    st.stop()

min_ts = core_df['timestamp_share'].min()
max_ts = core_df['timestamp_share'].max()
min_date = pd.to_datetime(min_ts, unit='s').date()
max_date = pd.to_datetime(max_ts, unit='s').date()

date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
start_ts = int(pd.Timestamp(date_range[0], tz='UTC').timestamp())
end_ts = int(pd.Timestamp(date_range[1], tz='UTC').timestamp()) + 86399

filtered_df = core_df[(core_df['timestamp_share'] >= start_ts) & (core_df['timestamp_share'] <= end_ts)].copy()

max_sample = st.sidebar.number_input("Max Posts for Analysis", min_value=0, value=2000, step=500, help="0 = no limit")
if max_sample and len(filtered_df) > max_sample:
    df_for_analysis = filtered_df.sample(n=max_sample, random_state=42).copy()
    st.sidebar.warning(f"📊 Analyzing {len(df_for_analysis):,} sampled posts (of {len(filtered_df):,}).")
else:
    df_for_analysis = filtered_df.copy()
    st.sidebar.info(f"📊 Analyzing {len(df_for_analysis):,} posts.")

analysis_mode = st.sidebar.radio("Analysis Mode", ("Text Mode", "URL Mode"))

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab0, tab1, tab2, tab3 = st.tabs(["📝 Raw Data", "❤️ Campaign Pulse", "🕸️ CIB Network", "💰 Fundraising & Risk"])

# -------- TAB 0: Raw Data --------
with tab0:
    st.subheader("Raw (Core) Data Preview")
    st.dataframe(core_df.head(50), use_container_width=True)
    with st.expander("Other Columns (includes URL and any extras)"):
        st.dataframe(other_df.head(50), use_container_width=True)

# -------- TAB 1: Campaign Pulse --------
with tab1:
    st.subheader("Pulse Metrics")
    if analysis_mode == "URL Mode":
        urls = df_for_analysis.explode('extracted_urls')['extracted_urls'].dropna()
        if urls.empty:
            st.info("No extracted URLs found in the current filter.")
        else:
            top_urls = urls.value_counts().head(10)
            fig = px.bar(top_urls, title="Top Shared URLs", labels={'value': 'Shares', 'index': 'URL'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        phrase_counts = df_for_analysis['assigned_phrase'].value_counts()
        if phrase_counts.empty:
            st.info("No posts for the current filter.")
        else:
            fig = px.bar(phrase_counts, title="Posts by Key Phrase", labels={'value': 'Count', 'index': 'Phrase'})
            st.plotly_chart(fig, use_container_width=True)

# -------- TAB 2: CIB Network --------
with tab2:
    st.subheader("Coordinated Inauthentic Behavior (CIB) Network")

    eps = st.sidebar.slider("DBSCAN eps", 0.10, 1.00, 0.30, 0.05)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 2, 1)
    max_features = st.sidebar.slider("Max TF-IDF Features", 500, 10000, 2000, 500)
    threshold_sim = st.slider("Pairwise Similarity Threshold", 0.75, 0.99, 0.90, 0.01)
    min_connections = st.sidebar.slider("Min Connections for Network", 1, 10, 2, 1)

    st.info("⚡ Heavy steps run only when you click **Run CIB Analysis**.")
    run_cib = st.button("▶️ Run CIB Analysis")

    if run_cib:
        t_total = time.time()

        # Clustering
        t1 = time.time()
        clustered_df = cached_clustering(df_for_analysis, eps, min_samples, max_features, analysis_mode)
        st.success(f"✅ Clustering completed in {time.time() - t1:.2f}s")
        st.dataframe(clustered_df.head(20), use_container_width=True)

        # Coordinated groups
        t2 = time.time()
        coordinated_groups = cached_find_coordinated_groups(clustered_df, threshold_sim, max_features, analysis_mode)
        st.success(f"✅ Coordination detection completed in {time.time() - t2:.2f}s")

        if coordinated_groups:
            st.info(f"🔎 Found {len(coordinated_groups)} coordinated groups. Showing top 5:")
            for i, g in enumerate(coordinated_groups[:5], start=1):
                st.markdown(f"**Group {i}** — Posts: `{g['num_posts']}`, Accounts: `{g['num_accounts']}`, Max Sim: `{g['max_similarity_score']}` (Cluster {g['cluster_id']})")
                posts_df = pd.DataFrame(g['posts'])
                posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s', utc=True)
                st.dataframe(posts_df, use_container_width=True)
                st.markdown("---")
        else:
            st.warning("No coordinated groups found with current settings.")

        # Network graph
        t3 = time.time()
        G, pos, cluster_map = cached_network_graph(clustered_df)
        st.caption(f"Network graph build: {time.time() - t3:.2f}s")

        if G.number_of_nodes() == 0:
            st.info("No network to display.")
        else:
            # Filter nodes by min connections
            G_filtered = G.copy()
            remove_nodes = [n for n in list(G_filtered.nodes()) if G_filtered.degree(n) < min_connections]
            G_filtered.remove_nodes_from(remove_nodes)

            if G_filtered.number_of_nodes() == 0:
                st.info("No nodes meet the minimum connection threshold.")
            else:
                # Build Plotly scatter for nodes and edge traces
                edge_x, edge_y = [], []
                for u, v, data in G_filtered.edges(data=True):
                    x0, y0 = pos.get(u, (0, 0))
                    x1, y1 = pos.get(v, (0, 0))
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color="#888"),
                    hoverinfo='none',
                    mode='lines'
                )

                node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                for n in G_filtered.nodes():
                    x, y = pos.get(n, (0, 0))
                    node_x.append(x); node_y.append(y)
                    deg = G_filtered.degree(n)
                    platform = G_filtered.nodes[n].get('platform', 'Unknown')
                    cmap = cluster_map.get(n, -2)
                    node_color.append(cmap)
                    node_size.append(6 + 2*deg)
                    node_text.append(f"{n}<br>deg: {deg} • {platform} • cluster:{cmap}")

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='Viridis',
                        reversescale=True,
                        color=node_color,
                        size=node_size,
                        colorbar=dict(
                            thickness=10,
                            title='Cluster',
                            xanchor='left',
                            titleside='right'
                        ),
                        line_width=0.5
                    )
                )

                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title="CIB Interaction Network (accounts co-appearing in clusters)",
                                    title_x=0.5,
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=40),
                                ))
                st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Total CIB tab runtime: {time.time() - t_total:.2f}s")

# -------- TAB 3: Fundraising & Risk --------
with tab3:
    st.subheader("Fundraising Links & Risk Scoring")
    st.info("🔎 Scans outbound links in posts for known fundraising platforms.")
    run_funds = st.button("▶️ Scan for Fundraising Links")

    if run_funds:
        t4 = time.time()
        # If you want to enrich scoring using coordinated groups, you can first run CIB, then pass a DF here.
        fundraising_df = cached_find_fundraising_campaigns(df_for_analysis, pd.DataFrame())
        if not fundraising_df.empty:
            st.success(f"✅ Fundraising scan completed in {time.time() - t4:.2f}s")
            st.dataframe(fundraising_df, use_container_width=True)
        else:
            st.info("No fundraising links detected with current filters.")
