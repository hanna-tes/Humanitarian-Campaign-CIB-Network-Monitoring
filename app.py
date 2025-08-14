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
from collections import Counter

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
FUNDRAISING_KEYWORDS = ["donate", "fund", "gofundme", "paypal", "patreon", "venmo", "give", "help"]

# --- Helper Functions ---
def infer_platform_from_url(url):
    """Infers the social media or news platform from a given URL."""
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    if "tiktok.com" in url:
        return "TikTok"
    elif "facebook.com" in url or "fb.watch" in url:
        return "Facebook"
    elif "twitter.com" in url or "x.com" in url:
        return "X"
    elif "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    elif "instagram.com" in url:
        return "Instagram"
    elif "telegram.me" in url or "t.me" in url:
        return "Telegram"
    elif url.startswith("https://") or url.startswith("http://"):
        media_domains = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"]
        if any(domain in url for domain in media_domains):
            return "News/Media"
        return "Media"
    else:
        return "Unknown"

# NEW function to extract URLs from text
def extract_urls_from_text(text):
    """Extracts all URLs from a given text string."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    return url_pattern.findall(text)

# NEW function to extract only fundraising URLs
def extract_fundraising_urls(text, keywords):
    """Extracts all URLs from a text and filters for fundraising keywords."""
    all_urls = extract_urls_from_text(text)
    fundraising_urls = [url for url in all_urls if any(kw in url.lower() for kw in keywords)]
    return fundraising_urls

def extract_original_text(text):
    """
    Cleans text by removing RT/QT prefixes, @mentions, URLs, and normalizing spaces.
    Crucially, it now removes dates and years to prevent them from dominating the narrative summary.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    
    # --- NEW: Remove dates, years, and common month names ---
    # Matches patterns like "June 17", "17 June", "17/06/2025", "2025"
    cleaned = re.sub(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{4}\b', '', cleaned)
    
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

# --- Robust Timestamp Parser: Returns UNIX Timestamp (Integer) ---
def parse_timestamp_robust(timestamp):
    """
    Converts a timestamp string to a UNIX timestamp (integer seconds since epoch).
    Returns None if parsing fails.
    """
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, (int, float)):
        if 0 < timestamp < 253402300800:  # Valid range: 1970–9999
            return int(timestamp)
        else:
            return None

    # List of common timestamp formats
    date_formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p',
        '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p',
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]

    # Try direct parsing
    try:
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed):
            return int(parsed.timestamp())
    except:
        pass

    # Try each format
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed):
                return int(parsed.timestamp())
        except (ValueError, TypeError):
            continue
    return None

# --- Combine Multiple Datasets with Flexible Object Column ---
def combine_social_media_data(
    meltwater_dfs,
    civicsignals_df,
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

    # Process Meltwater (now accepts a list of dataframes)
    if meltwater_dfs is not None and meltwater_dfs:
        for mw_df in meltwater_dfs:
            if mw_df is None or mw_df.empty:
                continue
            mw_df.columns = mw_df.columns.str.lower()
            mw = pd.DataFrame()
            mw['account_id'] = get_specific_col(mw_df, 'influencer')
            mw['content_id'] = get_specific_col(mw_df, 'tweet id')
            mw['object_id'] = get_specific_col(mw_df, meltwater_object_col.lower())
            mw['original_url'] = get_specific_col(mw_df, 'url')
            mw['timestamp_share'] = get_specific_col(mw_df, 'date')
            mw['source_dataset'] = 'Meltwater'
            combined_dfs.append(mw)

    # Process CivicSignals
    if civicsignals_df is not None and not civicsignals_df.empty:
        civicsignals_df.columns = civicsignals_df.columns.str.lower()
        cs = pd.DataFrame()
        cs['account_id'] = get_specific_col(civicsignals_df, 'media_name')
        cs['content_id'] = get_specific_col(civicsignals_df, 'stories_id')
        cs['object_id'] = get_specific_col(civicsignals_df, civicsignals_object_col.lower())
        cs['original_url'] = get_specific_col(civicsignals_df, 'url')
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
        om['original_url'] = get_specific_col(openmeasure_df, 'url')
        om['timestamp_share'] = get_specific_col(om_df, 'created_at')
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)

    if not combined_dfs:
        return pd.DataFrame()

    combined = pd.concat(combined_dfs, ignore_index=True)
    combined = combined.dropna(subset=['account_id', 'content_id', 'timestamp_share', 'object_id']).copy()
    combined['account_id'] = combined['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    combined['content_id'] = combined['content_id'].astype(str).str.replace('"', '', regex=False).str.strip()
    combined['original_url'] = combined['original_url'].astype(str).replace('nan', '').fillna('')
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')

    # Convert timestamp to UNIX
    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined['timestamp_share'] = combined['timestamp_share'].astype('Int64')  # Nullable integer

    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')
    combined = combined[combined['object_id'].str.strip() != ""].copy()
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share']).reset_index(drop=True)
    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    """
    Performs final preprocessing steps on the combined DataFrame.
    Respects coordination_mode: uses text or URL as object_id.
    Ensures timestamp_share is UNIX integer.
    """
    df_processed = df.copy()

    # Create all required columns even if the dataframe is empty
    if df_processed.empty:
        df_processed['URL'] = pd.Series([], dtype='object')
        df_processed['Platform'] = pd.Series([], dtype='object')
        df_processed['original_text'] = pd.Series([], dtype='object')
        df_processed['Outlet'] = pd.Series([], dtype='object')
        df_processed['Channel'] = pd.Series([], dtype='object')
        df_processed['object_id'] = pd.Series([], dtype='object')
        df_processed['timestamp_share'] = pd.Series([], dtype='Int64')
        df_processed['fundraising_urls_in_text'] = pd.Series([], dtype='object')
        return df_processed
    
    # NEW: Extract fundraising URLs from text and put them in a dedicated column
    df_processed['fundraising_urls_in_text'] = df_processed['object_id'].apply(lambda x: extract_fundraising_urls(x, FUNDRAISING_KEYWORDS))
    
    # We use a single, reliable URL for other analyses
    df_processed['URL'] = df_processed['original_url'].fillna(df_processed['fundraising_urls_in_text'].apply(lambda x: x[0] if x else None))

    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()

    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    elif coordination_mode == "Shared URLs":
        # For URL-based analysis, we use the extracted URL itself as the text to cluster on
        df_processed = df_processed[df_processed['URL'].notna()].copy()
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')

    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)

    if 'Outlet' not in df_processed.columns:
        df_processed['Outlet'] = np.nan
    if 'Channel' not in df_processed.columns:
        df_processed['Channel'] = np.nan

    if df_processed.empty:
        st.error("❌ No valid data after final preprocessing.")

    return df_processed

# --- Analysis Functions ---
def cluster_texts(df, eps, min_samples, max_features):
    if 'original_text' not in df.columns or df['original_text'].nunique() <= 1:
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    
    texts_to_cluster = df['original_text'].astype(str).tolist()
    if not texts_to_cluster or all(t.strip() == "" for t in texts_to_cluster):
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts_to_cluster)
    except ValueError as e:
        df_copy = df.copy()
        df_copy['cluster'] = -1
        return df_copy
    
    eps = max(0.01, min(0.99, eps))
    clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(tfidf_matrix)
    df_copy = df.copy()
    df_copy['cluster'] = clustering.labels_
    return df_copy

def find_coordinated_groups(df, threshold, max_features):
    """
    Groups highly similar posts into coordination groups for better analysis.
    Crucially, a group is only considered coordinated if it involves more than one unique account.
    """
    text_col = 'original_text'
    social_media_platforms = {'TikTok', 'Facebook', 'X', 'YouTube', 'Instagram', 'Telegram'}
    
    coordination_groups = {}
    
    # We now perform a simpler grouping for URLs
    if df['original_text'].nunique() > 1 and len(df) > 1:
        df_clustered = cluster_texts(df, eps=0.4, min_samples=2, max_features=max_features)
        clustered_groups = df_clustered[df_clustered['cluster'] != -1].groupby('cluster')
    else:
        clustered_groups = df.groupby(pd.Series(range(len(df)))) # Treat each post as a single group if there's no clustering possible
    
    for cluster_id, group in clustered_groups:
        if len(group) < 2:
            continue
        
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col, 'fundraising_urls_in_text']].copy()
        clean_df = clean_df.rename(columns={text_col: 'text', 'timestamp_share': 'Timestamp'})
        clean_df = clean_df.reset_index(drop=True)
        
        # Build adjacency list
        adj = {i: [] for i in range(len(clean_df))}
        if len(clean_df) > 1:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 1),
                max_features=max_features
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(clean_df['text'])
                cosine_sim = cosine_similarity(tfidf_matrix)
            except Exception:
                continue

            for i in range(len(clean_df)):
                for j in range(i + 1, len(clean_df)):
                    if cosine_sim[i, j] >= threshold:
                        adj[i].append(j)
                        adj[j].append(i)
                    
        visited = set()
        group_id_counter = 0
        
        for i in range(len(clean_df)):
            if i not in visited:
                group_indices = []
                q = [i]
                visited.add(i)
                while q:
                    u = q.pop(0)
                    group_indices.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                
                if len(group_indices) > 1:
                    group_posts = clean_df.iloc[group_indices].copy()
                    
                    if len(group_posts['account_id'].unique()) > 1:
                        group_sim_scores = cosine_sim[np.ix_(group_indices, group_indices)] if 'cosine_sim' in locals() else np.array([[0]])
                        max_sim = group_sim_scores.max() if group_sim_scores.size > 0 else 0.0

                        coordination_groups[f"group_{group_id_counter}"] = {
                            "posts": group_posts.to_dict('records'),
                            "num_posts": len(group_posts),
                            "num_accounts": len(group_posts['account_id'].unique()),
                            "max_similarity_score": round(max_sim, 3),
                            "coordination_type": "TBD"
                        }
                        group_id_counter += 1

    final_groups = []
    for group_id, group_data in coordination_groups.items():
        posts_df = pd.DataFrame(group_data['posts'])
        platforms = posts_df['Platform'].unique()
        
        social_media_platforms_in_group = [p for p in platforms if p in social_media_platforms]
        media_platforms_in_group = [p for p in platforms if p in {'News/Media', 'Media'}]

        if len(media_platforms_in_group) > 1 and len(social_media_platforms_in_group) == 0:
            coordination_type = "Syndication (Media Outlets)"
        elif len(social_media_platforms_in_group) > 1 and len(media_platforms_in_group) == 0:
            coordination_type = "Coordinated Amplification (Social Media)"
        elif len(social_media_platforms_in_group) > 0 and len(media_platforms_in_group) > 0:
            coordination_type = "Media-to-Social Replication"
        else:
            coordination_type = "Other / Uncategorized"
        
        group_data['coordination_type'] = coordination_type
        final_groups.append(group_data)
        
    return final_groups


def build_user_interaction_graph(df, coordination_type="text"):
    G = nx.Graph()
    influencer_column = 'account_id'
    
    if df.empty or 'account_id' not in df.columns:
        return G, {}, {}

    if coordination__type == "text":
        if 'cluster' not in df.columns:
            return G, {}, {}
        grouped = df.groupby('cluster')
        for cluster_id, group in grouped:
            if cluster_id == -1 or len(group[influencer_column].unique()) < 2:
                for user in group[influencer_column].dropna().unique():
                    if user not in G:
                        G.add_node(user, cluster=cluster_id)
                continue
            users_in_cluster = group[influencer_column].dropna().unique().tolist()
            for u1, u2 in combinations(users_in_cluster, 2):
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
                    if user not in G:
                        G.add_node(user)
                continue
            for u1, u2 in combinations(users_sharing_url, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)

    all_influencers = df[influencer_column].dropna().unique().tolist()
    influencer_platform_map = df.groupby(influencer_column)['Platform'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()

    for inf in all_influencers:
        if inf not in G.nodes():
            G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')
        if coordination_type == "text":
            clusters = df[df[influencer_column] == inf]['cluster'].dropna()
            G.nodes[inf]['cluster'] = clusters.mode()[0] if not clusters.empty else -2
        elif coordination_type == "url":
            shared_urls = df[(df[influencer_column] == inf) & df['URL'].notna() & (df['URL'].str.strip() != '')]['URL'].unique()
            G.nodes[inf]['cluster'] = f"SharedURL_Group_{hash(tuple(sorted(shared_urls))) % 100}" if len(shared_urls) > 0 else "NoSharedURL"

    if G.nodes():
        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
        top_n_nodes = sorted_nodes[:st.session_state.get('max_nodes_to_display', 40)]
        subgraph = G.subgraph(top_n_nodes)
        
        pos = nx.kamada_kawai_layout(subgraph)
        cluster_map = {node: G.nodes[node].get('cluster', -2) for node in subgraph.nodes()}
        return subgraph, pos, cluster_map
    else:
        return G, {}, {}


# --- Cached Functions ---
@st.cache_data(show_spinner="🔍 Finding coordinated posts within clusters...")
def cached_find_coordinated_groups(_df, threshold, max_features, data_source="default"):
    if _df.empty:
        return []
    return find_coordinated_groups(_df, threshold, max_features)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df, eps, min_samples, max_features, data_source="default"):
    if _df.empty:
        return _df.assign(cluster=-1)
    return cluster_texts(_df, eps, min_samples, max_features)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df_for_graph, coordination_type="text", data_source="default"):
    if _df_for_graph.empty:
        return nx.Graph(), {}, {}
    return build_user_interaction_graph(_df_for_graph, coordination_type)

# --- Auto-Encoding CSV Reader for Uploaded Files ---
def read_uploaded_file(uploaded_file, file_name):
    if not uploaded_file:
        return pd.DataFrame()
    
    bytes_data = uploaded_file.getvalue()
    encodings = ['utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin1', 'cp1252']
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
    sep = '\t' if '\t' in sample_line else ','
    
    try:
        df = pd.read_csv(StringIO(decoded_content), sep=sep, low_memory=False)
        st.sidebar.success(f"✅ {file_name}: Loaded {len(df)} rows (sep='{sep}', enc='{detected_enc}')")
        return df
    except Exception as e:
        st.error(f"❌ Failed to parse {file_name} CSV after decoding: {e}")
        return pd.DataFrame()

# --- Sidebar: Data Source & Coordination Mode ---
st.sidebar.header("📥 Data Upload")

# Coordination Mode Selector
st.sidebar.header("🎯 Coordination Analysis Mode")
coordination_mode = st.sidebar.radio(
    "Analyze coordination by:",
    ("Text Content", "Shared URLs"),
    help="Choose what defines a coordinated action: similar text messages or sharing the same external link."
)

# Clear cache when mode changes
if 'last_coordination_mode' not in st.session_state or st.session_state.last_coordination_mode != coordination_mode:
    st.cache_data.clear()
    st.session_state.last_coordination_mode = coordination_mode

combined_raw_df = pd.DataFrame()

# "Upload CSV Files" mode is now the only mode
st.sidebar.info("Upload your CSV files below.")
uploaded_meltwater_files = st.sidebar.file_uploader("Upload Meltwater CSV(s)", type=["csv"], accept_multiple_files=True, key="meltwater_upload")
uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

# Process multiple Meltwater files
meltwater_dfs_upload = []
if uploaded_meltwater_files:
    for i, file in enumerate(uploaded_meltwater_files):
        df_temp = read_uploaded_file(file, f"Meltwater File {i+1}")
        if not df_temp.empty:
            meltwater_dfs_upload.append(df_temp)

civicsignals_df_upload = read_uploaded_file(uploaded_civicsignals, "CivicSignals")
openmeasure_df_upload = read_uploaded_file(uploaded_openmeasure, "Open-Measure")

if meltwater_dfs_upload or not civicsignals_df_upload.empty or not openmeasure_df_upload.empty:
    with st.spinner("📥 Combining uploaded datasets..."):
        obj_map = {
            "meltwater": "hit sentence",
            "civicsignals": "title",
            "openmeasure": "text"
        }
        combined_raw_df = combine_social_media_data(
            meltwater_dfs_upload,
            civicsignals_df_upload,
            openmeasure_df_upload,
            meltwater_object_col=obj_map["meltwater"],
            civicsignals_object_col=obj_map["civicsignals"],
            openmeasure_object_col=obj_map["openmeasure"]
        )

if combined_raw_df.empty:
    st.warning("No data loaded from uploaded files. Please upload at least one CSV file.")
    st.stop()
st.sidebar.success(f"✅ Combined {len(combined_raw_df):,} posts from uploaded datasets.")


# --- Final Preprocess ---
with st.spinner("⏳ Preprocessing and mapping combined data..."):
    df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)

if df.empty:
    st.warning("No valid data after final preprocessing. Please adjust the filters or check your data source.")
    st.stop()


# --- Sidebar Filters ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Global Filters (Apply to all tabs)")
if 'timestamp_share' not in df.columns or df['timestamp_share'].dtype != 'Int64':
    st.error("timestamp_share must be an integer (UNIX timestamp).")
    min_ts = 0
    max_ts = 253402300800
else:
    min_ts = df['timestamp_share'].min()
    max_ts = df['timestamp_share'].max()

if pd.isna(min_ts) or pd.isna(max_ts) or min_ts == max_ts:
    min_date = pd.Timestamp.now().date()
    max_date = pd.Timestamp.now().date()
else:
    min_date = pd.to_datetime(min_ts, unit='s').date() if pd.notna(min_ts) else pd.Timestamp.now().date()
    max_date = pd.to_datetime(max_ts, unit='s').date() if pd.notna(max_ts) else pd.Timestamp.now().date()

try:
    selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
except:
    selected_date_range = [min_date, max_date]

if len(selected_date_range) == 2:
    start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
    end_ts = int((pd.Timestamp(selected_date_range[1], tz='UTC') + timedelta(days=1) - timedelta(microseconds=1)).timestamp())
else:
    start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
    end_ts = start_ts + 86400 - 1

if 'Platform' in df.columns:
    filtered_df_global = df[
        (df['timestamp_share'] >= start_ts) &
        (df['timestamp_share'] <= end_ts) &
        (df['Platform'].isin(df['Platform'].dropna().unique()))
    ].copy()
else:
    st.error("Platform column is missing from the DataFrame.")
    filtered_df_global = pd.DataFrame()

# Add new control to limit posts
st.sidebar.markdown("---")
st.sidebar.subheader("⏩ Performance Controls")
max_posts_for_analysis = st.sidebar.number_input(
    "Limit Posts for Analysis (0 for all)",
    min_value=0,
    value=0,
    step=1000,
    help="To speed up analysis on large datasets, enter a number to process a random sample of posts. Set to 0 to use all posts."
)
st.sidebar.markdown(f"**Filtered Posts:** `{len(filtered_df_global):,}`")

# Apply sampling if requested
if max_posts_for_analysis > 0 and len(filtered_df_global) > max_posts_for_analysis:
    df_for_analysis = filtered_df_global.sample(n=max_posts_for_analysis, random_state=42).copy()
    st.sidebar.warning(f"⚠️ Analyzing a random sample of **{len(df_for_analysis):,}** posts to improve performance.")
else:
    df_for_analysis = filtered_df_global.copy()
    st.sidebar.info(f"✅ Analyzing all **{len(df_for_analysis):,}** posts.")


if filtered_df_global.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


# --- Download Combined Data ---
st.sidebar.markdown("### 💾 Download Combined & Preprocessed Data")
@st.cache_data
def convert_df_to_csv(data_frame):
    return data_frame.to_csv(index=False).encode('utf-8')

download_df_columns = ['account_id', 'content_id', 'object_id', 'timestamp_share']
downloadable_df = df[download_df_columns].copy() if all(col in df.columns for col in download_df_columns) else pd.DataFrame()

if not downloadable_df.empty:
    combined_preprocessed_csv = convert_df_to_csv(downloadable_df)
    st.sidebar.download_button(
        "Download Preprocessed Dataset (Core Columns)",
        combined_preprocessed_csv,
        f"preprocessed_combined_core_data_{coordination_mode.replace(' ', '_').lower()}.csv",
        "text/csv",
        help="Downloads the data after all preprocessing and column mapping. 'object_id' contains either text or URL based on your selection."
    )
else:
    st.sidebar.warning("Could not create downloadable dataset with core columns.")


# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Humanitarian Campaign Overview")
    st.markdown(f"**Coordination Mode:** `{coordination_mode}` | **Total Posts (All Time):** `{len(df):,}` | **Posts After Filters:** `{len(filtered_df_global):,}`")

    if not filtered_df_global.empty:
        # Post Volume per Time
        st.markdown("### 📈 Post Volume Over Time")
        filtered_df_global['date_utc'] = pd.to_datetime(filtered_df_global['timestamp_share'], unit='s', utc=True)
        time_series_data = filtered_df_global.set_index('date_utc').resample('D').size().reset_index(name='post_count')
        fig_volume = px.line(time_series_data, x='date_utc', y='post_count', title='Posts per Day (UTC)')
        st.plotly_chart(fig_volume, use_container_width=True)

        # Sample data
        st.markdown("### 📋 Raw Data Sample (First 10 Rows)")
        display_df = filtered_df_global.copy()
        display_df['date_utc'] = pd.to_datetime(display_df['timestamp_share'], unit='s', utc=True)
        st.dataframe(display_df[['account_id', 'content_id', 'object_id', 'date_utc']].head(10), use_container_width=True)

        # Top Influencers
        st.markdown("---")
        top_influencers = filtered_df_global['account_id'].value_counts().head(10)
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

        # Apply the function to create a new column
        filtered_df_global['phrase_type'] = filtered_df_global['object_id'].apply(contains_phrase)
        
        # Plotting the count of each phrase
        phrase_counts = filtered_df_global['phrase_type'].value_counts()
        fig_phrases = px.bar(
            phrase_counts.drop("Other", errors='ignore'),
            title="Emotional & Algorithmic Phrases Detected",
            labels={'value': 'Number of Posts', 'index': 'Phrase'}
        )
        st.plotly_chart(fig_phrases, use_container_width=True)

# ==================== TAB 2: Analysis ====================
with tab2:
    st.subheader("🔎 Coordinated Activity Analysis")
    
    if df_for_analysis.empty:
        st.warning("No data available for analysis. Please adjust the filters or check your data source.")
    else:
        # Analysis parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            if coordination_mode == "Text Content":
                eps = st.slider("DBSCAN Epsilon (Text)", min_value=0.1, max_value=1.0, value=0.4, step=0.05,
                                help="Lower value means stricter similarity. Recommended: 0.3-0.5.")
            else:
                eps = 0.5
        with col2:
            if coordination_mode == "Text Content":
                min_samples = st.slider("Min Samples (Text)", min_value=2, max_value=10, value=3, step=1,
                                        help="Minimum number of posts to form a cluster.")
            else:
                min_samples = 2
        with col3:
            threshold = st.slider("Coordination Similarity Threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
                                  help="Posts must have a similarity score above this to be considered coordinated.")
        
        max_features = st.slider("Max TF-IDF Features", min_value=100, max_value=10000, value=5000, step=100,
                                 help="Limits the vocabulary size for text vectorization. Helps with performance and noise reduction.")
        
        run_analysis = st.button("Run Coordination Analysis")
        
        if run_analysis:
            with st.spinner("⏳ Running analysis..."):
                df_clustered = cached_clustering(df_for_analysis, eps, min_samples, max_features, "uploaded_data")
                
                if df_clustered is not None and 'cluster' in df_clustered.columns:
                    coordinated_groups = cached_find_coordinated_groups(df_clustered, threshold, max_features, "uploaded_data")

                    st.markdown("### 📈 Coordination Summary")
                    total_coordinated_posts = sum(g['num_posts'] for g in coordinated_groups)
                    total_coordinated_accounts = sum(g['num_accounts'] for g in coordinated_groups)
                    
                    # --- NEW: Visually appealing metrics ---
                    col_met1, col_met2, col_met3 = st.columns(3)
                    with col_met1:
                        st.metric("Coordinated Groups Found", len(coordinated_groups), help="Total number of unique coordinated groups identified.")
                    with col_met2:
                        st.metric("Posts in Groups", total_coordinated_posts, help="Total number of posts that are part of a coordinated group.")
                    with col_met3:
                        st.metric("Accounts in Groups", total_coordinated_accounts, help="Total number of unique accounts participating in coordinated groups.")
                    
                    st.info(f"✨ Found **{len(coordinated_groups):,}** coordinated groups with a total of **{total_coordinated_posts:,}** posts from **{total_coordinated_accounts:,}** unique accounts.")

                    if coordinated_groups:
                        st.markdown("### 📊 Top 10 Coordinated Groups")
                        
                        top_groups = sorted(coordinated_groups, key=lambda x: x['num_posts'], reverse=True)[:10]

                        for i, group in enumerate(top_groups):
                            with st.expander(f"Group {i+1}: {group['num_posts']} Posts ({group['coordination_type']}) - Max Sim: {group['max_similarity_score']}"):
                                group_df = pd.DataFrame(group['posts'])
                                st.dataframe(group_df[['account_id', 'text', 'URL', 'Platform']].head(10), use_container_width=True)
                                
                                # Plot a timeline for this group
                                group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='s', utc=True)
                                group_df['Date'] = group_df['Timestamp'].dt.date
                                timeline = group_df.groupby('Date').size().reset_index(name='count')
                                fig = px.line(timeline, x='Date', y='count', title="Posts per Day for this Group")
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"group_chart_{i}")
                    else:
                        st.info("No coordinated groups found with the current parameters.")
                else:
                    st.error("Clustering failed. Please check your data or parameters.")


# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🌐 Network & Risk Analysis")
    
    if df_for_analysis.empty:
        st.warning("No data available for network analysis. Please adjust the filters or check your data source.")
    else:
        # Network graph parameters
        col1, col2 = st.columns(2)
        with col1:
            network_mode = st.radio(
                "Network Based On:",
                ("Similar Content", "Shared URLs"),
                help="Build the network based on who posts similar content or who shares the same URLs."
            )
        with col2:
            st.session_state.max_nodes_to_display = st.slider(
                "Max Nodes to Display",
                min_value=10, max_value=200, value=40, step=10,
                help="Limits the number of nodes (accounts) shown in the graph for performance."
            )

        run_network = st.button("Generate Network Graph")

        if run_network:
            with st.spinner("⏳ Generating network graph..."):
                if network_mode == "Similar Content":
                    df_clustered_for_network = cached_clustering(df_for_analysis, eps=0.4, min_samples=3, max_features=5000)
                    graph, pos, cluster_map = cached_network_graph(df_clustered_for_network, coordination_type="text", data_source="uploaded_data")
                else: # Shared URLs
                    graph, pos, cluster_map = cached_network_graph(df_for_analysis, coordination_type="url", data_source="uploaded_data")

                if graph.number_of_nodes() > 0:
                    edge_x = []
                    edge_y = []
                    for edge in graph.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.append(x0)
                        edge_x.append(x1)
                        edge_x.append(None)
                        edge_y.append(y0)
                        edge_y.append(y1)
                        edge_y.append(None)

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )

                    node_x = []
                    node_y = []
                    for node in graph.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        marker=dict(
                            showscale=True,
                            colorscale='Viridis',
                            reversescale=True,
                            color=[],
                            size=10,
                            colorbar=dict(
                                thickness=15,
                                title='Node Centrality',
                                xanchor='left',
                                titleside='right'
                            ),
                            line_width=2
                        )
                    )

                    node_adjacencies = []
                    node_text = []
                    for node, adjacencies in enumerate(graph.adjacency()):
                        node_adjacencies.append(len(adjacencies[1]))
                        node_text.append(f'Account: {list(graph.nodes())[node]}<br># of connections: {len(adjacencies[1])}')

                    node_trace.marker.color = node_adjacencies
                    node_trace.text = node_text

                    fig = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title='<br>Network of Coordinated Accounts',
                                        titlefont_size=16,
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20, l=5, r=5, t=40),
                                        annotations=[dict(
                                            showarrow=False,
                                            text="Nodes represent accounts. Links show coordinated activity.",
                                            xref="paper", yref="paper",
                                            x=0.005, y=-0.002)],
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("The network graph could not be generated. There may not be enough coordinated activity to form a network.")
        st.markdown("---")
        st.subheader("💰 Suspicious Fundraising Analysis")
        
        # This part of the code needs to be run after a clustering/grouping analysis has been completed.
        # It relies on the 'coordinated_groups' variable being populated.
        # We'll use a check to ensure it exists before proceeding.
        
        if 'coordinated_groups' not in locals():
            st.info("Please run the 'Coordination Analysis' in the 'Analysis' tab first to identify coordinated posts.")
        else:
            # We now iterate through the dedicated fundraising_urls_in_text column
            all_fundraising_urls = []
            for group in coordinated_groups:
                for post in group['posts']:
                    if 'fundraising_urls_in_text' in post and post['fundraising_urls_in_text']:
                        all_fundraising_urls.extend(post['fundraising_urls_in_text'])
            
            if all_fundraising_urls:
                st.info(f"🔎 Found **{len(all_fundraising_urls)}** posts containing potential fundraising links.")
                
                def get_domain(url):
                    extracted = tldextract.extract(url)
                    return f"{extracted.domain}.{extracted.suffix}" if extracted.domain and extracted.suffix else "Unknown"

                fundraising_df = pd.DataFrame(all_fundraising_urls, columns=['URL'])
                fundraising_df['Domain'] = fundraising_df['URL'].apply(get_domain)
                
                url_counts = fundraising_df.groupby(['Domain', 'URL']).size().reset_index(name='Times Shared in Coordinated Posts')
                url_counts = url_counts.sort_values(by='Times Shared in Coordinated Posts', ascending=False)
                
                st.markdown("##### Top Fundraising URLs in Coordinated Posts")
                st.warning("🚨 **Warning**: The table below shows fundraising links with a high frequency of coordinated posts. Investigate their legitimacy.")
                
                st.dataframe(url_counts, use_container_width=True)

            else:
                st.info("No fundraising-related URLs were found in coordinated groups.")
