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
from io import BytesIO
from io import StringIO
from collections import Counter
import tldextract
import requests # Added for fetching default datasets from URL


# --- Set Page Config ---
st.set_page_config(page_title="CIB Dashboard", layout="wide")
st.title("🕵️ CIB Network Monitoring Dashboard")

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
    cleaned = re.sub(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{4}\b', '', cleaned)
    
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

# --- FIX: Added the missing extract_all_urls function here ---
def extract_all_urls(text):
    """Extracts all URLs from a given string."""
    if pd.isna(text):
        return []
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(str(text))


# --- Robust Timestamp Parser: Returns UNIX Timestamp (Integer) ---
def parse_timestamp_robust(timestamp):
    """
    Converts a timestamp string to a UNIX timestamp (integer seconds since epoch).
    Returns None if parsing fails.
    """
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, (int, float)):
        if 0 < timestamp < 253402300800:
            return int(timestamp)
        else:
            return None

    date_formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p',
        '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p',
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]

    try:
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed):
            return int(parsed.timestamp())
    except:
        pass

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
    meltwater_df,
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

    # Process Meltwater
    if meltwater_df is not None and not meltwater_df.empty:
        meltwater_df.columns = meltwater_df.columns.str.lower()
        mw = pd.DataFrame()
        mw['account_id'] = get_specific_col(meltwater_df, 'influencer')
        mw['content_id'] = get_specific_col(meltwater_df, 'tweet id')
        mw['object_id'] = get_specific_col(meltwater_df, meltwater_object_col.lower())
        mw['original_url'] = get_specific_col(meltwater_df, 'url')
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
        om['timestamp_share'] = get_specific_col(openmeasure_df, 'created_at')
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
        return df_processed
    
    df_processed.rename(columns={'original_url': 'URL'}, inplace=True)
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()

    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    if coordination_mode == "Text Content":
        df_processed['object_id'] = df_processed['object_id'].apply(clean_text_for_display)
        df_processed = df_processed[df_processed['object_id'].str.len() > 0].reset_index(drop=True)
        df_processed = df_processed[
            ~df_processed['object_id'].str.lower().str.startswith('rt @') &
            ~df_processed['object_id'].str.lower().str.startswith('qt @')
        ].reset_index(drop=True)

    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    elif coordination_mode == "Shared URLs":
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
    
    clustered_groups = df[df['cluster'] != -1].groupby('cluster')
    
    for cluster_id, group in clustered_groups:
        if len(group) < 2:
            continue
            
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col]].copy()
        clean_df = clean_df.rename(columns={text_col: 'text', 'timestamp_share': 'Timestamp'})
        clean_df = clean_df.reset_index(drop=True)
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(3, 5),
            max_features=max_features
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_df['text'])
        except Exception:
            continue
        
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        adj = {i: [] for i in range(len(clean_df))}
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
                        group_sim_scores = cosine_sim[np.ix_(group_indices, group_indices)]
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

    if coordination_type == "text":
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
    return find_coordinated_groups(_df, threshold, max_features)

@st.cache_data(show_spinner="🧩 Clustering texts...")
def cached_clustering(_df, eps, min_samples, max_features, data_source="default"):
    return cluster_texts(_df, eps, min_samples, max_features)

@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df_for_graph, coordination_type="text", data_source="default"):
    return build_user_interaction_graph(_df_for_graph, coordination_type)


# --- Auto-Encoding CSV Reader for Uploaded Files ---
# This helper function is now defined outside the main script flow
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
st.sidebar.header("📥 Data Source")
data_source = st.sidebar.radio("Choose data source:", ("Use Default Datasets", "Upload CSV Files"))

# Coordination Mode Selector
st.sidebar.header("🎯 Coordination Analysis Mode")
coordination_mode = st.sidebar.radio(
    "Analyze coordination by:",
    ("Text Content", "Shared URLs"),
    help="Choose what defines a coordinated action: similar text messages or sharing the same external link."
)

# Clear cache when mode or source changes
if 'last_data_source' not in st.session_state or st.session_state.last_data_source != data_source:
    st.cache_data.clear()
    st.session_state.last_data_source = data_source
if 'last_coordination_mode' not in st.session_state or st.session_state.last_coordination_mode != coordination_mode:
    st.cache_data.clear()
    st.session_state.last_coordination_mode = coordination_mode

combined_raw_df = pd.DataFrame()

# Load data
if data_source == "Use Default Datasets":
    st.sidebar.info("Using default datasets from GitHub.")
    with st.spinner("📥 Loading and combining default datasets..."):
        base_url = "https://raw.githubusercontent.com/hanna-tes/Humanitarian-Campaign-CIB-Network-Monitoring/refs/heads/main/"
        urls = {
            "meltwater": f"{base_url}3_replies_%E2%80%94_even_dots_%E2%80%94_can_break_the_algorithm_AN%20-%20Aug%2013%2C%202025%20-%2010%2037%2011%20AM.csv"
            #"civicsignals": f"{base_url}togo-or-lome-or-togo-all-story-urls-20250707142808.csv"
        }
        meltwater_df = pd.DataFrame()
        civicsignals_df = pd.DataFrame()
        
        for key, url in urls.items():
            try:
                df_temp = pd.read_csv(url, sep=',', low_memory=False)
                if not df_temp.empty:
                    if key == "meltwater":
                        meltwater_df = df_temp
                    elif key == "civicsignals":
                        civicsignals_df = df_temp
                    st.sidebar.success(f"✅ {key.capitalize()}: Loaded {len(df_temp)} rows")
                else:
                    st.sidebar.warning(f"⚠️ {key.capitalize()}: Empty file.")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Failed to load {key}: {e}")

        obj_map = {
            "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
            "civicsignals": "title" if coordination_mode == "Text Content" else "url",
            "openmeasure": "text" if coordination_mode == "Text Content" else "url"
        }
        combined_raw_df = combine_social_media_data(
            meltwater_df if not meltwater_df.empty else None,
            civicsignals_df if not civicsignals_df.empty else None,
            None,
            meltwater_object_col=obj_map["meltwater"],
            civicsignals_object_col=obj_map["civicsignals"],
            openmeasure_object_col=obj_map["openmeasure"]
        )
    if combined_raw_df.empty:
        st.warning("No data loaded from default datasets.")
    st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from default datasets.")

elif data_source == "Upload CSV Files":
    st.sidebar.info("Upload your CSV files below.")
    uploaded_meltwater = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater_upload")
    uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
    uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

    meltwater_df_upload = pd.DataFrame()
    civicsignals_df_upload = pd.DataFrame()
    openmeasure_df_upload = pd.DataFrame()

    meltwater_df_upload = read_uploaded_file(uploaded_meltwater, "Meltwater")
    civicsignals_df_upload = read_uploaded_file(uploaded_civicsignals, "CivicSignals")
    openmeasure_df_upload = read_uploaded_file(uploaded_openmeasure, "Open-Measure")
    
    with st.spinner("📥 Combining uploaded datasets..."):
        obj_map = {
            "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
            "civicsignals": "title" if coordination_mode == "Text Content" else "url",
            "openmeasure": "text" if coordination_mode == "Text Content" else "url"
        }
        combined_raw_df = combine_social_media_data(
            meltwater_df_upload,
            civicsignals_df_upload,
            openmeasure_df_upload,
            meltwater_object_col=obj_map["meltwater"],
            civicsignals_object_col=obj_map["civicsignals"],
            openmeasure_object_col=obj_map["openmeasure"]
        )

    if combined_raw_df.empty:
        st.warning("No data loaded from uploaded files.")
    st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from uploaded datasets.")
    
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mode:** `{coordination_mode}`")
st.sidebar.markdown(f"**Source:** `{data_source}`")
st.sidebar.markdown(f"**Total Rows After Combine:** `{len(combined_raw_df):,}`")

# --- Final Preprocess ---
with st.spinner("⏳ Preprocessing and mapping combined data..."):
    df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)

if df.empty:
    st.warning("No valid data after final preprocessing.")

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

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Filters (Apply to all tabs)")
if 'timestamp_share' not in df.columns or df['timestamp_share'].dtype != 'Int64':
    st.error("timestamp_share must be an integer (UNIX timestamp).")
    min_ts_filter = 0
    max_ts_filter = 253402300800
    filtered_df_global = pd.DataFrame()
else:
    min_ts_filter = df['timestamp_share'].min()
    max_ts_filter = df['timestamp_share'].max()
    if pd.isna(min_ts_filter) or pd.isna(max_ts_filter):
        min_date = pd.Timestamp.now().date()
        max_date = pd.Timestamp.now().date()
        filtered_df_global = pd.DataFrame()
    else:
        min_date = pd.to_datetime(min_ts_filter, unit='s', errors='coerce').date()
        max_date = pd.to_datetime(max_ts_filter, unit='s', errors='coerce').date()
        
        if pd.isna(min_date) or pd.isna(max_date):
            min_date = pd.Timestamp.now().date()
            max_date = pd.Timestamp.now().date()

        if min_date > max_date:
            min_date, max_date = max_date, min_date

        try:
            selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
        except Exception as e:
            st.sidebar.error(f"Error with date input: {e}. Resetting date range.")
            selected_date_range = [pd.Timestamp.now().date(), pd.Timestamp.now().date()]
        
        if len(selected_date_range) == 2:
            start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
            end_ts = int((pd.Timestamp(selected_date_range[1], tz='UTC') + timedelta(days=1) - timedelta(microseconds=1)).timestamp())
        else:
            start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
            end_ts = start_ts + 86400 - 1

        filtered_df_global = df[
            (df['timestamp_share'] >= start_ts) &
            (df['timestamp_share'] <= end_ts)
        ].copy()
        
        if 'Platform' in df.columns and not df['Platform'].empty:
            filtered_df_global = filtered_df_global[
                (filtered_df_global['Platform'].isin(df['Platform'].dropna().unique()))
            ].copy()
        else:
            st.warning("Platform column is missing or empty. Cannot filter by platform.")


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

if max_posts_for_analysis > 0 and len(filtered_df_global) > max_posts_for_analysis:
    df_for_analysis = filtered_df_global.sample(n=max_posts_for_analysis, random_state=42).copy()
    st.sidebar.warning(f"⚠️ Analyzing a random sample of **{len(df_for_analysis):,}** posts to improve performance.")
else:
    df_for_analysis = filtered_df_global.copy()
    st.sidebar.info(f"✅ Analyzing all **{len(df_for_analysis):,}** posts.")


if filtered_df_global.empty:
    st.warning("No data matches the selected filters.")
    df_for_analysis = pd.DataFrame() 

st.sidebar.markdown("### 📄 Export Filtered Results")
if not filtered_df_global.empty:
    filtered_csv_data = convert_df_to_csv(filtered_df_global)
    st.sidebar.download_button("Download Filtered Data (All Columns)", filtered_csv_data, "filtered_dashboard_data.csv", "text/csv")
else:
    st.sidebar.warning("No data to export after filtering.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Summary Statistics")
    st.markdown("### 🔬 Preprocessed Data Sample")
    st.markdown(f"**Data Source:** `{data_source}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(df):,}`")
    display_cols_overview = ['account_id', 'content_id', 'object_id', 'timestamp_share']
    existing_cols = [col for col in df.columns if col in display_cols_overview]
    if not df.empty and existing_cols:
        st.dataframe(df[existing_cols].head(10))
    else:
        st.info("No data available to display. Please upload or check the default data source.")
    
    if not filtered_df_global.empty:
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
        
        if coordination_mode == 'Text Content':
            filtered_df_global['emotional_phrase'] = filtered_df_global['object_id'].apply(contains_phrase)
            phrase_counts = filtered_df_global['emotional_phrase'].value_counts()
            if "Other" in phrase_counts.index:
                phrase_counts = phrase_counts.drop("Other")
            if not phrase_counts.empty:
                fig_phrases = px.bar(phrase_counts, title="Posts by Emotional Phrase", labels={'value': 'Count', 'index': 'Phrase'})
                st.plotly_chart(fig_phrases, use_container_width=True)
            else:
                st.info("No emotional phrases detected for the selected period.")
        else:
            st.info("Emotional phrase analysis is available in 'Text Content' coordination mode.")

        # Daily Volume
        st.markdown("---")
        plot_df = filtered_df_global.copy()
        plot_df['date'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True).dt.date
        time_series = plot_df.groupby('date').size()
        fig_ts = px.line(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'index': 'Date'}, markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)

        # Platform Distribution
        st.markdown("---")
        if 'Platform' in filtered_df_global.columns:
            platform_counts = filtered_df_global['Platform'].value_counts()
            # FIX: Pass values and names explicitly to px.pie for a Series
            fig_platform = px.pie(values=platform_counts.values, names=platform_counts.index, title="Posts by Platform")
            st.plotly_chart(fig_platform, use_container_width=True)
    else:
        st.info("Adjust the filters or upload data to see the overview. Current data selection is empty.")

# ==================== TAB 2: Analysis ====================
with tab2:
    st.subheader("🔍 Fundraising & Coordination Analysis")

    if coordination_mode == "Text Content":
        st.markdown("### 🔁 Similar Message Groups Detected")
        st.info("This analysis identifies groups of accounts posting very similar text messages.")
        eps = st.slider("DBSCAN eps (Similarity Tolerance)", 0.1, 1.0, 0.3, 0.05, key="eps_tab2")
        min_samples = st.slider("Min Posts in a Group", 2, 10, 2, key="min_samples_tab2")
        max_features = st.slider("Max TF-IDF Features", 1000, 5000, 2000, 500, key="max_features_tab2")
        threshold = st.slider("Pairwise Similarity Threshold", 0.8, 0.99, 0.9, 0.01, key="threshold_tab2")

        if st.button("🔍 Run Text-Based Analysis"):
            if df_for_analysis.empty:
                st.warning("No data available for text-based analysis. Adjust filters or check data source.")
            else:
                clustered = cached_clustering(df_for_analysis, eps, min_samples, max_features, data_source=data_source)
                groups = cached_find_coordinated_groups(clustered, threshold, max_features, data_source=data_source)
                
                if not groups:
                    st.warning("No text-based coordinated groups found above the threshold. Try lowering the threshold or min posts.")
                else:
                    st.success(f"Found {len(groups)} coordinated groups.")
                    for group in groups:
                        with st.expander(f"Group with {group['num_posts']} Posts from {group['num_accounts']} Accounts (Max Sim: {group['max_similarity_score']})"):
                            st.write(f"**Type of Coordination:** {group['coordination_type']}")
                            group_df = pd.DataFrame(group['posts'])
                            st.dataframe(group_df, use_container_width=True)
                    
                    all_posts = [post for group in groups for post in group['posts']]
                    if all_posts:
                        report_df = pd.DataFrame(all_posts)
                        csv = convert_df_to_csv(report_df) 
                        st.download_button("📥 Download Full Coordination Report", csv, "coordinated_groups_report.csv", "text/csv")

    elif coordination_mode == "Shared URLs":
        st.markdown("### 🔗 Shared URL Groups Detected")
        st.info("This analysis identifies groups of accounts that have shared the same external links.")
        if st.button("🔍 Run URL-Based Analysis"):
            if df_for_analysis.empty:
                st.warning("No data available for URL-based analysis. Adjust filters or check data source.")
            else:
                temp_df_urls = df_for_analysis[df_for_analysis['URL'].str.strip() != ''].copy()
                url_groups = temp_df_urls.groupby('URL').filter(lambda x: x['account_id'].nunique() > 1)
                
                if url_groups.empty:
                    st.warning("No URLs were shared by more than one unique account in the filtered data.")
                else:
                    grouped_data = url_groups.groupby('URL').agg(
                        Num_Accounts=('account_id', 'nunique'),
                        Post_Count=('content_id', 'size'),
                        Accounts_Shared_By=('account_id', lambda x: ', '.join(x.unique().astype(str)))
                    ).reset_index().sort_values(by='Num_Accounts', ascending=False)
                    
                    st.success(f"Found {len(grouped_data)} unique URLs shared by multiple accounts.")
                    st.dataframe(grouped_data, use_container_width=True)
                    
                    csv = convert_df_to_csv(grouped_data)
                    st.download_button("📥 Download Shared URL Report", csv, "shared_url_report.csv", "text/csv")

    st.markdown("---")
    st.subheader("💰 Suspicious Fundraising Links")

    if 'URL' in df_for_analysis.columns and not df_for_analysis.empty:
        all_urls_extracted = df_for_analysis.copy()
        all_urls_extracted['extracted_urls'] = all_urls_extracted['object_id'].apply(extract_all_urls)
        all_urls_exploded = all_urls_extracted.explode('extracted_urls').dropna(subset=['extracted_urls'])
        
        fundraising_domains = ['gofundme.com', 'paypal.me', 'donorbox.org']
        all_urls_exploded['domain'] = all_urls_exploded['extracted_urls'].astype(str).apply(lambda x: tldextract.extract(x).registered_domain)
        fundraising_links = all_urls_exploded[all_urls_exploded['domain'].isin(fundraising_domains)]

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
            
            st.dataframe(summary.sort_values(by="Num_Posts", ascending=False), use_container_width=True)
            csv = convert_df_to_csv(summary)
            st.download_button("📥 Download Fundraising Links Report", csv, "fundraising_links_report.csv", "text/csv")
        else:
            st.info("No known fundraising links detected for the selected period.")
    else:
        st.info("No URL data available for fundraising analysis.")

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

    if st.button("Build Network Graph", key="build_network"):
        if df_for_analysis.empty:
            st.warning("No data available to build a network. Adjust filters or check data source.")
        else:
            with st.spinner("🕸️ Building network..."):
                if coordination_mode == "Text Content":
                    clustered_df = cached_clustering(df_for_analysis, 0.3, 2, 2000, data_source=data_source)
                    G, pos, cluster_map = cached_network_graph(clustered_df, coordination_type="text", data_source=data_source)
                else:
                    G, pos, cluster_map = cached_network_graph(df_for_analysis, coordination_type="url", data_source=data_source)

            if not G.nodes():
                st.warning("No coordinated activity detected for the selected filters. The graph is empty.")
            else:
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u][0], pos[u][1]
                    x1, y1 = pos[v][0], pos[v][1]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

                node_x, node_y, node_text, node_color = [], [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    deg = G.degree(node)
                    platform = G.nodes[node]['platform']
                    node_text.append(f"Account: {node}<br>Degree: {deg}<br>Platform: {platform}")
                    node_color.append(G.nodes[node].get('cluster', -2))

                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                                        marker=dict(size=[G.degree(n)*3 + 10 for n in G.nodes()],
                                                    color=node_color, colorscale='Viridis', showscale=True))
                fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="CIB Network", showlegend=False, hovermode='closest'))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🚨 Risk Assessment: Top Central Accounts")
                degree_centrality = nx.degree_centrality(G)
                risk_df = pd.DataFrame(degree_centrality.items(), columns=['Account', 'Centrality']).sort_values('Centrality', ascending=False).head(20)
                risk_df['Risk Score'] = (risk_df['Centrality'] * 100).round(2)
                st.dataframe(risk_df, use_container_width=True)
