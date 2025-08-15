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

def extract_urls_from_text(text):
    """Extracts all URLs from a given text string."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    return url_pattern.findall(text)

def extract_fundraising_urls(text, keywords):
    """Extracts all URLs from a text and filters for fundraising keywords."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # First, find all URLs in the text
    all_urls = extract_urls_from_text(text)
    
    # Then, check each URL for fundraising keywords
    fundraising_urls = [url for url in all_urls if any(kw in url.lower() for kw in keywords)]
    
    # Also, check the original text for fundraising keywords
    # This captures posts that mention "Gofundme" without a full URL
    text_fundraising_mentions = [kw for kw in keywords if kw in text.lower()]
    
    # For now, we'll return the URLs, as that's what's been requested to be displayed
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
    
    cleaned = re.sub(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{4}\b', '', cleaned)
    
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

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
        om['object_id'] = get_specific_col(om_df, 'text')
        om['original_url'] = get_specific_col(om_df, 'created_at')
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

    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined['timestamp_share'] = combined['timestamp_share'].astype('Int64')

    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')
    combined = combined[combined['object_id'].str.strip() != ""].copy()
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share']).reset_index(drop=True)
    return combined

def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    """
    Performs final preprocessing steps on the combined DataFrame.
    Respects coordination_mode: uses text or URL as object_id.
    Ensures timestamp_share is UNIX integer.
    """
    df_processed = df.copy()

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
    
    df_processed['URL'] = df_processed['original_url'].fillna(df_processed['fundraising_urls_in_text'].apply(lambda x: x[0] if x else None))

    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()

    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    # FIX: Correctly set original_text based on coordination_mode
    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    elif coordination_mode == "Shared URLs":
        # When coordinating by URLs, original_text should be the URL itself
        df_processed = df_processed[df_processed['URL'].notna() & (df_processed['URL'].str.strip() != "")].copy()
        df_processed['original_text'] = df_processed['URL'].astype(str)

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

def find_coordinated_groups(df, threshold, max_features, time_window_minutes):
    text_col = 'original_text'
    social_media_platforms = {'TikTok', 'Facebook', 'X', 'YouTube', 'Instagram', 'Telegram'}
    
    coordination_groups = {}
    
    if df['original_text'].nunique() > 1 and len(df) > 1:
        # We perform clustering first to narrow down the search space
        df_clustered = cluster_texts(df, eps=0.4, min_samples=2, max_features=max_features)
        clustered_groups = df_clustered[df_clustered['cluster'] != -1].groupby('cluster')
    else:
        clustered_groups = df.groupby(pd.Series(range(len(df))))
    
    for cluster_id, group in clustered_groups:
        if len(group) < 2:
            continue
        
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col, 'fundraising_urls_in_text']].copy()
        clean_df = clean_df.rename(columns={text_col: 'text', 'timestamp_share': 'Timestamp'})
        clean_df = clean_df.sort_values('Timestamp').reset_index(drop=True)
        
        adj = {i: [] for i in range(len(clean_df))}
        
        # New, more efficient logic to avoid O(N^2) similarity matrix calculation
        texts_in_group = clean_df['text'].tolist()
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 1),
            max_features=max_features
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(texts_in_group)
        except Exception:
            continue
        
        for i in range(len(clean_df)):
            post_i_ts = clean_df.loc[i, 'Timestamp']
            # Find all subsequent posts within the time window
            for j in range(i + 1, len(clean_df)):
                post_j_ts = clean_df.loc[j, 'Timestamp']
                time_diff = (post_j_ts - post_i_ts) / 60
                
                if time_diff > time_window_minutes:
                    # Since the data is sorted by time, we can break the inner loop early
                    break
                
                # Only compute similarity if posts are within the time window
                cosine_sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])
                if cosine_sim[0, 0] >= threshold:
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
                        # Re-calculate max similarity for the final, coordinated group
                        group_texts = group_posts['text'].tolist()
                        group_tfidf = vectorizer.transform(group_texts)
                        group_sim_scores = cosine_similarity(group_tfidf)
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

def get_account_campaign_involvement(df_for_analysis, threshold, max_features, time_window_minutes):
    """
    Identifies accounts involved in coordinated groups across multiple campaign phrases.
    Now correctly passes parameters for coordination analysis.
    """
    account_campaigns = {}
    for phrase in PHRASES_TO_TRACK:
        posts_with_phrase = df_for_analysis[df_for_analysis['object_id'].str.contains(phrase, case=False, na=False)].copy()
        
        if not posts_with_phrase.empty:
            df_clustered_phrase = cached_clustering(posts_with_phrase, eps=0.4, min_samples=2, max_features=max_features, data_source=f"phrase_{phrase}")
            
            # FIX: Call find_coordinated_groups with all required parameters
            coordinated_groups = find_coordinated_groups(df_clustered_phrase, threshold, max_features, time_window_minutes)
            
            if coordinated_groups:
                for group in coordinated_groups:
                    for post in group['posts']:
                        account_id = post['account_id']
                        if account_id not in account_campaigns:
                            account_campaigns[account_id] = set()
                        account_campaigns[account_id].add(phrase)
    
    multi_campaign_accounts = []
    for account, phrases in account_campaigns.items():
        if len(phrases) > 1:
            multi_campaign_accounts.append({
                "Account": account,
                "Campaigns Involved": len(phrases),
                "Phrases Used": ', '.join(sorted(list(phrases)))
            })
    
    return pd.DataFrame(multi_campaign_accounts)

# --- Cached Functions ---
@st.cache_data(show_spinner="🔍 Finding coordinated posts within clusters...")
def cached_find_coordinated_groups(_df, threshold, max_features, time_window_minutes, data_source="default"):
    if _df.empty:
        return []
    return find_coordinated_groups(_df, threshold, max_features, time_window_minutes)

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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analysis", "🌐 Network Graph", "⚠️ Risk & Fundraising"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Humanitarian Campaign Overview")
    st.markdown(f"**Coordination Mode:** `{coordination_mode}` | **Total Posts (All Time):** `{len(df):,}` | **Posts After Filters:** `{len(filtered_df_global):,}`")

    if not filtered_df_global.empty:
        st.markdown("### 📈 Post Volume Over Time")
        filtered_df_global['date_utc'] = pd.to_datetime(filtered_df_global['timestamp_share'], unit='s', utc=True)
        time_series_data = filtered_df_global.set_index('date_utc').resample('D').size().reset_index(name='post_count')
        fig_volume = px.line(time_series_data, x='date_utc', y='post_count', title='Posts per Day (UTC)')
        st.plotly_chart(fig_volume, use_container_width=True)

        st.markdown("### 📋 Raw Data Sample (First 10 Rows)")
        
        # CORRECTED: Display the raw timestamp_share column
        st.dataframe(filtered_df_global[['account_id', 'content_id', 'object_id', 'timestamp_share', 'Platform']].head(10))

        st.markdown("### 📊 Top 10 Platforms by Post Count")
        platform_counts = filtered_df_global['Platform'].value_counts().reset_index()
        platform_counts.columns = ['Platform', 'Post Count']
        fig_platforms = px.bar(platform_counts.head(10), x='Platform', y='Post Count', title='Top Platforms by Posts')
        st.plotly_chart(fig_platforms, use_container_width=True)


# ==================== TAB 2: Analysis ====================
with tab2:
    st.subheader("🔍 Coordinated Content Analysis")
    st.info("This tab identifies clusters of highly similar content that were shared by different accounts within a specified time window.")
    
    # --- Analysis Controls ---
    st.markdown("---")
    st.subheader("Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Renamed for clarity
        similarity_threshold = st.slider(
            "Minimum Cosine Similarity for Coordination",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Higher values require closer text matches (0.0=no match, 1.0=exact match)."
        )
    with col2:
        max_features = st.slider(
            "Max TF-IDF Features",
            min_value=100,
            max_value=5000,
            value=2000,
            step=100,
            help="The vocabulary size for the analysis. A lower value speeds up analysis but may reduce nuance."
        )
    with col3:
        time_window_minutes = st.slider(
            "Max Time Difference (minutes)",
            min_value=1,
            max_value=1440, # 24 hours
            value=120,
            step=10,
            help="Groups posts shared within this time frame. A smaller window improves performance."
        )

    # --- Run Analysis ---
    if st.button("Run Coordinated Content Analysis"):
        if df_for_analysis.empty:
            st.warning("No data to analyze. Please upload and filter the data first.")
        else:
            with st.spinner("⏳ Running coordination analysis..."):
                # Pass all relevant parameters to the cached function
                coordination_groups = cached_find_coordinated_groups(
                    df_for_analysis,
                    similarity_threshold,
                    max_features,
                    time_window_minutes,
                    data_source=f"{coordination_mode}_{len(df_for_analysis)}_{similarity_threshold}_{max_features}_{time_window_minutes}"
                )
            
            st.success(f"✅ Found **{len(coordination_groups):,}** coordinated groups.")
            
            # Display results
            if coordination_groups:
                df_groups = pd.DataFrame([
                    {
                        "Group ID": i,
                        "Type": g["coordination_type"],
                        "Num Posts": g["num_posts"],
                        "Num Accounts": g["num_accounts"],
                        "Max Similarity Score": g["max_similarity_score"],
                        "Post Sample (Top 3)": [p['text'] for p in g['posts'][:3]],
                    } for i, g in enumerate(coordination_groups)
                ])
                
                # Exclude the "Post Sample" column from sorting
                st.dataframe(df_groups.sort_values("Num Posts", ascending=False).drop(columns="Post Sample (Top 3)"))
                
                st.markdown("### Top 5 Coordinated Groups")
                for i, group in enumerate(coordination_groups):
                    if i >= 5:
                        break
                    
                    st.markdown(f"**Group {i+1}** - **{group['coordination_type']}** | Posts: `{group['num_posts']}` | Accounts: `{group['num_accounts']}` | Max Sim: `{group['max_similarity_score']}`")
                    group_df = pd.DataFrame(group['posts'])
                    group_df = group_df.sort_values('Timestamp', ascending=True).reset_index(drop=True)
                    group_df['Timestamp'] = pd.to_datetime(group_df['Timestamp'], unit='s', utc=True)
                    
                    # Display URLs if available
                    if 'URL' in group_df.columns:
                        display_cols = ['account_id', 'Timestamp', 'text', 'Platform', 'URL']
                    else:
                        display_cols = ['account_id', 'Timestamp', 'text', 'Platform']

                    st.dataframe(group_df[display_cols])

# ==================== TAB 3: Network Graph ====================
with tab3:
    st.subheader("🌐 User Interaction Network Graph")
    st.info("This graph shows accounts that shared similar content. Nodes represent accounts, and edges indicate a shared connection (similar content or shared URL).")
    
    st.session_state['max_nodes_to_display'] = st.slider("Max Nodes to Display", min_value=10, max_value=200, value=40, step=10)

    with st.spinner("⏳ Building network graph..."):
        df_for_graph = df_for_analysis.copy()
        
        # We need to run the clustering analysis on the graph dataframe first
        # to ensure the 'cluster' column exists for the text-based graph.
        if coordination_mode == "Text Content":
            df_for_graph = cached_clustering(
                df_for_graph, 
                eps=0.4, 
                min_samples=2, 
                max_features=st.session_state.get('max_features_value', 2000), 
                data_source=f"graph_{coordination_mode}_{len(df_for_graph)}"
            )
            graph_type = "text"
        else:
            graph_type = "url"

        G, pos, cluster_map = cached_network_graph(df_for_graph, coordination_type=graph_type, data_source=f"graph_build_{coordination_mode}_{len(df_for_graph)}")

    if G.nodes():
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            marker=dict(
                color=[cluster_map.get(node, -2) for node in G.nodes()],
                colorscale='Viridis',
                size=[G.degree(node) * 5 + 5 for node in G.nodes()],
                line=dict(width=1, color='DarkSlateGrey'),
            ),
            textposition="bottom center"
        )
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='User Interaction Network',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No network graph could be generated with the current data and parameters.")


# ==================== TAB 4: Risk & Fundraising ====================
with tab4:
    st.subheader("⚠️ Risk Analysis: Campaign Cohesion & Fundraising")
    st.info("This section analyzes accounts that are involved in multiple campaign themes and identifies posts containing fundraising links or keywords.")
    
    st.markdown("### Accounts Involved in Multiple Campaigns")
    with st.spinner("⏳ Analyzing cross-campaign involvement..."):
        # The function now uses the cached clustering and coordinated groups logic internally
        multi_campaign_df = get_account_campaign_involvement(
            df_for_analysis,
            similarity_threshold,
            max_features,
            time_window_minutes
        )

    if not multi_campaign_df.empty:
        st.dataframe(multi_campaign_df)
    else:
        st.info("No accounts were found to be involved in more than one tracked campaign.")
    
    st.markdown("### Posts with Fundraising Keywords/URLs")
    fundraising_posts = df_for_analysis[df_for_analysis['fundraising_urls_in_text'].apply(lambda x: len(x) > 0)].copy()
    
    if not fundraising_posts.empty:
        st.success(f"✅ Found **{len(fundraising_posts):,}** posts with fundraising keywords or URLs.")
        fundraising_posts['Fundraising_URLs'] = fundraising_posts['fundraising_urls_in_text'].apply(lambda urls: ', '.join(urls))
        st.dataframe(fundraising_posts[['account_id', 'Platform', 'object_id', 'Fundraising_URLs', 'timestamp_share']])
    else:
        st.info("No posts containing fundraising keywords or URLs were found in the filtered data.")
