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
        st.warning(f"Column '{col_name_lower}' not found in one of the uploaded files. Filling with NaN.")
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
        om['object_id'] = get_specific_col(openmeasure_df, 'text')
        om['original_url'] = get_specific_col(om, 'created_at')
        om['timestamp_share'] = get_specific_col(om, 'created_at')
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

    df_processed['fundraising_urls_in_text'] = df_processed['object_id'].apply(lambda x: extract_fundraising_urls(x, FUNDRAISING_KEYWORDS))

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

    coordination_groups = []

    if df['original_text'].nunique() > 1 and len(df) > 1:
        df_clustered = df[df['cluster'] != -1].groupby('cluster')
    else:
        df_clustered = df.groupby(pd.Series(range(len(df))))

    for cluster_id, group in df_clustered:
        if len(group) < 2:
            continue

        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col, 'fundraising_urls_in_text']].copy()
        clean_df = clean_df.rename(columns={text_col: 'text', 'timestamp_share': 'Timestamp'})
        clean_df = clean_df.sort_values('Timestamp').reset_index(drop=True)

        adj = {i: [] for i in range(len(clean_df))}

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
            for j in range(i + 1, len(clean_df)):
                post_j_ts = clean_df.loc[j, 'Timestamp']
                time_diff = (post_j_ts - post_i_ts) / 60

                if time_diff > time_window_minutes:
                    break

                cosine_sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])
                if cosine_sim[0, 0] >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()

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
                        group_texts = group_posts['text'].tolist()
                        group_tfidf = vectorizer.transform(group_texts)
                        group_sim_scores = cosine_similarity(group_tfidf)
                        max_sim = group_sim_scores.max() if group_sim_scores.size > 0 else 0.0

                        platforms = group_posts['Platform'].unique()
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

                        coordination_groups.append({
                            "posts": group_posts.to_dict('records'),
                            "num_posts": len(group_posts),
                            "num_accounts": len(group_posts['account_id'].unique()),
                            "max_similarity_score": round(max_sim, 3),
                            "coordination_type": coordination_type
                        })

    return coordination_groups

def build_user_interaction_graph(df, coordination_type="text"):
    G = nx.Graph()
    influencer_column = 'account_id'

    if df.empty or 'account_id' not in df.columns:
        return G, {}, {}

    if coordination_type == "Text Content":
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

    elif coordination_type == "Shared URLs":
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
        if coordination_type == "Text Content":
            clusters = df[df[influencer_column] == inf]['cluster'].dropna()
            G.nodes[inf]['cluster'] = clusters.mode()[0] if not clusters.empty else -2
        elif coordination_type == "Shared URLs":
            shared_urls = df[(df[influencer_column] == inf) & df['URL'].notna() & (df['URL'].str.strip() != '')]['URL'].unique()
            G.nodes[inf]['cluster'] = f"SharedURL_Group_{hash(tuple(sorted(shared_urls))) % 100}" if len(shared_urls) > 0 else "NoSharedURL"

    if G.nodes():
        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
        top_n_nodes = sorted_nodes[:st.session_state.get('max_nodes_to_display', 40)]
        subgraph = G.subgraph(top_n_nodes)

        # Use a different layout for better visual separation
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        cluster_map = {node: G.nodes[node].get('cluster', -2) for node in subgraph.nodes()}
        return subgraph, pos, cluster_map
    else:
        return G, {}, {}

def get_account_campaign_involvement(df_for_analysis, threshold, max_features, time_window_minutes):
    """
    Identifies accounts involved in coordinated groups across multiple campaign phrases.
    """
    account_campaigns = {}
    for phrase in PHRASES_TO_TRACK:
        posts_with_phrase = df_for_analysis[df_for_analysis['object_id'].str.contains(phrase, case=False, na=False)].copy()

        if not posts_with_phrase.empty:
            df_clustered_phrase = cached_clustering(posts_with_phrase, eps=st.session_state.get('eps', 0.4), min_samples=st.session_state.get('min_samples', 3), max_features=max_features, data_source=f"phrase_{phrase}")

            coordinated_groups = cached_find_coordinated_groups(df_clustered_phrase, threshold, max_features, time_window_minutes, data_source=f"phrase_{phrase}")

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

def get_top_fundraising_urls(coordinated_groups):
    """
    Extracts and counts unique fundraising URLs from a list of coordinated groups.
    """
    url_counts = Counter()
    for group in coordinated_groups:
        for post in group['posts']:
            # The 'fundraising_urls_in_text' is a list of URLs
            if 'fundraising_urls_in_text' in post and post['fundraising_urls_in_text']:
                for url in post['fundraising_urls_in_text']:
                    url_counts[url] += 1

    if not url_counts:
        return pd.DataFrame()

    df_urls = pd.DataFrame(url_counts.items(), columns=['URL', 'Times Shared in Coordinated Posts'])
    df_urls['Domain'] = df_urls['URL'].apply(lambda x: tldextract.extract(x).domain + '.' + tldextract.extract(x).suffix)
    df_urls = df_urls.sort_values('Times Shared in Coordinated Posts', ascending=False)

    return df_urls

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

# New function to read and combine fundraising data
def read_fundraising_data(uploaded_files):
    if not uploaded_files:
        return pd.DataFrame()

    # Define the list of correct column names, excluding the 'Error' column
    columns_to_load = ['Donation Link', 'Title', 'Amount Raised', 'Target Amount', 'Currency', 'Creator', 'Location']

    dfs = []
    for i, file in enumerate(uploaded_files):
        try:
            # Read the CSV file, loading only the specified columns and handling encoding
            bytes_data = file.getvalue()
            decoded_content = bytes_data.decode('utf-8-sig', errors='ignore')
            df = pd.read_csv(StringIO(decoded_content), usecols=columns_to_load)
            st.sidebar.success(f"✅ Fundraising File {i+1}: Loaded {len(df)} rows")

            # Normalization steps
            # Fill missing numerical values with 0
            df['Amount Raised'] = df['Amount Raised'].fillna(0)
            df['Target Amount'] = df['Target Amount'].fillna(0)

            # Fill missing categorical values with 'Unknown'
            df['Title'] = df['Title'].fillna('Unknown')
            df['Currency'] = df['Currency'].fillna('Unknown')
            df['Location'] = df['Location'].fillna('Unknown')
            df['Creator'] = df['Creator'].fillna('Unknown')

            dfs.append(df)

        except Exception as e:
            st.error(f"Error reading fundraising file {i+1}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.columns = combined_df.columns.str.lower().str.replace(' ', '_')

    # Ensure numerical columns are correctly typed
    for col in ['amount_raised', 'target_amount']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

    # Remove duplicate rows
    combined_df.drop_duplicates(inplace=True)

    return combined_df

# --- Sidebar: Data Source & Coordination Mode ---
st.sidebar.header("📥 Data Upload")

# Coordination Mode Selector
st.sidebar.header("🎯 Coordination Analysis Mode")
coordination_mode = st.sidebar.radio(
    "Analyze coordination by:",
    ("Text Content", "Shared URLs"),
    help="Choose what defines a coordinated action: similar text messages or sharing the same external link."
)

if 'last_coordination_mode' not in st.session_state or st.session_state.last_coordination_mode != coordination_mode:
    st.cache_data.clear()
    st.session_state.last_coordination_mode = coordination_mode

# Store coordination mode in session state
st.session_state.coordination_mode = coordination_mode

combined_raw_df = pd.DataFrame()

st.sidebar.info("Upload your CSV files below.")
uploaded_meltwater_files = st.sidebar.file_uploader("Upload Meltwater CSV(s)", type=["csv"], accept_multiple_files=True, key="meltwater_upload")
uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

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

with st.spinner("⏳ Preprocessing and mapping combined data..."):
    df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)

if df.empty:
    st.warning("No valid data after final preprocessing. Please adjust the filters or check your data source.")
    st.stop()

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

# Store final filtered data in session state for other tabs
st.session_state.df_for_analysis = df_for_analysis

if df_for_analysis.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


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
    st.sidebar.warning("Could not create downloadable dataset with core columns. Please ensure your uploaded data contains these columns.")

# New Fundraising Data Uploader
st.sidebar.markdown("---")
st.sidebar.header("💰 Fundraising Data")
uploaded_fundraising_files = st.sidebar.file_uploader("Upload Fundraising CSV(s)", type=["csv"], accept_multiple_files=True, key="fundraising_upload")
if uploaded_fundraising_files:
    fundraising_df = read_fundraising_data(uploaded_fundraising_files)
    st.session_state.fundraising_df = fundraising_df
    if not fundraising_df.empty:
        st.sidebar.success(f"✅ Loaded {len(fundraising_df):,} fundraising campaigns.")
    else:
        st.sidebar.error("❌ Failed to load fundraising data.")

# --- Main App Tabs ---
tab_coordination, tab_networks, tab_phrases, tab_fundraising = st.tabs([
    "🤝 Coordination Analysis",
    "🕸️ Network Graph",
    "📢 Phrase Monitoring",
    "💰 Fundraising Campaigns"
])

# --- Coordination Analysis Tab ---
with tab_coordination:
    st.header("Coordinated Activity Dashboard")
    st.info("This tab identifies clusters of posts that are highly similar and were shared within a short time frame, suggesting coordinated activity.")

    st.subheader("⚙️ Analysis Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        eps_value = st.slider(
            'Text Similarity Threshold (Epsilon)',
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.get('eps', 0.4),
            step=0.05,
            help="Lower values mean higher similarity is required to form a cluster."
        )
        st.session_state.eps = eps_value
    with col2:
        min_samples_value = st.slider(
            'Minimum Posts per Cluster',
            min_value=2,
            max_value=20,
            value=st.session_state.get('min_samples', 3),
            step=1,
            help="The minimum number of similar posts required to form a cluster."
        )
        st.session_state.min_samples = min_samples_value
    with col3:
        time_window_minutes = st.slider(
            'Time Window for Coordination (minutes)',
            min_value=1,
            max_value=120,
            value=st.session_state.get('time_window', 30),
            step=5,
            help="The maximum time difference between posts to be considered a coordinated group."
        )
        st.session_state.time_window = time_window_minutes

    df_clustered = cached_clustering(
        st.session_state.df_for_analysis,
        eps=eps_value,
        min_samples=min_samples_value,
        max_features=2000,
        data_source=f"coord_analysis_df_eps_{eps_value}_minsamples_{min_samples_value}_maxfeats_{2000}_mode_{coordination_mode}"
    )

    if not df_clustered.empty:
        total_clusters = df_clustered['cluster'].nunique()
        noise_points = (df_clustered['cluster'] == -1).sum()
        non_noise_clusters = total_clusters - (1 if -1 in df_clustered['cluster'].unique() else 0)
        st.write(f"**Clustering Results:** `{non_noise_clusters}` clusters found (excluding `{noise_points}` noise posts).")

        coordinated_groups = cached_find_coordinated_groups(
            df_clustered,
            threshold=0.8,
            max_features=2000,
            time_window_minutes=time_window_minutes,
            data_source=f"coord_groups_df_thresh_0.8_time_{time_window_minutes}_maxfeats_{2000}_mode_{coordination_mode}"
        )

        st.subheader("Summary of Coordinated Activity")
        if not coordinated_groups:
            st.info("No coordinated groups found with the current parameters. Try adjusting the sliders.")
        else:
            total_posts_in_groups = sum(g['num_posts'] for g in coordinated_groups)
            total_accounts_in_groups = sum(g['num_accounts'] for g in coordinated_groups)
            avg_posts_per_group = np.mean([g['num_posts'] for g in coordinated_groups])
            avg_accounts_per_group = np.mean([g['num_accounts'] for g in coordinated_groups])

            summary_df = pd.DataFrame([
                {"Metric": "Total Coordinated Groups Found", "Value": len(coordinated_groups)},
                {"Metric": "Total Posts in Groups", "Value": total_posts_in_groups},
                {"Metric": "Total Unique Accounts in Groups", "Value": len(set(p['account_id'] for g in coordinated_groups for p in g['posts']))},
                {"Metric": "Avg Posts per Group", "Value": f"{avg_posts_per_group:.2f}"},
                {"Metric": "Avg Accounts per Group", "Value": f"{avg_accounts_per_group:.2f}"},
            ])
            st.table(summary_df.set_index('Metric'))

            st.subheader("Coordinated Group Breakdown")
            group_details = []
            for i, group in enumerate(sorted(coordinated_groups, key=lambda x: x['num_posts'], reverse=True)):
                group_details.append({
                    "Group ID": i + 1,
                    "Coordination Type": group['coordination_type'],
                    "Number of Posts": group['num_posts'],
                    "Number of Unique Accounts": group['num_accounts'],
                    "Max Similarity Score": group['max_similarity_score'],
                    "Top 3 Phrases": ', '.join(list(set([re.search(r'\b(?!the|a|an)\w+\b', p['text'])[0] for p in group['posts'][:3] if re.search(r'\b(?!the|a|an)\w+\b', p['text'])]))) if len(group['posts']) > 0 else 'N/A'
                })

            group_df = pd.DataFrame(group_details)
            st.dataframe(group_df, use_container_width=True)

            st.subheader("Top Fundraising URLs in Coordinated Posts")
            top_urls_df = get_top_fundraising_urls(coordinated_groups)
            if not top_urls_df.empty:
                st.dataframe(top_urls_df.head(10), use_container_width=True)
            else:
                st.info("No fundraising URLs found in coordinated posts.")

            st.subheader("Detailed View of Top Coordinated Groups")
            for i, group in enumerate(coordinated_groups):
                if i >= 5: break
                with st.expander(f"Group {i+1}: {group['coordination_type']} ({group['num_posts']} posts, {group['num_accounts']} accounts)"):
                    group_df_display = pd.DataFrame(group['posts'])
                    st.dataframe(group_df_display, use_container_width=True)


# --- Network Graph Tab ---
with tab_networks:
    st.header("User Interaction Network")
    st.info("This graph visualizes how users are connected based on their participation in coordinated groups (by text or shared URLs). A larger node indicates a higher number of connections.")

    st.subheader("⚙️ Graph Parameters")
    col1, col2 = st.columns(2)
    with col1:
        max_nodes_to_display = st.slider("Max Nodes to Display", min_value=10, max_value=200, value=st.session_state.get('max_nodes_to_display', 40))
        st.session_state.max_nodes_to_display = max_nodes_to_display
    with col2:
        if coordination_mode == "Text Content":
            st.write("Node color represents the cluster ID.")
        elif coordination_mode == "Shared URLs":
            st.write("Node color represents the platform.")

    G, pos, cluster_map = cached_network_graph(st.session_state.df_for_analysis, coordination_type=coordination_mode, data_source=f"network_graph_mode_{coordination_mode}_nodes_{max_nodes_to_display}")
    
    if G.nodes:
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_text = []
        node_size = []
        if G.nodes():
            degrees = dict(G.degree())
            max_degree = max(degrees.values()) if degrees else 1
            node_size = [20 * degrees.get(node, 0) / max_degree + 5 for node in G.nodes()]
            
            for node in G.nodes():
                text = f"User: {node}<br>Degree: {degrees.get(node, 0)}"
                if coordination_mode == "Text Content":
                    text += f"<br>Cluster: {cluster_map.get(node, 'N/A')}"
                else:
                    text += f"<br>Platform: {G.nodes[node]['platform']}"
                node_text.append(text)

        if coordination_mode == "Text Content":
            node_colors = [cluster_map.get(node, -2) for node in G.nodes()]
            color_label = 'Cluster ID'
        else: # Shared URLs
            node_colors = [G.nodes[node]['platform'] for node in G.nodes()]
            color_label = 'Platform'
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_colors,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title=color_label,
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>User Interaction Network Graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 ) ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No network connections found. This could be because no posts met the coordination criteria.")

# --- Phrase Monitoring Tab ---
with tab_phrases:
    st.header("Campaign Phrase Monitoring")
    st.info("This tab tracks the usage of specific campaign phrases over time.")

    phrase_counts = {phrase: (st.session_state.df_for_analysis['object_id'].str.contains(phrase, case=False, na=False)).sum() for phrase in PHRASES_TO_TRACK}
    phrase_counts_df = pd.DataFrame(phrase_counts.items(), columns=["Phrase", "Count"])
    st.dataframe(phrase_counts_df.sort_values("Count", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Phrase Frequency Over Time")
    df_phrase_counts = st.session_state.df_for_analysis.copy()
    for phrase in PHRASES_TO_TRACK:
        df_phrase_counts[phrase] = df_phrase_counts['object_id'].str.contains(phrase, case=False, na=False)
    
    df_phrase_counts['date'] = pd.to_datetime(df_phrase_counts['timestamp_share'], unit='s').dt.date
    df_daily = df_phrase_counts.groupby('date')[PHRASES_TO_TRACK].sum().reset_index()
    df_daily_melt = df_daily.melt(id_vars='date', var_name='Phrase', value_name='Count')

    if not df_daily_melt.empty:
        fig = px.line(df_daily_melt, x='date', y='Count', color='Phrase',
                      title='Daily Usage of Campaign Phrases',
                      labels={'date': 'Date', 'Count': 'Number of Posts'})
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Posts", legend_title="Phrase")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No campaign phrases were found in the selected data.")

    st.subheader("Accounts Involved in Multiple Campaigns")
    multi_campaign_accounts_df = get_account_campaign_involvement(
        st.session_state.df_for_analysis,
        threshold=0.8,
        max_features=2000,
        time_window_minutes=st.session_state.get('time_window', 30)
    )

    if not multi_campaign_accounts_df.empty:
        st.dataframe(multi_campaign_accounts_df.sort_values("Campaigns Involved", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No accounts were found to be involved in multiple campaigns.")


# --- Fundraising Campaigns Tab ---
with tab_fundraising:
    st.header("Fundraising Campaign Dashboard")
    st.info("This tab analyzes fundraising campaigns uploaded separately, providing insights into their performance and characteristics.")

    if 'fundraising_df' not in st.session_state or st.session_state.fundraising_df.empty:
        st.warning("Please upload fundraising data in the sidebar to view this dashboard.")
    else:
        fund_df = st.session_state.fundraising_df

        st.subheader("Top Fundraising Campaigns by Amount Raised")
        top_campaigns = fund_df.sort_values('amount_raised', ascending=False).head(10)
        st.dataframe(top_campaigns, use_container_width=True)

        st.subheader("Overall Fundraising Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_campaigns = len(fund_df)
            st.metric("Total Campaigns", f"{total_campaigns:,}")
        with col2:
            total_raised = fund_df['amount_raised'].sum()
            st.metric("Total Amount Raised", f"${total_raised:,.2f}")
        with col3:
            total_target = fund_df['target_amount'].sum()
            st.metric("Total Target Amount", f"${total_target:,.2f}")

        st.subheader("Campaigns by Currency")
        currency_counts = fund_df.groupby('currency')['donation_link'].count().sort_values(ascending=False)
        fig_currency = px.bar(
            x=currency_counts.index,
            y=currency_counts.values,
            labels={'x': 'Currency', 'y': 'Number of Campaigns'},
            title='Number of Campaigns by Currency'
        )
        st.plotly_chart(fig_currency, use_container_width=True)

        st.subheader("Amount Raised vs. Target Amount")
        fig_scatter = px.scatter(
            fund_df,
            x='target_amount',
            y='amount_raised',
            color='currency',
            hover_data=['title', 'creator'],
            labels={'target_amount': 'Target Amount', 'amount_raised': 'Amount Raised'},
            title='Amount Raised vs. Target Amount'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
