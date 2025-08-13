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
from io import BytesIO, StringIO
from collections import Counter
import tldextract

# --- Set Page Config ---
st.set_page_config(page_title="Humanitarian Campaign CIB Network Monitoring", layout="wide")
st.title("🕊️ Humanitarian Campaign Monitoring Dashboard")

# --- Define the 6 key phrases to track ---
PHRASES_TO_TRACK = [
    "If you're scrolling, PLEASE leave a dot", "I'm so hungry, I'm not ashamed to say that", "3 replies — even dots — can break the algorithm", "My body is slowly falling apart from malnutrition, dizziness, and weight loss", "Good bye. If we die, don't forget us", "If you see this reply with a dot"
]

# --- GitHub URL for default data ---
GITHUB_DEFAULT_DATA_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/data/default_data.csv"

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
    date_formats = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p', '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p', '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y']
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

def final_preprocess_and_map_columns(df):
    df_processed = df.copy()
    if df_processed.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Map raw columns to standardized names for internal use
    df_processed = df_processed.rename(columns={
        'tweet id': 'content_id', 'influencer': 'account_id', 'hit sentence': 'object_id',
        'date': 'timestamp_share', 'text': 'object_id', 'url': 'URL'
    }, errors='ignore')
    
    for col in ['account_id', 'content_id', 'object_id', 'timestamp_share', 'URL']:
        if col not in df_processed.columns: df_processed[col] = np.nan
    
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed['account_id'] = df_processed['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    df_processed['URL'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')
    df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    df_processed['timestamp_share'] = df_processed['timestamp_share'].apply(parse_timestamp_robust)
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed = df_processed.dropna(subset=['timestamp_share', 'account_id', 'original_text']).copy()
    df_processed['extracted_urls'] = df_processed['object_id'].apply(extract_all_urls)
    
    df_processed['has_phrase'] = df_processed['original_text'].apply(lambda x: any(phrase in x for phrase in PHRASES_TO_TRACK))
    df_processed = df_processed[df_processed['has_phrase'] == True].reset_index(drop=True)

    core_columns = ['account_id', 'content_id', 'object_id', 'original_text', 'timestamp_share', 'Platform', 'extracted_urls']
    other_columns = [col for col in df_processed.columns if col not in core_columns and col != 'has_phrase']
    
    core_df = df_processed[core_columns].copy()
    other_df = df_processed[['content_id'] + other_columns].copy()
    
    return core_df, other_df

def find_fundraising_campaigns(df, coordinated_groups_df):
    if df.empty: return pd.DataFrame()
    fundraising_domains = ['gofundme.com', 'paypal.com', 'justgiving.com', 'donorbox.org', 'charity.com', 'charitynavigator.org', 'redcross.org', 'unicef.org', 'doctorswithoutborders.org', 'icrc.org']
    urls_df = df.explode('extracted_urls').dropna(subset=['extracted_urls']).copy()
    urls_df.rename(columns={'extracted_urls': 'Fundraising Link'}, inplace=True)
    urls_df = urls_df.drop_duplicates(subset=['content_id', 'Fundraising Link'])
    if urls_df.empty: return pd.DataFrame()
    def get_domain(url):
        ext = tldextract.extract(url)
        if ext.domain == 'paypal' and ext.suffix == 'me': return 'paypal.me'
        return ext.domain + '.' + ext.suffix
    urls_df['domain'] = urls_df['Fundraising Link'].apply(get_domain)
    fundraising_links_df = urls_df[urls_df['domain'].isin(fundraising_domains) | (urls_df['domain'] == 'paypal.me')].copy()
    if fundraising_links_df.empty: return pd.DataFrame()
    campaign_summary_df = fundraising_links_df.groupby('Fundraising Link').agg(
        Num_Posts=('content_id', 'size'), Num_Unique_Accounts=('account_id', 'nunique'),
        First_Shared=('timestamp_share', 'min'), Last_Shared=('timestamp_share', 'max')
    ).reset_index()
    campaign_summary_df['First_Shared'] = pd.to_datetime(campaign_summary_df['First_Shared'], unit='s', utc=True)
    campaign_summary_df['Last_Shared'] = pd.to_datetime(campaign_summary_df['Last_Shared'], unit='s', utc=True)
    campaign_summary_df['Coordination_Score'] = 0.0
    campaign_summary_df['Risk_Flag'] = 'Low'
    if not coordinated_groups_df.empty:
        coordinated_groups_df['Account ID'] = coordinated_groups_df['Account ID'].astype(str)
        account_coordination_scores = coordinated_groups_df.groupby('Account ID')['group_size'].transform('max').fillna(0)
        account_coordination_mapping = dict(zip(coordinated_groups_df['Account ID'], account_coordination_scores))
        def calculate_link_score(group):
            accounts_sharing_link = group['account_id'].unique()
            scores = [account_coordination_mapping.get(acc, 0) for acc in accounts_sharing_link]
            if not scores: return 0
            return np.max(scores)
        link_scores = fundraising_links_df.groupby('Fundraising Link').apply(calculate_link_score)
        campaign_summary_df = campaign_summary_df.set_index('Fundraising Link')
        campaign_summary_df['Coordination_Score'] = link_scores.reindex(campaign_summary_df.index, fill_value=0)
        campaign_summary_df = campaign_summary_df.reset_index()
    campaign_summary_df['Risk_Flag'] = np.where(campaign_summary_df['Coordination_Score'] > campaign_summary_df['Coordination_Score'].quantile(0.8), 'High', 'Low')
    campaign_summary_df['Risk_Flag'] = np.where((campaign_summary_df['Num_Unique_Accounts'] < 5) & (campaign_summary_df['Num_Posts'] > 20), 'Needs Review', campaign_summary_df['Risk_Flag'])
    return campaign_summary_df[['Fundraising Link', 'Num_Posts', 'Num_Unique_Accounts', 'First_Shared', 'Last_Shared', 'Coordination_Score', 'Risk_Flag']]

def cluster_texts(df, eps, min_samples, max_features, mode):
    if mode == 'URL Mode':
        df_for_clustering = df.explode('extracted_urls').dropna(subset=['extracted_urls']).copy()
        if 'extracted_urls' not in df_for_clustering.columns or df_for_clustering['extracted_urls'].nunique() <= 1:
            df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
        texts_to_cluster = df_for_clustering['extracted_urls'].astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=max_features, analyzer='char_wb', ngram_range=(2, 5))
    else: # Text Mode
        if 'original_text' not in df.columns or df['original_text'].nunique() <= 1:
            df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
        texts_to_cluster = df['original_text'].astype(str).tolist()
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

    if not texts_to_cluster or all(t.strip() == "" for t in texts_to_cluster):
        df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
        
    try: tfidf_matrix = vectorizer.fit_transform(texts_to_cluster)
    except ValueError as e: df_copy = df.copy(); df_copy['cluster'] = -1; return df_copy
    eps = max(0.01, min(0.99, eps)); clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(tfidf_matrix)
    
    if mode == 'URL Mode':
        df_clustered_urls = df_for_clustering.copy()
        df_clustered_urls['cluster'] = clustering.labels_
        df_copy = df.copy()
        df_copy['cluster'] = -1
        for url_cluster, group in df_clustered_urls.groupby('cluster'):
            if url_cluster != -1:
                post_ids_in_cluster = group['content_id'].unique()
                df_copy.loc[df_copy['content_id'].isin(post_ids_in_cluster), 'cluster'] = url_cluster
        return df_copy
    else: # Text Mode
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
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col]].copy().reset_index(drop=True)
        if mode == 'URL Mode':
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=max_features)
        else:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3, 5), max_features=max_features)
        try: tfidf_matrix = vectorizer.fit_transform(clean_df[text_col])
        except Exception: continue
        cosine_sim = cosine_similarity(tfidf_matrix); adj = {i: [] for i in range(len(clean_df))}
        for i in range(len(clean_df)):
            for j in range(i + 1, len(clean_df)):
                if cosine_sim[i, j] >= threshold: adj[i].append(j); adj[j].append(i)
        visited = set()
        for i in range(len(clean_df)):
            if i not in visited:
                group_indices = []; q = [i]; visited.add(i)
                while q:
                    u = q.pop(0); group_indices.append(u)
                    for v in adj[u]:
                        if v not in visited: visited.add(v); q.append(v)
                if len(group_indices) > 1:
                    group_posts = clean_df.iloc[group_indices].copy()
                    if len(group_posts['account_id'].unique()) > 1:
                        group_sim_scores = cosine_sim[np.ix_(group_indices, group_indices)]
                        max_sim = group_sim_scores.max() if group_sim_scores.size > 0 else 0.0
                        posts_data = group_posts.rename(columns={'account_id': 'Account ID', 'Platform': 'Platform', 'timestamp_share': 'Timestamp', text_col: 'Text', 'URL': 'URL'}).to_dict('records')
                        coordination_groups.append({"posts": posts_data,"num_posts": len(posts_data),"num_accounts": len(group_posts['account_id'].unique()),"max_similarity_score": round(max_sim, 3)})
    return sorted(coordination_groups, key=lambda x: x['num_posts'], reverse=True)

def build_user_interaction_graph(df):
    G = nx.Graph(); influencer_column = 'account_id'
    if 'cluster' not in df.columns: return G, {}, {}
    grouped = df.groupby('cluster')
    for cluster_id, group in grouped:
        if cluster_id == -1 or len(group[influencer_column].unique()) < 2: continue
        users_in_cluster = group[influencer_column].dropna().unique().tolist()
        for u1, u2 in combinations(users_in_cluster, 2):
            if G.has_edge(u1, u2): G[u1][u2]['weight'] += 1
            else: G.add_edge(u1, u2, weight=1)
    all_influencers = df[influencer_column].dropna().unique().tolist()
    influencer_platform_map = df.groupby(influencer_column)['Platform'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()
    for inf in all_influencers:
        if inf not in G.nodes(): G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')
        clusters = df[df[influencer_column] == inf]['cluster'].dropna()
        G.nodes[inf]['cluster'] = clusters.mode()[0] if not clusters.empty else -2
    if G.nodes():
        node_degrees = dict(G.degree()); sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
        top_n_nodes = sorted_nodes[:st.session_state.get('max_nodes_to_display', 40)]
        subgraph = G.subgraph(top_n_nodes); pos = nx.kamada_kawai_layout(subgraph)
        cluster_map = {node: G.nodes[node].get('cluster', -2) for node in subgraph.nodes()}
        return subgraph, pos, cluster_map
    else: return G, {}, {}

# --- Cached Functions ---
@st.cache_data(show_spinner="🔍 Finding coordinated posts within clusters...")
def cached_find_coordinated_groups(_df, threshold, max_features, mode): return find_coordinated_groups(_df, threshold, max_features, mode)
@st.cache_data(show_spinner="🧩 Clustering content...")
def cached_clustering(_df, eps, min_samples, max_features, mode): return cluster_texts(_df, eps, min_samples, max_features, mode)
@st.cache_data(show_spinner="🕸️ Building network graph...")
def cached_network_graph(_df_for_graph): return build_user_interaction_graph(_df_for_graph)
@st.cache_data(show_spinner="🔗 Identifying fundraising campaigns...")
def cached_find_fundraising_campaigns(df, coordinated_groups_df): return find_fundraising_campaigns(df, coordinated_groups_df)

# --- Main Dashboard Logic ---
st.sidebar.header("📥 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"], help="Upload a single CSV file containing your campaign data. It should have columns for 'account_id', 'object_id' (the text content), 'timestamp_share', and 'URL'.")

df_raw = pd.DataFrame()
core_df, other_df = pd.DataFrame(), pd.DataFrame()

if uploaded_file:
    with st.spinner("⏳ Loading and preprocessing uploaded data..."):
        try:
            df_raw = pd.read_csv(uploaded_file, low_memory=False)
            core_df, other_df = final_preprocess_and_map_columns(df_raw)
            if core_df.empty: st.error("❌ The uploaded file is empty or missing required columns after preprocessing, or no posts with the specified phrases were found.")
            else: st.sidebar.success(f"✅ Loaded {len(core_df)} posts for analysis.")
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")
else:
    st.info("No file uploaded. Loading default sample data.")
    try:
        # Attempt to read default data from GitHub
        df_raw = pd.read_csv(GITHUB_DEFAULT_DATA_URL, low_memory=False)
        core_df, other_df = final_preprocess_and_map_columns(df_raw)
        st.sidebar.info(f"✅ Loaded {len(core_df)} posts from GitHub default data.")
    except Exception as e:
        st.warning(f"⚠️ Could not load data from GitHub URL. Generating a dummy dataset instead. Error: {e}")
        # Generate a dummy dataset for demonstration fallback
        data_points = 5000
        start_date = pd.Timestamp('2025-06-01', tz='UTC')
        end_date = pd.Timestamp('2025-08-13', tz='UTC')
        timestamps = np.random.randint(start_date.timestamp(), end_date.timestamp(), data_points)
        accounts = [f'user_{i}' for i in np.random.randint(1, 1000, data_points)]
        phrases = np.random.choice(PHRASES_TO_TRACK, data_points, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
        texts = [f"{phrase} and some other words." for phrase in phrases]
        urls = np.random.choice(['https://gofundme.com/campaign1', 'https://paypal.me/campaign2', 'https://gofundme.com/campaign3', 'https://google.com/search?q=gaza'], data_points)
        default_data = {
            'account_id': accounts, 'content_id': range(data_points), 'timestamp_share': timestamps,
            'object_id': texts, 'URL': urls, 'engagement_rate': np.random.rand(data_points) * 10,
            'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], data_points)
        }
        df_raw = pd.DataFrame(default_data)
        core_df, other_df = final_preprocess_and_map_columns(df_raw)
        st.sidebar.info(f"✅ Loaded {len(core_df)} posts from generated sample data.")

if not core_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Global Filters")
    min_ts, max_ts = core_df['timestamp_share'].min(), core_df['timestamp_share'].max()
    min_date = pd.to_datetime(min_ts, unit='s').date() if pd.notna(min_ts) else pd.Timestamp.now().date()
    max_date = pd.to_datetime(max_ts, unit='s').date() if pd.notna(max_ts) else pd.Timestamp.now().date()
    selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(selected_date_range) == 2:
        start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
        end_ts = int((pd.Timestamp(selected_date_range[1], tz='UTC') + timedelta(days=1) - timedelta(microseconds=1)).timestamp())
    else: start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp()); end_ts = start_ts + 86400 - 1
    
    filtered_df = core_df[(core_df['timestamp_share'] >= start_ts) & (core_df['timestamp_share'] <= end_ts)].copy()
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏩ Performance Controls")
    max_posts_for_analysis = st.sidebar.number_input("Limit Posts for Analysis (0 for all)", min_value=0, value=50000, step=1000, help="To speed up analysis on large datasets, enter a number to process a random sample of posts.")
    if max_posts_for_analysis > 0 and len(filtered_df) > max_posts_for_analysis:
        df_for_analysis = filtered_df.sample(n=max_posts_for_analysis, random_state=42).copy()
        st.sidebar.warning(f"⚠️ Analyzing a random sample of **{len(df_for_analysis):,}** posts.")
    else:
        df_for_analysis = filtered_df.copy()
        st.sidebar.info(f"✅ Analyzing all **{len(df_for_analysis):,}** posts.")
    
    analysis_mode = st.sidebar.radio("Analysis Mode", ("Text Mode", "URL Mode"))

else: st.info("Please upload your CSV file or use the default data."); df_for_analysis = pd.DataFrame()

# --- Tabs ---
tab0, tab1, tab2, tab3 = st.tabs(["📝 Raw Data", "❤️ Campaign Pulse", "🕸️ CIB Network", "💰 Fundraising & Risk"])

# ==================== TAB 0: Raw Data ====================
with tab0:
    st.subheader("Data Preview and Download")
    if not core_df.empty:
        st.markdown("### 🧹 Preprocessed Core Columns")
        st.dataframe(core_df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ⬇️ Download Core Data for Analysis")
        
        csv_download_df = core_df[['account_id', 'content_id', 'object_id', 'timestamp_share']].copy()
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
            
        csv = convert_df_to_csv(csv_download_df)
        
        st.download_button(
            label="Download Core Columns as CSV",
            data=csv,
            file_name=f"humanitarian_campaign_core_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download a CSV file containing the core columns for external analysis."
        )

# ==================== TAB 1: Campaign Pulse ====================
with tab1:
    st.subheader("Campaign Performance at a Glance")
    if not df_for_analysis.empty:
        if analysis_mode == "URL Mode":
            total_posts = len(df_for_analysis)
            unique_accounts = df_for_analysis['account_id'].nunique()
            start_date = pd.to_datetime(df_for_analysis['timestamp_share'].min(), unit='s', utc=True).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(df_for_analysis['timestamp_share'].max(), unit='s', utc=True).strftime('%Y-%m-%d')
            st.markdown(f"**Total Posts:** {total_posts:,} | **Unique Accounts:** {unique_accounts:,} | **Date Range:** {start_date} to {end_date}")
            
            all_urls = df_for_analysis['extracted_urls'].explode().dropna()
            if not all_urls.empty:
                url_counts = all_urls.value_counts().head(10)
                fig_urls = px.bar(url_counts, title="Top 10 Shared URLs", labels={'value': 'Shares', 'index': 'URL'})
                st.plotly_chart(fig_urls, use_container_width=True)
            else:
                st.warning("No URLs found in the posts for this mode.")

        else: # Text Mode
            total_posts = len(df_for_analysis)
            unique_accounts = df_for_analysis['account_id'].nunique()
            start_date = pd.to_datetime(df_for_analysis['timestamp_share'].min(), unit='s', utc=True).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(df_for_analysis['timestamp_share'].max(), unit='s', utc=True).strftime('%Y-%m-%d')
            st.markdown(f"**Total Posts:** {total_posts:,} | **Unique Accounts:** {unique_accounts:,} | **Date Range:** {start_date} to {end_date}")
            plot_df = df_for_analysis.copy()
            plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
            phrase_mentions = pd.DataFrame(columns=['datetime', 'phrase', 'count'])
            for phrase in PHRASES_TO_TRACK:
                phrase_df = plot_df[plot_df['original_text'].str.contains(phrase, case=False, na=False)].copy()
                time_series = phrase_df.set_index('datetime').resample('D').size().rename('count')
                temp_df = time_series.reset_index()
                temp_df['phrase'] = phrase
                phrase_mentions = pd.concat([phrase_mentions, temp_df], ignore_index=True)
            fig_phrases = px.area(phrase_mentions, x='datetime', y='count', color='phrase', title="Daily Mentions of Key Phrases", labels={'count': 'Mentions', 'datetime': 'Date', 'phrase': 'Phrase'}, markers=True)
            st.plotly_chart(fig_phrases, use_container_width=True)
            top_accounts = df_for_analysis['account_id'].value_counts().head(10)
            fig_accounts = px.bar(top_accounts, title="Top 10 Most Active Accounts", labels={'value': 'Posts', 'index': 'Account'})
            st.plotly_chart(fig_accounts, use_container_width=True)
            df_for_analysis['hashtags'] = df_for_analysis['original_text'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
            all_hashtags = [tag for tags_list in df_for_analysis['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
            if all_hashtags:
                hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags", labels={'value': 'Frequency', 'index': 'Hashtag'})
                st.plotly_chart(fig_ht, use_container_width=True)

# ==================== TAB 2: CIB Network ====================
with tab2:
    st.subheader("Coordinated Amplification & Network Analysis")
    st.markdown("This section detects coordinated behavior by identifying highly similar posts from different accounts.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ CIB Detection Tuning")
    eps = st.sidebar.slider("Cluster Similarity Threshold (eps)", min_value=0.1, max_value=1.0, value=0.3, step=0.05, help="Lower values create more, smaller clusters of very similar content.")
    min_samples = st.sidebar.slider("Min Posts per Cluster", min_value=2, max_value=10, value=2, step=1)
    max_features = st.sidebar.slider("TF-IDF Max Features", min_value=1000, max_value=10000, value=5000, step=1000)
    threshold_sim = st.slider("Pairwise Similarity Threshold", min_value=0.75, max_value=0.99, value=0.95, step=0.01)
    
    clustered_df = pd.DataFrame(); coordinated_groups_df = pd.DataFrame()
    if not df_for_analysis.empty:
        with st.spinner("🚀 Analyzing for coordinated activity..."):
            clustered_df = cached_clustering(df_for_analysis, eps=eps, min_samples=min_samples, max_features=max_features, mode=analysis_mode)
            coordinated_groups = cached_find_coordinated_groups(clustered_df, threshold=threshold_sim, max_features=max_features, mode=analysis_mode)
        if coordinated_groups:
            st.info(f"✅ Found {len(coordinated_groups)} groups of posts with similarity score ≥ {threshold_sim:.2f}.")
            flat_groups = [{'group_id': i, 'group_size': g['num_posts'], 'group_accounts': g['num_accounts'], 'max_similarity': g['max_similarity_score'], **p} for i, g in enumerate(coordinated_groups) for p in g['posts']]
            coordinated_groups_df = pd.DataFrame(flat_groups)
            for i, group in enumerate(coordinated_groups[:10]):
                st.markdown(f"**Group {i+1}** | **Posts:** {group['num_posts']} | **Accounts:** {group['num_accounts']} | **Max Sim:** {group['max_similarity_score']}")
                posts_df = pd.DataFrame(group['posts'])
                posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s', utc=True)
                st.dataframe(posts_df, use_container_width=True)
                st.markdown("---")
        else: st.warning("No coordinated content found above the selected threshold.")
        st.subheader("Network of Coordinated Accounts")
        st.markdown("This graph shows accounts connected by their participation in a coordinated post group.")
        st.session_state.max_nodes_to_display = st.slider("Max Nodes to Display", min_value=10, max_value=200, value=40, step=10, help="Limit to the most active accounts to improve graph readability.")
        with st.spinner("🕸️ Building network graph..."):
            G, pos, cluster_map = cached_network_graph(clustered_df)
        if not G.nodes():
            st.warning("Not enough coordinated activity to build a network graph.")
        else:
            fig_net = go.Figure()
            edge_x, edge_y = [], []
            for edge in G.edges(): x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))
            node_x, node_y, node_text, node_color = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]; node_x.append(x); node_y.append(y)
                hover_text = f"User: {node}<br>Platform: {G.nodes[node].get('platform', 'N/A')}<br>Cluster: {cluster_map.get(node, 'N/A')}"
                node_text.append(hover_text)
                cluster_id = cluster_map.get(node)
                node_color.append(cluster_id if cluster_id not in [-1, -2] else -1)
            nodes_df = pd.DataFrame({'x': node_x, 'y': node_y, 'text': node_text, 'color': node_color, 'size': [G.degree(node) for node in G.nodes()]})
            fig_net.add_trace(go.Scatter(x=nodes_df['x'], y=nodes_df['y'], mode='markers', hoverinfo='text', text=nodes_df['text'],
                                         marker=dict(showscale=False, colorscale='Viridis', size=nodes_df['size'] * 1.5 + 5, color=nodes_df['color'], line_width=2, opacity=0.8)))
            fig_net.update_layout(title='Network of Coordinated Accounts', showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=700)
            st.plotly_chart(fig_net, use_container_width=True)

# ==================== TAB 3: Fundraising & Risk ====================
with tab3:
    st.subheader("Fundraising Campaign Integrity")
    st.markdown("This section automatically identifies fundraising campaigns based on links found in your data. It assesses their risk based on CIB network analysis.")
    if not df_for_analysis.empty:
        with st.spinner("🔗 Identifying and assessing fundraising campaigns..."):
            fundraising_campaigns_df = cached_find_fundraising_campaigns(df_for_analysis, coordinated_groups_df)
        if not fundraising_campaigns_df.empty:
            st.info(f"✅ Found {len(fundraising_campaigns_df)} potential fundraising campaigns.")
            st.dataframe(fundraising_campaigns_df)
            st.markdown("""
            **How to interpret this table:**
            - **Coordination_Score:** A higher score indicates the campaign link is being shared by accounts that are also part of a detected CIB network.
            - **Risk_Flag:** `High` flags campaigns pushed by a coordinated network. `Low` suggests organic virality. `Needs Review` flags smaller groups that might warrant closer inspection.
            """)
        else: st.warning("No fundraising links were detected in the provided data. Please ensure your data contains links to common fundraising platforms like GoFundMe or PayPal.")
