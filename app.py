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

# --- Auto-Encoding CSV Reader ---
def read_uploaded_file_with_encoding_detection(uploaded_file, file_name):
    if not uploaded_file:
        return pd.DataFrame()
    bytes_data = uploaded_file.getvalue()
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'utf-16', 'utf-16le']
    for enc in encodings:
        try:
            decoded = bytes_data.decode(enc)
            st.sidebar.info(f"✅ {file_name}: Decoded using '{enc}'")
            sample = decoded.splitlines()[0] if decoded.splitlines() else ""
            sep = '\t' if '\t' in sample else ','
            df = pd.read_csv(StringIO(decoded), sep=sep, on_bad_lines='skip', low_memory=False)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to parse {file_name} with `{enc}`: {e}")
            continue
    st.sidebar.error(f"❌ Failed to decode '{file_name}'.")
    return pd.DataFrame()

# --- Combine Datasets ---
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
        (openmeasure_df, 'Open-Measure', openmeasure_object_col, 'id', 'created_at', 'url')
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

# --- Final Preprocessing ---
def final_preprocess_and_map_columns(df):
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
        clean = group[['account_id', 'timestamp_share', 'original_text', 'URL']].copy()
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3,5), max_features=max_features)
            X = vectorizer.fit_transform(clean['original_text'])
            sim = cosine_similarity(X)
            high_sim = np.where(sim >= threshold)
            pairs = [(i, j) for i, j in zip(*high_sim) if i < j]
            if len(pairs) > 0:
                accounts = clean.iloc[[i for i, j in pairs]]['account_id'].unique()
                groups.append({
                    "posts": clean.rename(columns={'account_id': 'Account ID', 'timestamp_share': 'Timestamp', 'original_text': 'Text', 'URL': 'URL'}).to_dict('records'),
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

# --- Sidebar: Upload Multiple Files ---
st.sidebar.header("📤 Upload Your Data")
st.sidebar.info("Upload one or more CSV files. You can upload multiple Meltwater files.")

# Allow multiple Meltwater uploads
uploaded_meltwater_files = st.sidebar.file_uploader(
    "Upload Meltwater CSV(s)", type="csv", accept_multiple_files=True, key="meltwater_uploads"
)
uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type="csv", key="civicsignals_upload")
uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type="csv", key="openmeasure_upload")

# Read and combine
dfs = []

# Read multiple Meltwater files
if uploaded_meltwater_files:
    for i, file in enumerate(uploaded_meltwater_files):
        file.seek(0)
        df_temp = read_uploaded_file_with_encoding_detection(file, f"Meltwater_{i+1}")
        if not df_temp.empty:
            dfs.append(df_temp)

# Read CivicSignals
if uploaded_civicsignals:
    uploaded_civicsignals.seek(0)
    df_temp = read_uploaded_file_with_encoding_detection(uploaded_civicsignals, "CivicSignals")
    if not df_temp.empty:
        dfs.append(df_temp)

# Read Open-Measure
if uploaded_openmeasure:
    uploaded_openmeasure.seek(0)
    df_temp = read_uploaded_file_with_encoding_detection(uploaded_openmeasure, "Open-Measure")
    if not df_temp.empty:
        dfs.append(df_temp)

if not dfs:
    st.warning("⚠️ Please upload at least one CSV file.")
    st.stop()

combined_raw_df = combine_social_media_data(
    meltwater_df=pd.concat([d for d in dfs if 'influencer' in d.columns], ignore_index=True) if any('influencer' in d.columns for d in dfs) else None,
    civicsignals_df=next((d for d in dfs if 'media_name' in d.columns), None),
    openmeasure_df=next((d for d in dfs if 'actor_username' in d.columns), None)
)

if combined_raw_df.empty:
    st.error("❌ No valid data after combining. Check column names.")
    st.stop()

st.sidebar.success(f"✅ Combined {len(combined_raw_df):,} total posts.")

# --- Preprocess ---
with st.spinner("🧹 Preprocessing data..."):
    df = final_preprocess_and_map_columns(combined_raw_df)

if df.empty:
    st.error("❌ No valid data after preprocessing.")
    st.stop()

st.sidebar.success(f"✅ Processed {len(df):,} valid posts.")

# --- Filters ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

min_date = pd.to_datetime(df['timestamp_share'].min(), unit='s').date()
max_date = pd.to_datetime(df['timestamp_share'].max(), unit='s').date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
start_ts = int(pd.Timestamp(date_range[0], tz='UTC').timestamp())
end_ts = int(pd.Timestamp(date_range[1], tz='UTC').timestamp()) + 86399
filtered_df = df[(df['timestamp_share'] >= start_ts) & (df['timestamp_share'] <= end_ts)].copy()

max_sample = st.sidebar.number_input("Max Posts for Analysis", 0, 100_000, 5000, help="0 = no limit")
if max_sample > 0 and len(filtered_df) > max_sample:
    df_for_analysis = filtered_df.sample(n=max_sample, random_state=42).copy()
    st.sidebar.warning(f"📊 Analyzing {len(df_for_analysis):,} sampled posts.")
else:
    df_for_analysis = filtered_df.copy()
    st.sidebar.info(f"📊 Analyzing {len(df_for_analysis):,} posts.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Similarity & Coordination", "🌐 Network & Risk"])

# ==================== TAB 1: Overview ====================
with tab1:
    st.subheader("📌 Overview of Combined Data")

    st.markdown("### 📋 Raw Data Sample (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 🏆 Top 10 Most Active Influencers")
    top_influencers = df['account_id'].value_counts().head(10)
    fig_influencers = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Number of Posts', 'index': 'Account'})
    st.plotly_chart(fig_influencers, use_container_width=True)

    st.markdown("### 🗣️ Top Emotional Phrases Tracked")
    phrase_counts = df['assigned_phrase'].value_counts()
    if "Other" in phrase_counts.index:
        phrase_counts = phrase_counts.drop("Other")
    fig_phrases = px.bar(phrase_counts, title="Posts by Emotional Phrase", labels={'value': 'Count', 'index': 'Phrase'})
    st.plotly_chart(fig_phrases, use_container_width=True)

    st.markdown("### 📈 Daily Post Volume")
    plot_df = df.copy()
    plot_df['date'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True).dt.date
    time_series = plot_df.groupby('date').size()
    fig_ts = px.line(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'index': 'Date'})
    st.plotly_chart(fig_ts, use_container_width=True)

    if 'Platform' in df.columns:
        st.markdown("### 🌐 Platform Distribution")
        platform_counts = df['Platform'].value_counts()
        fig_platform = px.bar(platform_counts, title="Posts by Platform", labels={'value': 'Count', 'index': 'Platform'})
        st.plotly_chart(fig_platform, use_container_width=True)

# ==================== TAB 2: Similarity & Coordination ====================
with tab2:
    st.subheader("🔍 Detection of Similar Messages")

    eps = st.slider("DBSCAN eps", 0.1, 1.0, 0.3, 0.05, help="Lower = stricter similarity")
    min_samples = st.slider("Min Samples", 2, 10, 2, help="Min posts in a cluster")
    max_features = st.slider("Max TF-IDF Features", 1000, 5000, 2000, 500, help="Vocabulary size")
    threshold = st.slider("Pairwise Similarity Threshold", 0.8, 0.99, 0.9, 0.01, help="Min similarity to be coordinated")

    if st.button("🔍 Run Similarity Analysis"):
        with st.spinner("Clustering posts..."):
            clustered = cached_clustering(df_for_analysis, eps, min_samples, max_features)
        with st.spinner("Finding similar pairs..."):
            groups = cached_find_coordinated_groups(clustered, threshold, max_features)

        if not groups:
            st.warning("No similar messages found above the threshold.")
        else:
            st.success(f"✅ Found **{len(groups)}** coordinated groups.")

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

# ==================== TAB 3: Network & Risk ====================
with tab3:
    st.subheader("🌐 Network of Coordinated Accounts")

    if 'max_nodes_to_display' not in st.session_state:
        st.session_state.max_nodes_to_display = 40

    st.session_state.max_nodes_to_display = st.slider(
        "Max Nodes in Graph", 10, 200, st.session_state.max_nodes_to_display
    )

    if st.button("Build Network Graph"):
        with st.spinner("Building network..."):
            G, pos, cmap = cached_network_graph(clustered)

        if not G.nodes():
            st.warning("No network to display.")
        else:
            G_filtered = G.copy()
            G_filtered.remove_nodes_from([n for n in G if G.degree(n) < 2])
            if not G_filtered.nodes():
                st.info("No nodes meet the minimum connection threshold.")
            else:
                edge_x, edge_y = [], []
                for u, v in G_filtered.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

                node_x, node_y, node_text, node_color = [], [], [], []
                for node in G_filtered.nodes():
                    x, y = pos[node]
                    node_x.append(x); node_y.append(y)
                    deg = G_filtered.degree(node)
                    platform = G_filtered.nodes[node].get('platform', 'Unknown')
                    node_text.append(f"{node}<br>deg: {deg} • {platform}")
                    node_color.append(cmap.get(node, -2))

                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                                      marker=dict(size=[G_filtered.degree(n)*3 + 10 for n in G_filtered.nodes()],
                                                  color=node_color, colorscale='Viridis', showscale=True))
                fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="CIB Network", showlegend=False, hovermode='closest'))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🚨 Risk Assessment: Top Central Accounts")
            degree_centrality = nx.degree_centrality(G_filtered)
            risk_df = pd.DataFrame(degree_centrality.items(), columns=['Account', 'Centrality']).sort_values('Centrality', ascending=False).head(20)
            risk_df['Risk Score'] = (risk_df['Centrality'] * 100).round(2)
            st.dataframe(risk_df, use_container_width=True)
