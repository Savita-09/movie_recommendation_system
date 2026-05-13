import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Root variables */
    :root {
        --primary: #E50914;
        --secondary: #221f1f;
        --accent: #564d4d;
        --light: #f5f5f1;
        --success: #00D084;
        --warning: #FF6B6B;
        --info: #4A90E2;
    }
    
    /* Overall styling */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f5f1 0%, #ffffff 100%);
        padding: 20px;
    }
    
    /* Typography enhancements */
    h1 {
        color: var(--primary) !important;
        text-shadow: 2px 2px 4px rgba(60, 60, 60, 60);
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: var(--secondary) !important;
        border-bottom: 3px solid var(--primary);
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-weight: 700;
    }
    
    h3 {
        color: var(--teritary) !important;
        font-weight: 600;
    }
    
    h4 {
        color: var(--teritary) !important;
        font-weight: 600;
    }
    
    /* Enhanced metric styling */
    .stMetric {
        background: linear-gradient(135deg, #cc0000 10%, #f9f9f9 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 10, 0.08);
        border: 2px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.15);
        transform: translateY(-2px);
        border-color: var(--primary);
    }
    
    .stMetric label {
        color: var(--accent) !important;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetric > div > div > p {
        color: var(--primary) !important;
        font-weight: 800;
        font-size: 2rem;
    }
    
    /* Recommendation card styling */
    .recommendation-card {
        background: linear-gradient(135deg, #6fa8dc 10%, #7ab8bb 100%);
        padding: 20px;
        border-radius: 14px;
        margin: 15px 0;
        border-left: 5px solid var(--primary);
        box-shadow: 0 6px 20px rgba(0, 10, 10, 0.12);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .recommendation-card:hover {
        transform: translateX(8px) translateY(-3px);
        box-shadow: 0 12px 35px rgba(229, 9, 20, 0.25);
        border-left-width: 6px;
    }
    
    .recommendation-card:hover::before {
        left: 100%;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 3px solid #e0e0e0;
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 10px 25px;
        font-weight: 700;
        color: var(--teritary);
        border-radius: 10px 10px 0 0;
        border-bottom: 4px solid transparent;
        transition: all 0.3s ease;
        background: linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.02) 100%);
        cursor: pointer;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(180deg, rgba(229,9,20,0.05) 0%, transparent 100%);
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom-color: var(--primary) !important;
        color: var(--primary) !important;
        background: linear-gradient(180deg, rgba(229,9,20,0.08) 0%, transparent 100%);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div > input {
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stSlider > div > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(229, 9, 20, 0.1) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, #c40812 100%);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
        background: linear-gradient(135deg, #c40812 0%, #a20610 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(229, 9, 20, 0.2);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: linear-gradient(135deg, #cc0000 10%, #f9f9f9 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 2px solid #f0f0f0;
    }
    
    .stRadio > div > label {
        font-weight: 600;
        color: var(--secondary);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stRadio > div > label:hover {
        color: var(--primary);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, #c40812 100%);
    }
    
    .stSlider > label {
        font-weight: 700;
        color: var(--teritary);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f9f9f9 100%);
        border-right: 3px solid #e0e0e0;
    }
    
    /* Expander styling */
    .stExpander {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stExpander:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.1);
    }
    
    /* Expander button styling */
    .stExpander > div > details > summary {
        padding: 15px;
        font-weight: 700;
        color: var(--secondary);
        cursor: pointer;
        background: linear-gradient(135deg, #f9f9f9 0%, #f0f0f0 100%);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stExpander > div > details > summary:hover {
        background: linear-gradient(135deg, rgba(229,9,20,0.05) 0%, rgba(229,9,20,0.02) 100%);
        color: var(--primary);
    }
    
    /* Info/Warning box styling */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        padding: 15px 20px;
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .stAlert:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }
    
    /* Success badge */
    .success-badge {
        background: linear-gradient(135deg, var(--success) 0%, #00a86b 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(0, 208, 132, 0.3);
    }
    
    /* Rating badge */
    .rating-badge {
        background: linear-gradient(135deg, var(--primary) 0%, #c40812 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        overflow: hidden;
        border: 2px solid #e0e0e0;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, var(--primary) 0%, transparent 100%);
        margin: 30px 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary) 0%, #c40812 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #c40812 0%, #a20610 100%);
    }
    
    /* Animation for cards */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .recommendation-card {
        animation: slideIn 0.5s ease-out forwards;
    }
    
    /* Interactive text styling */
    a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 700;
        transition: all 0.3s ease;
        border-bottom: 2px solid transparent;
    }
    
    a:hover {
        color: #c40812;
        border-bottom-color: var(--primary);
    }
    
    /* Container styling */
    .stContainer {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.8rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 15px;
            font-size: 0.85rem;
        }
        
        .recommendation-card {
            margin: 10px 0;
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_imdb_data():
    """
    Load IMDB dataset. 
    Users need to:
    1. Download from Kaggle: https://www.kaggle.com/datasets/chenyanglim/imdb-v2
    2. Place movies.csv in the same directory as this app
    
    Or use this sample data structure if CSV not available
    """
    try:
        movies_df = pd.read_csv("C:\\Users\\savita\\Downloads\\imdb_movie_dataset.csv")
    except FileNotFoundError:
        st.warning("⚠️ movies.csv not found. Using sample IMDB-inspired data. To use real data:")
        st.info("""
        1. Download from: https://www.kaggle.com/datasets/chenyanglim/imdb-v2
        2. Extract movies.csv to the app directory
        3. Restart the app
        """)
    
    return movies_df


@st.cache_resource
def load_ratings_data(movies_df):
    """Generate sample user ratings data based on movie genres"""
    np.random.seed(42)
    n_users = 150
    n_movies = len(movies_df)
    
    user_ids = []
    movie_ids = []
    ratings = []
    
    for user_id in range(n_users):
        n_ratings = np.random.randint(20, 40)
        rated_movies = np.random.choice(n_movies, min(n_ratings, n_movies), replace=False)
        
        for movie_id in rated_movies:
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            base_rating = movies_df.iloc[movie_id]['Rating']
            rating = np.random.normal(base_rating, 1.2)
            rating = np.clip(rating, 1, 10)
            ratings.append(round(rating, 1))
    
    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings
    })
    
    return ratings_df


def get_content_based_recommendations(movie_title, movies_df, n_recommendations=5):
    """Get content-based recommendations using genre and description similarity"""
    if movie_title not in movies_df['Title'].values:
        return pd.DataFrame()
    
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))

    movies_df['combined'] = movies_df['Genre'].fillna('') + ' ' + movies_df['Description'].fillna('')
    genre_vectors = tfidf.fit_transform(movies_df['combined'])
    
    similarity_matrix = cosine_similarity(genre_vectors)
    movie_idx = movies_df[movies_df['Title'] == movie_title].index[0]
    
    similar_indices = similarity_matrix[movie_idx].argsort()[-n_recommendations-1:-1][::-1]
    recommendations = movies_df.iloc[similar_indices][['Title', 'Rating', 'Year', 'Genre', 'Votes']].copy()
    recommendations['similarity_score'] = similarity_matrix[movie_idx][similar_indices]
    
    return recommendations

def get_collaborative_recommendations(user_id, ratings_df, movies_df, n_recommendations=5):
    """Get collaborative filtering recommendations"""
    user_movie_matrix = ratings_df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating',
        fill_value=0
    )
    
    if user_id >= len(user_movie_matrix):
        user_id = len(user_movie_matrix) - 1
    
    user_similarity = cosine_similarity(user_movie_matrix)
    user_idx = user_id
    similar_users = user_similarity[user_idx].argsort()[-6:-1][::-1]
    
    user_rated = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values)
    
    recommendations_scores = {}
    for sim_user in similar_users:
        sim_user_ratings = ratings_df[ratings_df['user_id'] == sim_user]
        for _, row in sim_user_ratings.iterrows():
            movie_id = row['movie_id']
            if movie_id not in user_rated:
                if movie_id not in recommendations_scores:
                    recommendations_scores[movie_id] = []
                recommendations_scores[movie_id].append(row['rating'] * user_similarity[user_idx][sim_user])
    
    final_scores = {mid: np.mean(scores) for mid, scores in recommendations_scores.items()}
    top_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    if not top_movies:
        return pd.DataFrame()
    
    movie_ids = [m[0] for m in top_movies]
    scores = [m[1] for m in top_movies]
    
    recommendations = movies_df.iloc[movie_ids][['Title', 'Rating', 'Year', 'Genre', 'Votes']].copy()
    recommendations['cf_score'] = scores[:len(recommendations)]
    
    return recommendations


def render_movie_card(idx, movie, score_type, score_value):
    """Render an enhanced interactive movie card"""
    st.markdown(f"""
    <div class="recommendation-card">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 15px;">
            <div style="flex: 1;">
                <h4 style="margin: 0; color: #221f1f; font-size: 1.05rem; font-weight: 700;">
                    #{idx} • {movie['Title']}
                </h4>
                <p style="margin: 8px 0 0 0; color: #564d4d; font-size: 0.9rem; font-weight: 500;">
                    🎬 {int(movie['Year'])} • {movie['Genre'][:45]}
                </p>
                <p style="margin: 6px 0 0 0; color: #888; font-size: 0.85rem;">
                    👥 {int(movie['Votes']):,} votes
                </p>
            </div>
            <div style="text-align: right; min-width: 100px;">
                <div class="rating-badge">
                    ⭐ {movie['Rating']:.1f}
                </div>
                <div style="color: white; font-size: 0.75rem; margin-top: 10px; background: rgba(255,255,255,0.3); padding: 6px 8px; border-radius: 6px; font-weight: 600;">
                    {score_type}: {score_value:.0%}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin: 0 0 10px 0;">🎬 Movie Recommendation System</h1>
    </div>
    """, unsafe_allow_html=True)

movies_df = load_imdb_data()
ratings_df = load_ratings_data(movies_df)

with st.sidebar:
    st.markdown("### ⚙️ Settings & Configuration")
    
    recommendation_type = st.radio(
        "🎯 Recommendation Method",
        ["Content-Based", "Collaborative Filtering"],
        help="Content-Based: Find similar movies | Collaborative: What users like you enjoyed"
    )
    
    st.markdown("---")
    
    n_recommendations = st.slider(
        "📊 Number of Recommendations",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
        help="How many movies to recommend"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎬 Movies", len(movies_df))
        st.metric("👤 Users", ratings_df['user_id'].max() + 1)
    with col2:
        st.metric("⭐ Avg Rating", f"{movies_df['Rating'].mean():.1f}")
        st.metric("📝 Ratings", f"{len(ratings_df):,}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommendations", 
    "📈 Analytics", 
    "🔍 Movie Search", 
    "📋 Dataset"
])


with tab1:
    st.markdown("### ✨ Get Recommendations")
    
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("#### 🎯 Content-Based Filtering")
        st.write("*Find movies similar to ones you love*")
        
        selected_movie = st.selectbox(
            "Choose a movie:",
            sorted(movies_df['Title'].values),
            key="content_movie",
            help="Select any movie to find similar ones"
        )
        
        if st.button("🔍 Find Similar Movies", key="cb_search", use_container_width=True):
            with st.spinner("🔄 Analyzing movie similarities..."):
                recommendations = get_content_based_recommendations(
                    selected_movie, 
                    movies_df, 
                    n_recommendations
                )
                
                if not recommendations.empty:
                    
                    for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                        render_movie_card(idx, movie, "Similarity", movie['similarity_score'])
                else:
                    st.warning("⚠️ No recommendations found for this movie")
    

    with col2:
        st.markdown("#### 👥 Collaborative Filtering")
        st.write("*Get similar user recommendations*")
        
        user_id = st.slider(
            "Select a User Profile:",
            min_value=0,
            max_value=int(ratings_df['user_id'].max()),
            value=0,
            key="cf_user",
            help="Choose a user ID to get recommendations"
        )
        
        if st.button("🔍 Get User Recommendations", key="cf_search", use_container_width=True):
            with st.spinner("🔄 Finding similar users..."):
                recommendations = get_collaborative_recommendations(
                    user_id, 
                    ratings_df, 
                    movies_df, 
                    n_recommendations
                )
                
                if not recommendations.empty:
                    
                    for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                        render_movie_card(idx, movie, "CF Score", movie['cf_score'] / 10)
                else:
                    st.info("ℹ️ Not enough data for this user. Try another user ID!")


with tab2:
    st.markdown("### 📊 Dataset Analysis & Insights")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Avg Rating", f"{movies_df['Rating'].mean():.2f}/10")
    with col2:
        st.metric("📈 Median Rating", f"{movies_df['Rating'].median():.1f}")
    with col3:
        st.metric("📅 Year Range", f"{int(movies_df['Year'].min())}-{int(movies_df['Year'].max())}")
    with col4:
        st.metric("🗳️ Total Votes", f"{movies_df['Votes'].sum():,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Rating Distribution")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(movies_df['Rating'], bins=20, 
                color='#E50914', alpha=0.8, 
                edgecolor='#221f1f', linewidth=1.5)
        ax.set_xlabel('IMDB Rating', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Movies', fontsize=11, fontweight='bold')
        ax.set_title('Movie Ratings Distribution', fontsize=12, fontweight='bold', color='#221f1f')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    

    with col2:
        st.markdown("#### 📅 Movies by Year")
        year_counts = movies_df['Year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(year_counts.index, year_counts.values, 
                marker='o', color='#E50914', linewidth=2.5, markersize=8)
        ax.fill_between(year_counts.index, year_counts.values, 
                        alpha=0.3, color='#E50914')
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Movies Released Per Year', fontsize=12, fontweight='bold', color='#221f1f')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### ⭐ Top 10 Highest Rated Movies")
    top_movies = movies_df.nlargest(10, 'Rating')[['Title', 'Rating', 'Year', 'Votes']].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#E50914' if i < 5 else '#8DCBCE' for i in range(len(top_movies))]
    bars = ax.barh(range(len(top_movies)), top_movies['Rating'], 
                    color=colors, alpha=0.85, edgecolor='#221f1f', linewidth=1.5)
    ax.set_yticks(range(len(top_movies)))
    ax.set_yticklabels([f"{i+1}. {title}" for i, title in enumerate(top_movies['Title'])], fontsize=10)
    ax.set_xlabel('Rating', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Highest Rated Movies', fontsize=12, fontweight='bold', color='#221f1f')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f9f9f9')
    ax.set_xlim(7, 10)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### 🔥 Most Popular Movies (by Votes)")
    most_voted = movies_df.nlargest(10, 'Votes')[['Title', 'Rating', 'Votes']].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#E50914' if i < 5 else '#8DCBCE' for i in range(len(most_voted))]
    votes_millions = most_voted['Votes'] / 1_000_000
    ax.barh(range(len(most_voted)), votes_millions, 
            color=colors, alpha=0.85, edgecolor='#221f1f', linewidth=1.5)
    ax.set_yticks(range(len(most_voted)))
    ax.set_yticklabels([f"{i+1}. {title}" for i, title in enumerate(most_voted['Title'])], fontsize=10)
    ax.set_xlabel('Votes (Millions)', fontsize=11, fontweight='bold')
    ax.set_title('Most Voted Movies', fontsize=12, fontweight='bold', color='#221f1f')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


with tab3:
    st.markdown("### 🔍 Advanced Search & Filter")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input(
            "🔎 Search by movie title",
            "",
            placeholder="Type movie name... (e.g., Inception, The Matrix)"
        )
    with col2:
        min_rating = st.slider("⭐ Min rating", 0.0, 10.0, 6.0, step=0.5)
    with col3:
        min_votes = st.slider("👥 Min votes", 0, int(movies_df['Votes'].max()), 0, step=100000)
    
    
    filtered_movies = movies_df[
        (movies_df['Title'].str.contains(search_term, case=False, na=False)) &
        (movies_df['Rating'] >= min_rating) &
        (movies_df['Votes'] >= min_votes)
    ].sort_values('Rating', ascending=False)
    
    st.markdown(f"**🎬 Found {len(filtered_movies)} movies**")
    
    if not filtered_movies.empty:
        for idx, (_, movie) in enumerate(filtered_movies.iterrows(), 1):
            col1, col2, col3, col4, col5 = st.columns([2.5, 0.8, 1, 0.8, 0.8])
            
            with col1:
                st.write(f"**{idx}. {movie['Title']}**")
                st.caption(f"🎬 {movie['Genre']} • {int(movie['Year'])}")
            
            with col2:
                st.metric("Rating", f"{movie['Rating']:.1f}", label_visibility="collapsed")
            
            with col3:
                st.metric("Votes", f"{int(movie['Votes']/1000)}K", label_visibility="collapsed")
            
            with col4:
                if st.button("📝", key=f"detail_{idx}", help="Show details"):
                    with st.expander(f"Details - {movie['Title']}", expanded=True):
                        st.write(f"**Description:** {movie['Description']}")
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.write(f"**Year:** {int(movie['Year'])}")
                        with col_b:
                            st.write(f"**Rating:** {movie['Rating']}/10")
                        with col_c:
                            st.write(f"**Votes:** {int(movie['Votes']):,}")
                        with col_d:
                            st.write(f"**Genre:** {movie['Genre']}")
            
            with col5:
                if st.button("⭐", key=f"rate_{idx}", help="Like this movie"):
                    st.success("👍 Thanks for the feedback!")
            
            st.markdown("---")
    else:
        st.info("ℹ️ No movies found matching your criteria. Try adjusting the filters!")

with tab4:
    st.markdown("### 📋 Complete Movie Dataset")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Display Options:**")
        rows_to_show = st.select_slider(
            "Number of rows",
            options=[10, 25, 50],
            value=25
        )
        sort_by = st.selectbox(
            "Sort by",
            ["Rating", "Votes", "Year", "Title"]
        )
    
    with col2:
        sort_column = {
            "Title": "Title",
            "Rating": "Rating",
            "Year": "Year",
            "Votes": "Votes"
        }[sort_by]
        
        ascending = st.checkbox("Ascending order", value=False)
        displayed_movies = movies_df.sort_values(sort_column, ascending=ascending).head(rows_to_show)
        
        display_df = displayed_movies[['Title', 'Rating', 'Year', 'Genre', 'Votes']].copy()
        display_df['Votes'] = display_df['Votes'].apply(lambda x: f"{int(x):,}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    st.markdown("### ⬇️ Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        csv = movies_df[['Title', 'Rating', 'Year', 'Genre', 'Votes']].to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"imdb_movies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.info(f"📊 Dataset: **{len(movies_df)}** movies | **{len(ratings_df):,}** user ratings")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #564E4d; font-size: 0.9rem; padding: 20px 0;">
    <p>🎬 <b>Movie Recommendation System</b> | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
