# app.py - FIXED RECOMMENDATIONS SECTION (Movie Name + Poster + Rating + Genre ONLY)
import streamlit as st
import pandas as pd
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="🎬  Movie Recommendation system", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎥"
)

# -------------------------
# CUSTOM CSS (Advanced UI + Animations)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
* {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    color: #ffffff;
}

.title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2rem;
    text-shadow: 0 0 30px rgba(255,107,107,0.5);
}

.movie-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.5rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    height: 420px;
    display: flex;
    flex-direction: column;
}

.movie-card:hover {
    transform: translateY(-15px) scale(1.05);
    box-shadow: 0 20px 40px rgba(255,107,107,0.3);
    border-color: #ff6b6b;
}

.stButton > button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white !important;
    border: none;
    border-radius: 50px;
    height: 3.5em;
    width: 250px;
    font-size: 1.2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(255,107,107,0.4);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(255,107,107,0.6);
}

.metric-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
}

.rating-badge {
    display: inline-block;
    background: linear-gradient(45deg, #ffd700, #ffed4e);
    color: #000;
    padding: 0.4rem 1rem;
    border-radius: 25px;
    font-weight: 700;
    font-size: 1rem;
    box-shadow: 0 4px 15px rgba(255,215,0,0.4);
    margin: 0.5rem 0;
}

.genre-tag {
    background: rgba(255,255,255,0.1);
    padding: 0.3rem 0.8rem;
    border-radius: 15px;
    font-size: 0.85rem;
    margin: 0.2rem;
    display: inline-block;
}

.chart-container {
    background: rgba(0,0,0,0.7);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Set matplotlib dark theme
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = '#1a1a2e'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    movies_df = pd.read_csv("data/movies.csv")
    links_df = pd.read_csv("data/links.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    
    movies_df = movies_df.merge(links_df, on="movieId", how="left")
    content_sim_matrix = pickle.load(open("content_sim.pkl", "rb"))
    
    return movies_df, content_sim_matrix, ratings_df

movies_df, content_sim_matrix, ratings_df = load_data()

# -------------------------
# RECOMMENDATION FUNCTIONS
# -------------------------
def advanced_hybrid_recommend(movie_title, top_n=5):
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    content_sim = list(enumerate(content_sim_matrix[idx]))
    content_sim = sorted(content_sim, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommendations = []
    for i, score in content_sim:
        rec_movie_id = movies_df.iloc[i]['movieId']
        rec_popularity = len(ratings_df[ratings_df['movieId'] == rec_movie_id])
        hybrid_score = 0.8 * score + 0.2 * (rec_popularity / 10000)
        recommendations.append((i, hybrid_score))
    
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    return [movies_df.iloc[i[0]] for i in recommendations]

def get_movie_stats(movie_title):
    movie_data = movies_df[movies_df['title'] == movie_title].iloc[0]
    movie_id = movie_data['movieId']
    
    movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']
    avg_rating = movie_ratings.mean() if len(movie_ratings) > 0 else 0
    rating_count = len(movie_ratings)
    
    return {
        'avg_rating': round(avg_rating, 1),
        'rating_count': rating_count,
        'genres': str(movie_data.get('genres', 'N/A')),
        'movie_ratings': movie_ratings
    }

# -------------------------
# CHART FUNCTIONS (Pure Matplotlib)
# -------------------------
def create_rating_distribution_fig(movie_ratings, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(movie_ratings) > 0:
        n, bins, patches = ax.hist(movie_ratings, bins=10, 
                                  color='#ff6b6b', alpha=0.7, 
                                  edgecolor='white', linewidth=1.2)
        
        mean_val = movie_ratings.mean()
        ax.axvline(mean_val, color='gold', linestyle='--', linewidth=3, 
                  label=f'Mean: {mean_val:.1f}')
    else:
        ax.text(0.5, 0.5, 'No ratings data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16, color='white')
    
    ax.set_title(f'📊 {title}', fontsize=16, fontweight='bold', pad=20, color='white')
    ax.set_xlabel('Rating', fontsize=12, color='white')
    ax.set_ylabel('Number of Ratings', fontsize=12, color='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    return fig

def create_similarity_scores_fig(selected_title, recommendations, top_n):
    sim_scores = [1.0]
    rec_titles = [selected_title[:20] + '...' if len(selected_title) > 20 else selected_title]
    
    selected_idx = movies_df[movies_df['title'] == selected_title].index[0]
    for rec in recommendations:
        rec_idx = movies_df[movies_df['title'] == rec['title']].index[0]
        sim_score = content_sim_matrix[selected_idx][rec_idx]
        sim_scores.append(sim_score)
        short_title = rec['title'][:20] + '...' if len(rec['title']) > 20 else rec['title']
        rec_titles.append(short_title)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4ecdc4'] + ['#ff6b6b'] * top_n
    bars = ax.bar(range(len(rec_titles)), sim_scores, color=colors, 
                  alpha=0.8, edgecolor='white', linewidth=1.2)
    
    for bar, score in zip(bars, sim_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', 
                fontweight='bold', color='white')
    
    ax.set_title('🔗 Recommendation Confidence Scores', fontsize=16, 
                fontweight='bold', pad=20, color='white')
    ax.set_xlabel('Movies', fontsize=12, color='white')
    ax.set_ylabel('Similarity Score', fontsize=12, color='white')
    ax.set_xticks(range(len(rec_titles)))
    ax.set_xticklabels(rec_titles, rotation=45, ha='right', color='white')
    
    plt.tight_layout()
    return fig

# -------------------------
# TMDB POSTER FUNCTION
# -------------------------
def get_poster_url(tmdb_id):
    api_key = "YOUR_API_KEY"  # Replace with your actual TMDB API key
    
    if pd.isna(tmdb_id) or tmdb_id == 0:
        return "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
        data = requests.get(url, timeout=5).json()
        if data.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except:
        pass
    
    return "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="title">🎬  Movie Recommendation System</div>', unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.markdown("## ⚙️ Settings")
top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)
show_charts = st.sidebar.checkbox("📊 Show Analytics", True)

# -------------------------
# MAIN CONTENT
# -------------------------
search = st.text_input("🔍 Search movies")
filtered_movies = movies_df[movies_df['title'].str.contains(search, case=False, na=False)] if search else movies_df

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎥 Select Your Movie")
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        options=filtered_movies['title'].values,
        format_func=lambda x: f"🎬 {x[:50]}..." if len(x) > 50 else f"🎬 {x}"
    )
    
    # AUTO SHOW SELECTED MOVIE GRAPH
    if selected_movie:
        stats = get_movie_stats(selected_movie)
        st.markdown("### 📊 Selected Movie Analytics")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_selected = create_rating_distribution_fig(
            stats['movie_ratings'], 
            f"Ratings Distribution: {selected_movie}"
        )
        st.pyplot(fig_selected)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Movie info cards
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("⭐ Average Rating", f"{stats['avg_rating']}/5")
        with col_info2:
            st.metric("👥 Total Ratings", f"{stats['rating_count']:,}")
        with col_info3:
            st.markdown(f"**🎭 Genres:** {stats['genres']}")

with col2:
    st.markdown("### 💡 Quick Stats")
    overall_avg = ratings_df['rating'].mean()
    st.metric("Total Movies", f"{len(movies_df):,}")
    st.metric("Total Ratings", f"{len(ratings_df):,}")
    st.metric("Avg Rating", f"{overall_avg:.1f}/5")

# -------------------------
# RECOMMEND BUTTON - FIXED VERSION
# -------------------------
if st.button("🚀 Get Smart Recommendations", type="primary"):
    with st.spinner("🔮 Analyzing your taste..."):
        time.sleep(1.5)
        
        recommendations = advanced_hybrid_recommend(selected_movie, top_n)
        selected_stats = get_movie_stats(selected_movie)
        
        st.success(f"✨ Found **{len(recommendations)}** perfect matches for **{selected_movie}**!")
        
        # FIXED RECOMMENDATIONS SECTION - ONLY: Movie Name + Poster + Rating + Genre
        st.markdown("## 🎯 Your Personalized Recommendations")
        cols = st.columns(5)
        
        for i, movie_data in enumerate(recommendations):
            movie_stats = get_movie_stats(movie_data['title'])
            poster = get_poster_url(movie_data['tmdbId'])
            
            with cols[i % 5]:
               
                
                # 1. POSTER
                st.image(poster, use_container_width=True)
                
                # 2. MOVIE NAME
                st.markdown(f"**{movie_data['title']}**")
                
                # 3. RATING
                st.markdown(f'<span class="rating-badge">⭐ {movie_stats["avg_rating"]}</span>', unsafe_allow_html=True)
                
                # 4. GENRE
                st.markdown(f'<span class="genre-tag">🎭 {movie_stats["genres"]}</span>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # ENHANCED CHARTS SECTION
        if show_charts:
            st.markdown("---")
            st.markdown("## 📊 Smart Analytics Dashboard")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig1 = create_rating_distribution_fig(
                    selected_stats['movie_ratings'], 
                    f"Detailed: {selected_movie}"
                )
                st.pyplot(fig1)
            
            with chart_col2:
                fig2 = create_similarity_scores_fig(selected_movie, recommendations, top_n)
                st.pyplot(fig2)
            
            st.markdown("### 🌍 Dataset Overview")
            fig3 = create_rating_distribution_fig(ratings_df['rating'], "Complete Dataset")
            st.pyplot(fig3)
            
            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <h3>Vaibhavi Kale</h3>
    <p>🎬 Smart Movie Recommender using Machine Learning and Streamlit for personalized movie suggestions</p>
</div>
""", unsafe_allow_html=True)