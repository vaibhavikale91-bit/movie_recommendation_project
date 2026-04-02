def hybrid_recommend(movie_title, movies_df, content_sim, collab_sim=None):
    
    # 🔍 Check if movie exists
    if movie_title not in movies_df['title'].values:
        return ["Movie not found!"]

    # Get index of selected movie
    idx = movies_df[movies_df['title'] == movie_title].index[0]

    # -----------------------------
    # Content-based similarity
    # -----------------------------
    content_scores = list(enumerate(content_sim[idx]))

    # -----------------------------
    # Hybrid (optional)
    # -----------------------------
    if collab_sim is not None:
        collab_scores = list(enumerate(collab_sim[idx]))

        # Combine both scores (weighted)
        hybrid_scores = []
        for i in range(len(content_scores)):
            score = (0.7 * content_scores[i][1]) + (0.3 * collab_scores[i][1])
            hybrid_scores.append((i, score))
    else:
        hybrid_scores = content_scores

    # -----------------------------
    # Sort by similarity
    # -----------------------------
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

    # Remove selected movie itself
    hybrid_scores = hybrid_scores[1:6]

    # -----------------------------
    # Get movie names
    # -----------------------------
    recommended_movies = []
    for i in hybrid_scores:
        recommended_movies.append(movies_df.iloc[i[0]]['title'])

    return recommended_movies