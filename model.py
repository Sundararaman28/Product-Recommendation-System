import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("dataset.csv")

# Vectorize product descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])

# Compute cosine similarity between products
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_products(product_id, top_n=3):
    """Recommend similar products based on TF-IDF & Cosine Similarity."""
    sim_scores = list(enumerate(cosine_sim[product_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][["product_id", "product_name", "category"]].to_dict(orient="records")
