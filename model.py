import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from CSV
data = pd.read_csv("dataset.csv")

# Convert text descriptions into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
product_vectors = vectorizer.fit_transform(data['description'])

# Compute cosine similarity
cosine_sim = cosine_similarity(product_vectors, product_vectors)

def recommend_products(product_name, top_n=3):
    # Find the index of the given product
    idx = data.index[data['product_name'] == product_name].tolist()[0]
    
    # Get similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar products (excluding the first one as it’s the same product)
    top_products = [data.iloc[i[0]]['product_name'] for i in sim_scores[1:top_n+1]]
    
    return top_products

# Example usage
if __name__ == "__main__":
    print(recommend_products("iPhone 15 Pro"))
