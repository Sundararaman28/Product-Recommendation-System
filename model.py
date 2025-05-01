from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv("user_product.csv")

# Define Reader (for Surprise)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Use KNNBasic (Item-Item CF)
sim_options = {
    'name': 'cosine',
    'user_based': False  # Set to False for Item-Item
}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Recommend products for a given item
def get_similar_products(product_id, top_n=3):
    inner_id = model.trainset.to_inner_iid(product_id)
    neighbors = model.get_neighbors(inner_id, k=top_n)
    neighbor_items = [model.trainset.to_raw_iid(iid) for iid in neighbors]
    return neighbor_items

# Example call
if __name__ == "__main__":
    similar = get_similar_products("p1")
    print("Similar to p1:", similar)
