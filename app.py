from flask import Flask, request, jsonify
from model import get_similar_products

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Collaborative Recommender!"

@app.route('/recommend')
def recommend():
    product_id = request.args.get('product_id')
    recommendations = get_similar_products(product_id)
    return jsonify({"recommended": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
