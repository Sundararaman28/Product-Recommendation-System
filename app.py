from flask import Flask, request, jsonify
import pandas as pd
from model import recommend_products, df

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Product Recommendation API!"

@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        product_id = int(request.args.get("product_id"))  # Get product ID from request
        if product_id not in df["product_id"].values:
            return jsonify({"error": "Product ID not found!"}), 404
        
        recommendations = recommend_products(product_id)
        return jsonify({"product_id": product_id, "recommendations": recommendations})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
