from flask import Flask, request, jsonify
import pandas as pd
from model import recommend_products, data

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Product Recommendation API!"

@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        product_name = request.args.get("product_name")  # Get product name from request
        if product_name not in data["product_name"].values:
            return jsonify({"error": "Product name not found!"}), 404
        
        recommendations = recommend_products(product_name)
        return jsonify({"product_name": product_name, "recommendations": recommendations})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
