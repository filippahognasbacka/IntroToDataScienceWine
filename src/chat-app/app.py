import sys
from pathlib import Path
from flask import Flask, render_template, request
import math

# Add the src directory to the path so we can import from ml
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ml.embeddings import Embeddings

app = Flask(__name__)
app.debug = True

# Load embeddings engine at application start

print("Initializing Embeddings engine...")
embeddings_engine = Embeddings(approach="concatenation")

print("Loading data...")
embeddings_engine.load_data()

print("Loading embeddings (This will take a while)...")
embeddings_engine.load_embeddings()

print("Embeddings engine ready.")

# Serve web app

@app.route("/")
def index():
    query = request.args.get("query", "").strip()
    top_n = int(request.args.get("top_n", 5))
    max_price = request.args.get("price")

    error = None
    recommendations = []
    filtered_recommendations = []
    warning = None

    if query:
        try:
            results = embeddings_engine.recommend_wines(query, top_n=top_n)

            recommendations = results.to_dict(orient='records')

            low_recommendation = all(rec["similarity_score"] < 0.45 for rec in recommendations)
            if low_recommendation:
                warning = "The recommendations may be inaccurate. Please try a different query."

        except Exception as e:
            error = str(e)
    
    if type(max_price) == str:
        if len(max_price) < 1:
            max_price = 10000
        elif any(char.isalpha() for char in max_price):
            warning = "Price field can't have letters in it"
        elif "-" in max_price:
            warning = "Price can't be negative"

    if len(recommendations) > 0:
        for i in range(0, len(recommendations)):
            if math.isnan(recommendations[i]['price']):
                continue  # wine not added to recommendations
            elif warning is not None:
                max_price = 10000
                print("max_price set to:", max_price)
            elif int(recommendations[i]['price']) <= int(max_price):
                filtered_recommendations.append(recommendations[i])

    return render_template("index.html",
                          query=query,
                          top_n=top_n,
                          recommendations=filtered_recommendations,
                          error=error,
                          warning=warning)
