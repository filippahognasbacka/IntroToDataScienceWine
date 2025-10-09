import sys
from pathlib import Path
from flask import Flask, render_template, request

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

    error = None
    recommendations = None
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

    return render_template("index.html",
                          query=query,
                          top_n=top_n,
                          recommendations=recommendations,
                          error=error,
                          warning=warning)
