from flask import Flask
from flask import render_template
from embeddings_simple import Embeddings

app = Flask(__name__)
app.debug = True

@app.route("/")
def index():
    embeddings = Embeddings()
    embeddings.load_data()
    embeddings.load_embeddings()
    query = input("Enter a query: ")
    recommendations = embeddings.recommend_wines(query)
    print(recommendations)
    return render_template("index.html", recommendations=recommendations)
