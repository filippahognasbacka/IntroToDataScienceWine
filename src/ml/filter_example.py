from embeddings import Embeddings

embeddings = Embeddings(approach="concatenation")
embeddings.load_data()
embeddings.load_embeddings()

print("Filter example: Filter for only French Chardonnay")

results = embeddings.recommend_wines(
    "crisp and elegant", 
    top_n=5,
    filter_attributes={"country": "France", "variety": "Chardonnay"}
)
print(results[["variety", "country", "province", "points"]])
