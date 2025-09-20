import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = None
        self.embeddings = None

    def load_data(self):
        df = pd.read_csv("/path/to/wine-review-dataset/winemag-data-130k-v2.csv")
        df = df.dropna(subset=["description"])

        self.df = df

    def load_embeddings(self):
        self.embeddings = self.model.encode(self.df['description'].tolist(), show_progress_bar=True)

        print("Loaded embeddings with shape: ", self.embeddings.shape)
        print(self.embeddings[0])

    def recommend_wines(self, query: str, top_n: int = 5):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_n]

        return self.df.iloc[top_indices][["country", "variety", "winery", "points", "description"]]
