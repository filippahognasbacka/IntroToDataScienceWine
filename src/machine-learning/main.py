from embeddings import Embeddings

def main():
    embeddings = Embeddings()
    embeddings.load_data()
    embeddings.load_embeddings()

    while True:
        query = input("Enter a query: ")

        recommendations = embeddings.recommend_wines(query)

        print(recommendations)

if __name__ == "__main__":
    main()