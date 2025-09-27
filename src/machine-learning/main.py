from embeddings import Embeddings

def main():
    mode = input("Enter a mode [h/C]: ").lower()

    if mode == "h":
        mode = "hybrid"
    elif mode == "c" or mode == "":
        mode = "concatenation"
    else:
        raise ValueError("Invalid mode")

    print(f"Selected mode: {mode}")
    print("To change mode, please restart the program.")

    if mode == "hybrid":
        attribute_weight = float(input("Enter an attribute weight (Use a float value between 0 and 1): "))
        description_weight = float(input("Enter a description weight (Use a float value between 0 and 1): "))
    else:
        attribute_weight = None
        description_weight = None

    embeddings = Embeddings(mode)
    embeddings.load_data()
    embeddings.load_embeddings()

    while True:
        query = input("Enter a query: ")

        recommendations = embeddings.recommend_wines(query, 5, attribute_weight, description_weight)

        print(recommendations)

if __name__ == "__main__":
    main()