import pandas as pd
from os import environ
from dotenv import load_dotenv
import re

load_dotenv()

DEFAULT_PATH = "/path/to/fruits.csv"


data = pd.read_csv(environ.get("DEFAULT_PATH", DEFAULT_PATH))

food_pattern = re.compile(r"(FoodType|FoodVariety|FoodMeatCut|FoodPreparation|FoodFlavor)\s*:\s*([A-Za-zÀ-ÿ\-]+)")
wine_pattern = re.compile(r"(?:FoodVariety|FoodBrandName)\s*:\s*([A-Za-zÀ-ÿ\-]+)")

foods_with_wine = {}

def split_wine_names(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name)

for _, row in data.iterrows():
		wine = row["WineEntity"]

		wine_match = wine_pattern.search(str(wine))
		if not wine_match:
				continue
		wine = split_wine_names(wine_match.group(1))

		foods = []

		for col in ["CheeseAndNuts", "MeatAndSeafood", "FruitsAndVegetables", "SaucesAndDips", "Desserts"]:
				text = str(row[col])
				matches = food_pattern.findall(text)

				for _, food in matches:
					foods.append(food)

		foods_with_wine[wine] = list(set(foods))

cleaned_data = pd.DataFrame([(wine, food) for wine, foods in foods_with_wine.items()
for food in foods], columns=["Wine", "Food"])

cleaned_data.to_csv("cleaned_version_foods.csv", index=False)




