import pandas as pd
from os import environ
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PATH = "../../data/winemag-data-130k-v2.csv"

data = pd.read_csv(environ.get("DEFAULT_PATH", DEFAULT_PATH))

data["taster_name"].fillna("Unknown", inplace=True)
data["taster_twitter_handle"].fillna("Unknown", inplace=True)
data["designation"].fillna("Unknown", inplace=True)
data["region_1"].fillna("Unknown", inplace=True)
data["region_2"].fillna("-", inplace=True)

place_replacements = {
    "Sicilia": "Sicily",
    "Sicily & Sardinia": "Sicily and Sardinia",
    "Toscana": "Tuscany",
    "Navarra": "Navarre",
    "Bourgogne": "Burgundy",
    "Alsace ": "Alsace"
}

# dropping two columns
data = data.drop('Unnamed: 0', axis='columns')
data = data.drop('taster_twitter_handle', axis='columns')

data["province"] = data["province"].replace(place_replacements)
data["region_1"] = data["region_1"].replace(place_replacements)

mean_price_by_prov_region = data.groupby(["province", "region_1"])["price"].mean() #Calculate mean price using both province and region.
mean_price_by_prov = data.groupby("province")["price"].mean() #if no matches with them, calculate using only province.


def new_price(row):
    if pd.notna(row["price"]):
        return row['price']

    if (row["province"], row["region_1"]) in mean_price_by_prov_region:
        return mean_price_by_prov_region[(row["province"], row["region_1"])]

    if row["province"] in mean_price_by_prov:
        return mean_price_by_prov[row["province"]]

    return data["price"].mean()

data["price"] = data.apply(new_price, axis=1)
data["price"] = data["price"].round()

data.to_csv("cleaned_version.csv", index=False)
