#### Introduction to Data Science Group Project

# Cork Sniffer

Wine recommendation system using machine learning.

## Team members

- [@joonarafael](https://github.com/joonarafael)
- [@filippahognasbacka](https://github.com/filippahognasbacka)
- [@aarekr](https://github.com/aarekr)

## Get Started

Create a virtual environment and install the dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To check outdated dependencies, run:

```bash
pip-review
```

To update outdated dependencies, run:

```bash
pip-review --local --interactive
```

To start the web app, run (in chat-app folder):

```bash
flask run
```

## Source Data

Wine data is fetched from [Label Your Data](https://labelyourdata.com/datasets/wine-review-dataset "Wine Review Dataset | Label Your Data"). Raw file: winemag-data-130k-v2.csv

Wine food pairing data is fetched from [Wolfram Wine Food Pairing](https://resources.wolframcloud.com/FunctionRepository/resources/WineFoodPairing) by installing the Wolfram language.

## Demo Video

[Data Science 2025 demo video Cork Sniffer](https://youtu.be/2XmZGpOC66E)
