from flask import Flask, request, render_template
import spacy
import pandas as pd
import os

# Load spaCy Polish NLP model
nlp = spacy.load("pl_core_news_sm")

# Load tag list from CSV
file_path = os.path.join(os.path.dirname(__file__), "TagListPL.csv")
df = pd.read_csv(file_path, encoding="utf-8", sep=",")

# Convert CSV to dictionary {word: tag}
tag_dict = dict(zip(df["word"], df["tag"]))

def lemmatize_and_tag(text):
    """Lemmatize words using spaCy and assign custom tags from CSV."""
    doc = nlp(text)
    tagged_output = []

    for token in doc:
        lemma = token.lemma_.lower()  # Lemmatize and convert to lowercase
        tag = tag_dict.get(lemma, "UNKNOWN")  # Find tag in dictionary, default to "UNKNOWN"
        tagged_output.append((token.text, lemma, tag))  # (Original word, Lemma, Tag)

    return tagged_output

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        result = lemmatize_and_tag(text)
        return render_template("index.html", text=text, result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)