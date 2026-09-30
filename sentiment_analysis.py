import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_feedback_data():
    records = [
        ("The digital permits made entering very smooth and quick", "Positive"),
        (
            "Volunteers were extremely welcoming and helpful at the station",
            "Positive",
        ),
        (
            "Guidance signs were clear and multilingual across the path",
            "Positive",
        ),
        ("Clean facilities and well-organized shuttle service", "Positive"),
        ("Automated smart gates worked efficiently without delays", "Positive"),
        ("Long waiting times and heavy bottleneck at gate 4", "Negative"),
        (
            "Buses were delayed and crowd dispatching was uncoordinated",
            "Negative",
        ),
        (
            "Lack of clear signs caused severe confusion near luggage area",
            "Negative",
        ),
        (
            "Extreme crowding at water stations with no clear queue lanes",
            "Negative",
        ),
        ("Slow luggage handling and unhelpful counter staff", "Negative"),
    ]
    return pd.DataFrame(records, columns=["review_text", "sentiment"])


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression()),
    ])


def main():
    df = load_feedback_data()

    pipeline = build_pipeline()
    pipeline.fit(df["review_text"], df["sentiment"])

    sample_feedback = ["Bus arrived late and crowd management was chaotic"]
    prediction = pipeline.predict(sample_feedback)

    print(f"Sample: '{sample_feedback[0]}'")
    print(f"Predicted Sentiment: {prediction[0]}")

    counts = df["sentiment"].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(
        counts.index, counts.values, color=["#2ca02c", "#d62728"], width=0.4
    )
    plt.title("Pilgrim Feedback Sentiment Distribution")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.show()

    df.to_csv("pilgrim_sentiment_data.csv", index=False)


if __name__ == "__main__":
    main()
