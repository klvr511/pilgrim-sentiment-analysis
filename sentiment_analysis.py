import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = [
    ("The digital permits made entering very smooth and quick", "Positive"),
    ("Volunteers were extremely welcoming and helpful at the station", "Positive"),
    ("Guidance signs were clear and multilingual across the path", "Positive"),
    ("Clean facilities and well-organized shuttle service", "Positive"),
    ("Automated smart gates worked efficiently without delays", "Positive"),
    ("Long waiting times and heavy bottleneck at gate 4", "Negative"),
    ("Buses were delayed and crowd dispatching was uncoordinated", "Negative"),
    ("Lack of clear signs caused severe confusion near luggage area", "Negative"),
    ("Extreme crowding at water stations with no clear queue lanes", "Negative"),
    ("Slow luggage handling and unhelpful counter staff", "Negative"),
]

df = pd.DataFrame(data, columns=["review_text", "sentiment"])

model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression()),
])

X = df["review_text"]
y = df["sentiment"]
model_pipeline.fit(X, y)

sample_feedback = ["Bus arrived late and crowd management was chaotic"]
predicted_sentiment = model_pipeline.predict(sample_feedback)
print(f"Sample: '{sample_feedback[0]}'")
print(f"Predicted Sentiment: {predicted_sentiment[0]}")

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
print("Project generated successfully!")
