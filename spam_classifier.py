import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Label encoding
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Custom test
text = "Congratulations! You won a free iPhone!"
vec = vectorizer.transform([text])
print("Prediction:", "Spam" if model.predict(vec)[0] == 1 else "Not Spam")
