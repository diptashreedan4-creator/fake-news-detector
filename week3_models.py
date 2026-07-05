import pickle
import scipy.sparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading TF-IDF features and labels...")

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X_tfidf = scipy.sparse.load_npz("X_tfidf.npz")
y = pd.read_csv("y_labels.csv")["label"]

print("Feature matrix shape:", X_tfidf.shape)
print("Labels shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

results = {}

for name, model in models.items():
    print("=" * 50)
    print("Training " + name + "...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(name + " Accuracy: " + str(acc))
    print(name + " Classification Report:")
    print(classification_report(y_test, preds))
    print(name + " Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    with open("model_" + name + ".pkl", "wb") as f:
        pickle.dump(model, f)

print("=" * 50)
print("SUMMARY: Accuracy of all models")
for name, acc in results.items():
    print("  " + name + ": " + str(acc))

best_model = max(results, key=results.get)
print("Best performing model: " + best_model + " (" + str(results[best_model]) + ")")

scipy.sparse.save_npz("X_test.npz", X_test)
y_test.to_csv("y_test.csv", index=False)
print("Saved X_test.npz and y_test.csv for Week 4.")
print("Week 3 complete!")
