import pickle
import scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

print("Loading test data and trained models...")

X_test = scipy.sparse.load_npz("X_test.npz")
y_test = pd.read_csv("y_test.csv")["label"]

model_names = ["KNN", "LogisticRegression", "RandomForest", "NeuralNetwork"]
models = {}
for name in model_names:
    with open("model_" + name + ".pkl", "rb") as f:
        models[name] = pickle.load(f)

results = []
predictions = {}

for name, model in models.items():
    preds = model.predict(X_test)
    predictions[name] = preds
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
    print(name + " -> Accuracy: " + str(round(acc, 4)) + ", Precision: " + str(round(prec, 4)) + ", Recall: " + str(round(rec, 4)) + ", F1: " + str(round(f1, 4)))

results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_metrics.csv", index=False)
print("Saved model_comparison_metrics.csv")

plt.figure(figsize=(9, 6))
x = range(len(results_df))
plt.bar(x, results_df["Accuracy"], color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
plt.xticks(x, results_df["Model"])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
for i, v in enumerate(results_df["Accuracy"]):
    plt.text(i, v + 0.02, str(round(v, 3)), ha="center", fontweight="bold")
plt.savefig("model_accuracy_comparison.png")
plt.close()
print("Saved chart: model_accuracy_comparison.png")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for idx, name in enumerate(model_names):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices_all_models.png")
plt.close()
print("Saved chart: confusion_matrices_all_models.png")

plt.figure(figsize=(10, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = range(len(results_df))
width = 0.2
for i, metric in enumerate(metrics):
    plt.bar([p + width * i for p in x], results_df[metric], width=width, label=metric)
plt.xticks([p + width * 1.5 for p in x], results_df["Model"])
plt.ylabel("Score")
plt.title("Model Comparison Across All Metrics")
plt.legend()
plt.ylim(0, 1)
plt.savefig("model_all_metrics_comparison.png")
plt.close()
print("Saved chart: model_all_metrics_comparison.png")

print("Week 4 complete! All charts and metrics saved for your report.")
