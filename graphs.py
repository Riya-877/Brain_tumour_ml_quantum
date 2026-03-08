import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "SVM",
    "Quantum Model"
]

accuracy = [0.85, 0.90, 0.82, 0.88, 0.91]  # replace with your real results

plt.figure(figsize=(8,5))
plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")

plt.xticks(rotation=30)

plt.show()