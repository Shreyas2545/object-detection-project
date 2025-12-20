import joblib
from sklearn.svm import SVC

# =========================
# CLASS NAMES (SAME AS DL)
# =========================
class_names = ["birds", "cars", "cats", "dogs", "human", "watches"]


def train_svm(X_train, y_train, X_test, y_test):
    print("\n🚀 Training SVM Model...\n")

    # =========================
    # INITIALIZE SVM
    # =========================
    svm_model = SVC(
        kernel="linear",
        C=0.1,
        gamma="scale",
        probability=True   # REQUIRED for confidence output
    )

    # =========================
    # TRAIN
    # =========================
    svm_model.fit(X_train, y_train)

    # =========================
    # TEST (DL-LIKE OUTPUT)
    # =========================
    print("\n🧪 Testing SVM model...\n")

    correct = 0
    total = len(y_test)

    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, -1)
        actual = y_test[i]

        pred = svm_model.predict(sample)[0]
        probs = svm_model.predict_proba(sample)
        conf = probs[0][pred] * 100

        if pred == actual:
            correct += 1

        print(
            f"🧮 Predicted: {class_names[pred]} ({conf:.2f}%) | "
            f"Actual: {class_names[actual]}"
        )

    accuracy = correct / total
    print(f"\n🎯 SVM Accuracy: {accuracy * 100:.2f}%\n")

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(svm_model, "checkpoints/svm_model.pkl")
    print("💾 SVM model saved successfully!")

    return accuracy
