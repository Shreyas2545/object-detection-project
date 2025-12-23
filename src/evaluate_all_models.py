import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# =========================
# CLASS NAMES
# =========================
class_names = ["birds", "cars", "cats", "dogs", "human", "watches"]

# =========================
# LOAD FEATURES
# =========================
X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_test  = np.load("features/X_test.npy")
y_test  = np.load("features/y_test.npy")

print("✅ Loaded feature data for SVM")
print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


def run_svm_and_get_accuracy():
    # =========================
    # INITIALIZE SVM
    # =========================
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,   # REQUIRED for confidence scores
        random_state=42
    )

    # =========================
    # TRAIN
    # =========================
    print("\n🚀 Training SVM...\n")
    svm.fit(X_train, y_train)

    # =========================
    # TEST (DL-LIKE OUTPUT)
    # =========================
    print("\n🧪 Testing SVM...\n")

    correct = 0
    total = len(y_test)

    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, -1)
        actual = y_test[i]

        pred = svm.predict(sample)[0]
        probs = svm.predict_proba(sample)
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
    # REPORTS
    # =========================
    y_pred = svm.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # SAVE MODEL
    # =========================
    joblib.dump(svm, "checkpoints/svm_model.pkl")
    print("💾 SVM model saved")

    return accuracy


# =========================
# RUN DIRECTLY (OPTIONAL)
# =========================
if __name__ == "__main__":
    run_svm_and_get_accuracy()
