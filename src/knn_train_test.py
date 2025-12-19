import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ===== LOAD FEATURE DATA =====
X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")

X_test = np.load("features/X_test.npy")
y_test = np.load("features/y_test.npy")

print("✅ Loaded feature data")
print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ===== MAIN KNN FUNCTION (used by evaluate_all_models.py) =====
def run_knn_and_get_accuracy():
    # ===== INITIALIZE KNN =====
    knn = KNeighborsClassifier(
        n_neighbors=5,          # number of nearest neighbors
        metric="euclidean",     # distance metric
        weights="distance"      # closer points have more influence
    )

    # ===== TRAIN KNN =====
    print("\n🚀 Training KNN model...")
    knn.fit(X_train, y_train)

    # ===== TEST KNN =====
    print("\n🧪 Testing KNN model...")
    y_pred = knn.predict(X_test)

    # ===== EVALUATION =====
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 KNN Accuracy: {accuracy * 100:.2f}%\n")

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ===== SAVE MODEL =====
    joblib.dump(knn, "knn_model.pkl")
    print("\n💾 KNN model saved as knn_model.pkl")

    return accuracy


# ===== RUN ONLY WHEN THIS FILE IS EXECUTED DIRECTLY =====
if __name__ == "__main__":
    run_knn_and_get_accuracy()
