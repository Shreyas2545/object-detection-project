import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_svm_and_get_accuracy(X_train, y_train, X_test, y_test):
    print("\n🚀 Training SVM Model...\n")

    svm_model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True   # 🔥 REQUIRED for confidence
    )

    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"🎯 SVM Accuracy: {acc * 100:.2f}%")

    joblib.dump(svm_model, "checkpoints/svm_model.pkl")
    print("💾 SVM model saved")

    return acc
