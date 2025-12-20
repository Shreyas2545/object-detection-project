import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X_train, y_train, X_test, y_test):
    print("\n🚀 Training SVM Model...\n")

    svm_model = SVC(
        kernel='rbf',       # best default kernel
        C=10,               # regularization
        gamma='scale'       # kernel coefficient
    )

    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ SVM Accuracy: {acc * 100:.2f}%")

    # Save model
    joblib.dump(svm_model, "checkpoints/svm_model.pkl")
    print("💾 SVM model saved successfully!")

    return svm_model
