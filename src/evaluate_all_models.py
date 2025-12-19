import subprocess

print("\n📊 Evaluating Deep Learning Models...\n")
subprocess.run(["python", "test_model.py"])

print("\n📊 Evaluating KNN Model...\n")
subprocess.run(["python", "knn_train_test.py"])
