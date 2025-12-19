import subprocess
import sys
import os

# Get absolute path of the src folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n📊 Evaluating Deep Learning Models...\n")
subprocess.run([sys.executable, os.path.join(BASE_DIR, "test_model.py")])

print("\n📊 Evaluating KNN Model...\n")
subprocess.run([sys.executable, os.path.join(BASE_DIR, "knn_train_test.py")])
