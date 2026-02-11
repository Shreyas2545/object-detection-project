import sys
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ai_image_detector import AIImageDetector
except ImportError:
    print("Error: Could not import AIImageDetector. Run from project root.")
    sys.exit(1)

def create_test_image(filename, method='real'):
    """Create a dummy test image for verification"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    if method == 'real':
        # Add noise to simulate real image
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        cv2.putText(img, 'Real Photo', (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # Smooth image for AI simulation
        img[:] = (200, 200, 200) # Flat color
        cv2.circle(img, (112, 112), 50, (255, 0, 0), -1) # Perfect shape
        cv2.putText(img, 'AI Generated', (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
    cv2.imwrite(filename, img)
    return filename

def test_detector():
    print("="*60)
    print("🧪 VERIFYING AI DETECTOR")
    print("="*60)

    # Init detector
    try:
        detector = AIImageDetector(sensitivity='high', method='hybrid')
        print("✅ Detector initialized with High Sensitivity")
    except Exception as e:
        print(f"❌ Failed to init detector: {e}")
        return

    # Test 1: Real Image Simulation
    print("\n📸 Test 1: Simulating Real Image (High Noise)")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        real_img_path = tmp.name
    
    create_test_image(real_img_path, 'real')
    
    try:
        # Simulate app.py logic: Pass path to detector
        # The file is closed here because 'with' block ended for creating it, 
        # but we need to ensure detector handles it correctly
        
        start = time.time()
        result_real = detector.predict(real_img_path)
        end = time.time()
        
        print(f"⏱️ Time: {end - start:.2f}s")
        print(f"🏷️ Label: {result_real['label']}")
        print(f"📊 Confidence: {result_real['confidence']:.2f}%")
        print(f"🤔 Verdict: {result_real['verdict']}")
        print(f"ℹ️ Explanation: {result_real['explanation']}")
        
        if result_real['confidence'] == 0:
            print("❌ FAILURE: Confidence is 0%! Fix not working.")
        else:
            print("✅ SUCECSS: Non-zero confidence returned.")

    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        if os.path.exists(real_img_path):
            os.remove(real_img_path)

    # Test 2: AI Image Simulation
    print("\n🤖 Test 2: Simulating AI Image (Smooth/Perfect)")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        ai_img_path = tmp.name
    
    create_test_image(ai_img_path, 'ai')
    
    try:
        start = time.time()
        result_ai = detector.predict(ai_img_path)
        end = time.time()
        
        print(f"⏱️ Time: {end - start:.2f}s")
        print(f"🏷️ Label: {result_ai['label']}")
        print(f"📊 Confidence: {result_ai['confidence']:.2f}%")
        print(f"🤔 Verdict: {result_ai['verdict']}")
        print(f"ℹ️ Explanation: {result_ai['explanation']}")
        
        if result_ai['label'] == 'AI Generated':
            print("✅ SUCCESS: Correctly identified as AI.")
        else:
            print("⚠️ WARNING: AI Image identified as Real. Sensitivity might be too low.")
            print(f"   Debug Info: {result_ai}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        if os.path.exists(ai_img_path):
            os.remove(ai_img_path)

if __name__ == "__main__":
    test_detector()
