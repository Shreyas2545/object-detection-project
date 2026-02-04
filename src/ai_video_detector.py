import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import warnings
warnings.filterwarnings('ignore')


class AIVideoDetector:
    """
    AI Video Detector - Detects AI-generated vs Real videos
    Maximum video length: 10 seconds
    Analyzes temporal inconsistencies and frame artifacts
    """
    
    def __init__(self, model_path=None, max_duration=10):
        """
        Initialize AI Video Detector
        
        Args:
            model_path: Path to trained model (uses image detector if not available)
            max_duration: Maximum video duration in seconds (default: 10)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_duration = max_duration
        self.model = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Try to load custom model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Use image detector as fallback
            local_ckpt = os.path.join('checkpoints', 'ai_detector.pth')
            if os.path.exists(local_ckpt):
                try:
                    self._load_model(local_ckpt)
                except:
                    print("⚠️ Using frame-by-frame analysis (no model loaded)")
    
    def _load_model(self, model_path):
        """Load AI detection model"""
        try:
            print(f"🔄 Loading AI video detector from {model_path}...")
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict):
                model.load_state_dict(state)
            else:
                model = state
            
            self.model = model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.model = None
    
    def extract_frames(self, video_path, max_frames=30):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to analyze (default: 30)
        
        Returns:
            List of frames (numpy arrays)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Check duration
            if duration > self.max_duration:
                print(f"⚠️ Video duration ({duration:.1f}s) exceeds maximum ({self.max_duration}s)")
                print(f"   Analyzing first {self.max_duration} seconds only")
            
            # Calculate frame interval
            frames_to_analyze = min(max_frames, int(min(duration, self.max_duration) * fps))
            interval = max(1, total_frames // frames_to_analyze)
            
            frames = []
            frame_indices = []
            
            for i in range(0, total_frames, interval):
                if len(frames) >= max_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    frames.append(frame)
                    frame_indices.append(i)
            
            cap.release()
            
            print(f"📹 Extracted {len(frames)} frames from video (FPS: {fps:.1f}, Duration: {duration:.1f}s)")
            return frames, fps, duration
            
        except Exception as e:
            print(f"❌ Error extracting frames: {e}")
            return [], 0, 0
    
    def analyze_temporal_consistency(self, frames):
        """
        Analyze temporal consistency between frames
        AI videos often have:
        - Sudden color shifts
        - Inconsistent lighting
        - Unnatural motion patterns
        - Temporal artifacts
        """
        try:
            if len(frames) < 3:
                return None
            
            # Calculate frame differences
            frame_diffs = []
            color_shifts = []
            motion_inconsistencies = []
            
            for i in range(1, len(frames)):
                prev_frame = frames[i-1]
                curr_frame = frames[i]
                
                # Frame difference
                diff = cv2.absdiff(prev_frame, curr_frame)
                frame_diff = np.mean(diff)
                frame_diffs.append(frame_diff)
                
                # Color shift
                prev_mean = np.mean(prev_frame, axis=(0, 1))
                curr_mean = np.mean(curr_frame, axis=(0, 1))
                color_shift = np.linalg.norm(prev_mean - curr_mean)
                color_shifts.append(color_shift)
                
                # Motion analysis using optical flow
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_inconsistencies.append(np.std(motion_magnitude))
            
            # Calculate metrics
            frame_diff_std = np.std(frame_diffs)
            color_shift_std = np.std(color_shifts)
            motion_inconsistency_avg = np.mean(motion_inconsistencies)
            
            # Scoring
            temporal_score = 0
            explanations = []
            
            # Sudden changes indicate AI
            if frame_diff_std > 15:
                temporal_score += 20
                explanations.append(f"Inconsistent frame changes ({frame_diff_std:.1f})")
            
            if color_shift_std > 10:
                temporal_score += 15
                explanations.append(f"Unstable color grading ({color_shift_std:.1f})")
            
            if motion_inconsistency_avg > 8:
                temporal_score += 15
                explanations.append(f"Unnatural motion patterns ({motion_inconsistency_avg:.1f})")
            
            # Check for periodic artifacts (common in AI videos)
            diffs_fft = np.fft.fft(frame_diffs)
            diffs_power = np.abs(diffs_fft)**2
            if np.max(diffs_power[1:len(diffs_power)//2]) > np.mean(diffs_power) * 5:
                temporal_score += 10
                explanations.append("Periodic artifacts detected")
            
            return {
                'score': min(100, temporal_score),
                'frame_diff_std': float(frame_diff_std),
                'color_shift_std': float(color_shift_std),
                'motion_inconsistency': float(motion_inconsistency_avg),
                'explanations': explanations
            }
            
        except Exception as e:
            print(f"❌ Error in temporal analysis: {e}")
            return None
    
    def analyze_frame_artifacts(self, frames):
        """
        Analyze individual frames for AI generation artifacts
        Similar to image detection but adapted for video
        """
        try:
            ai_scores = []
            
            for frame in frames[::max(1, len(frames)//10)]:  # Sample 10 frames
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Noise analysis
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                noise_diff = gray.astype(float) - denoised.astype(float)
                noise_level = np.std(noise_diff)
                
                # Edge analysis
                edges = cv2.Canny(gray, 50, 150)
                edge_ratio = np.sum(edges > 0) / edges.size
                
                # Calculate AI score for this frame
                frame_ai_score = 0
                
                if noise_level < 3.0:
                    frame_ai_score += 30
                
                if edge_ratio < 0.05 or edge_ratio > 0.25:
                    frame_ai_score += 20
                
                ai_scores.append(frame_ai_score)
            
            avg_ai_score = np.mean(ai_scores)
            score_variance = np.std(ai_scores)
            
            # High variance in AI scores suggests AI generation
            if score_variance > 15:
                avg_ai_score += 10
            
            return {
                'score': min(100, avg_ai_score),
                'avg_frame_score': float(avg_ai_score),
                'score_variance': float(score_variance)
            }
            
        except Exception as e:
            print(f"❌ Error in frame artifact analysis: {e}")
            return None
    
    def predict_frames_with_model(self, frames):
        """Predict using deep learning model on sampled frames"""
        if self.model is None:
            return None
        
        try:
            ai_predictions = []
            
            # Sample frames (analyze every Nth frame)
            sample_interval = max(1, len(frames) // 15)
            sampled_frames = frames[::sample_interval][:15]
            
            for frame in sampled_frames:
                # Convert to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                
                # Transform and predict
                img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_class].item()
                    
                    # Assuming 0=Real, 1=AI
                    is_ai = (pred_class == 0)
                    ai_predictions.append(1.0 if is_ai else 0.0)
            
            # Calculate average
            ai_ratio = np.mean(ai_predictions)
            confidence = max(ai_ratio, 1 - ai_ratio) * 100
            is_ai = ai_ratio > 0.5
            
            return {
                'is_ai': is_ai,
                'confidence': float(confidence),
                'ai_frame_ratio': float(ai_ratio)
            }
            
        except Exception as e:
            print(f"❌ Error in model prediction: {e}")
            return None
    
    def predict(self, video_path):
        """
        Main prediction method
        
        Returns:
            Dictionary with detection results
        """
        results = {
            'is_ai_generated': False,
            'confidence': 0.0,
            'label': '',
            'verdict': '',
            'explanation': '',
            'metrics': {},
            'duration': 0.0,
            'frames_analyzed': 0
        }
        
        try:
            # Check file size (max ~50MB for 10 seconds)
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            if file_size > 100:
                results['verdict'] = f"❌ Video file too large ({file_size:.1f}MB). Maximum: 100MB"
                return results
            
            print(f"\n🎥 Analyzing video: {video_path}")
            print(f"   File size: {file_size:.1f}MB")
            
            # Extract frames
            frames, fps, duration = self.extract_frames(video_path)
            
            if len(frames) == 0:
                results['verdict'] = "❌ Could not extract frames from video"
                return results
            
            results['duration'] = duration
            results['frames_analyzed'] = len(frames)
            
            # === ANALYSIS 1: Temporal Consistency ===
            print("🔍 Analyzing temporal consistency...")
            temporal_result = self.analyze_temporal_consistency(frames)
            
            # === ANALYSIS 2: Frame Artifacts ===
            print("🔍 Analyzing frame artifacts...")
            artifact_result = self.analyze_frame_artifacts(frames)
            
            # === ANALYSIS 3: Model Prediction (if available) ===
            model_result = None
            if self.model is not None:
                print("🔍 Running deep learning model...")
                model_result = self.predict_frames_with_model(frames)
            
            # === COMBINE RESULTS ===
            final_score = 0
            explanations = []
            
            # Temporal analysis weight: 35%
            if temporal_result:
                final_score += temporal_result['score'] * 0.35
                explanations.extend(temporal_result.get('explanations', []))
                results['metrics'].update({
                    'temporal_score': temporal_result['score'],
                    'frame_diff_std': temporal_result['frame_diff_std'],
                    'color_shift_std': temporal_result['color_shift_std'],
                    'motion_inconsistency': temporal_result['motion_inconsistency']
                })
            
            # Artifact analysis weight: 30%
            if artifact_result:
                final_score += artifact_result['score'] * 0.30
                results['metrics'].update({
                    'artifact_score': artifact_result['score'],
                    'avg_frame_score': artifact_result['avg_frame_score'],
                    'score_variance': artifact_result['score_variance']
                })
            
            # Model prediction weight: 35%
            if model_result:
                model_score = model_result['confidence'] if model_result['is_ai'] else (100 - model_result['confidence'])
                final_score += model_score * 0.35
                explanations.append(f"Model prediction: {model_result['confidence']:.1f}% confidence")
                results['metrics']['model_confidence'] = model_result['confidence']
                results['metrics']['ai_frame_ratio'] = model_result['ai_frame_ratio']
            
            # Determine final result
            is_ai = final_score >= 50
            confidence = final_score if is_ai else (100 - final_score)
            
            results['is_ai_generated'] = is_ai
            results['confidence'] = float(confidence)
            results['label'] = 'AI Generated Video' if is_ai else 'Real Video'
            results['explanation'] = ' | '.join(explanations) if explanations else 'Analysis complete'
            
            # Generate verdict
            if is_ai:
                if confidence > 80:
                    results['verdict'] = f"⚠️ HIGHLY LIKELY AI-Generated Video ({confidence:.1f}% confidence)"
                elif confidence > 65:
                    results['verdict'] = f"⚠️ Likely AI-Generated Video ({confidence:.1f}% confidence)"
                else:
                    results['verdict'] = f"❓ Possibly AI-Generated Video ({confidence:.1f}% confidence)"
            else:
                if confidence > 80:
                    results['verdict'] = f"✅ HIGHLY LIKELY Real Video ({confidence:.1f}% confidence)"
                elif confidence > 65:
                    results['verdict'] = f"✅ Likely Real Video ({confidence:.1f}% confidence)"
                else:
                    results['verdict'] = f"❓ Possibly Real Video ({confidence:.1f}% confidence)"
            
            print(f"\n🎯 Analysis complete!")
            print(f"   Result: {results['label']}")
            print(f"   Confidence: {results['confidence']:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in video prediction: {e}")
            import traceback
            traceback.print_exc()
            results['verdict'] = f"❌ Error analyzing video: {str(e)}"
            return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING AI VIDEO DETECTOR")
    print("="*60 + "\n")
    
    detector = AIVideoDetector()
    
    test_video = "test_video.mp4"  # Replace with actual path
    
    if os.path.exists(test_video):
        result = detector.predict(test_video)
        
        print("\n🎯 RESULTS:")
        print("-" * 60)
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Verdict: {result['verdict']}")
        print(f"Duration: {result['duration']:.1f}s")
        print(f"Frames Analyzed: {result['frames_analyzed']}")
        
        if result.get('metrics'):
            print("\n📊 Technical Metrics:")
            for key, value in result['metrics'].items():
                print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
        
        if result.get('explanation'):
            print(f"\n💡 Explanation: {result['explanation']}")
    else:
        print(f"❌ Test video not found: {test_video}")
    
    print("\n" + "="*60 + "\n")