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
    
    def __init__(self, model_path=None, max_duration=10, max_frames=30, model_frame_count=15, analysis_resize=(320,240), hotspot_threshold=0.92, hotspot_boost=30, model_min_ai_ratio=0.25):
        """
        Initialize AI Video Detector
        
        Args:
            model_path: Path to trained model (uses image detector if not available)
            max_duration: Maximum video duration in seconds (default: 10)
            max_frames: Maximum number of frames to extract/analyze
            model_frame_count: Number of frames the DL model should evaluate (sampled)
            analysis_resize: Resize used for temporal/motion analysis to speed optical flow
            hotspot_threshold: Per-frame AI probability required to consider a hotspot
            hotspot_boost: Score boost applied when hotspot is detected
            model_min_ai_ratio: Minimum AI-frame ratio required for model evidence to count
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_duration = max_duration
        # Operational tuning (can be adjusted via environment variables)
        self.max_frames = int(os.getenv('AI_VIDEO_MAX_FRAMES', max_frames))
        self.model_frame_count = int(os.getenv('AI_VIDEO_MODEL_FRAMES', model_frame_count))
        # Hotspot configuration
        self.hotspot_threshold = float(os.getenv('AI_VIDEO_HOTSPOT_THRESHOLD', hotspot_threshold))
        # Tighten hotspot to require higher per-frame confidence to count as a hotspot
        self.hotspot_boost = float(os.getenv('AI_VIDEO_HOTSPOT_BOOST', hotspot_boost))
        # Model-level minimum AI ratio to consider model evidence (increase for precision)
        self.model_min_ai_ratio = float(os.getenv('AI_VIDEO_MODEL_MIN_RATIO', 0.40))
        # Temporal model hotspot config (if temporal model strongly predicts AI, boost final score)
        self.temporal_hotspot_threshold = float(os.getenv('AI_VIDEO_TEMPORAL_HOTSPOT_THRESHOLD', 0.85))
        self.temporal_hotspot_boost = float(os.getenv('AI_VIDEO_TEMPORAL_HOTSPOT_BOOST', 50.0))

        resize_env = os.getenv('AI_VIDEO_ANALYSIS_RESIZE', None)
        if resize_env:
            try:
                w,h = [int(x) for x in resize_env.split('x')]
                self.analysis_resize = (w,h)
            except Exception:
                self.analysis_resize = analysis_resize
        else:
            self.analysis_resize = analysis_resize

        self.model = None
        
        # Image preprocessing for DL model (keep original size for model accuracy)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"🛠️ Video detector config: max_frames={self.max_frames}, model_frame_count={self.model_frame_count}, analysis_resize={self.analysis_resize}, device={self.device}")

# Try to load supplied model, else prefer fine-tuned checkpoint if present
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Prefer fine-tuned checkpoint if available
            fine_ckpt = os.path.join('checkpoints', 'ai_detector_finetuned.pth')
            local_ckpt = os.path.join('checkpoints', 'ai_detector.pth')
            ckpt_to_use = None
            if os.path.exists(fine_ckpt):
                ckpt_to_use = fine_ckpt
            elif os.path.exists(local_ckpt):
                ckpt_to_use = local_ckpt
            if ckpt_to_use:
                try:
                    self._load_model(ckpt_to_use)
                except Exception:
                    print("⚠️ Using frame-by-frame analysis (no model loaded)")
            else:
                print("⚠️ No image model checkpoint found; running heuristic-only detector")
    
    def _load_model(self, model_path):
        """Load AI detection model"""
        try:
            print(f"🔄 Loading AI video detector from {model_path}...")
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            
            state = torch.load(model_path, map_location=self.device)
            # Support both plain state_dict and saved checkpoint dicts
            if isinstance(state, dict) and 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
                # If temperature saved, store it
                self.model_temperature = float(state.get('temperature', 1.0))
            elif isinstance(state, dict):
                # Old style: assume state_dict
                model.load_state_dict(state)
            else:
                model = state
            
            self.model = model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.model = None
    
    def extract_frames(self, video_path, max_frames=None):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to analyze (default: self.max_frames)
        
        Returns:
            List of frames (numpy arrays), fps, duration
        """
        if max_frames is None:
            max_frames = self.max_frames

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
            
            # Calculate number of frames to analyze and interval
            frames_to_analyze = min(max_frames, int(min(duration, self.max_duration) * fps) if fps>0 else max_frames)
            if frames_to_analyze <= 0:
                frames_to_analyze = min(max_frames, total_frames)
            interval = max(1, max(1, total_frames // frames_to_analyze))
            
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
            
            # Downscale frames for faster optical flow / motion analysis
            small_prev = cv2.resize(frames[0], self.analysis_resize)
            frame_diffs = []
            color_shifts = []
            motion_inconsistencies = []
            
            for i in range(1, len(frames)):
                prev_frame = cv2.resize(frames[i-1], self.analysis_resize)
                curr_frame = cv2.resize(frames[i], self.analysis_resize)
                
                # Frame difference
                diff = cv2.absdiff(prev_frame, curr_frame)
                frame_diff = np.mean(diff)
                frame_diffs.append(frame_diff)
                
                # Color shift
                prev_mean = np.mean(prev_frame, axis=(0, 1))
                curr_mean = np.mean(curr_frame, axis=(0, 1))
                color_shift = np.linalg.norm(prev_mean - curr_mean)
                color_shifts.append(color_shift)
                
                # Motion analysis using optical flow (on smaller frames for speed)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Use slightly faster/less-precise parameters for Farneback
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.4, 2, 9, 2, 5, 1.1, 0)
                motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_inconsistencies.append(np.std(motion_magnitude))
            
            # Calculate metrics
            frame_diff_std = np.std(frame_diffs)
            color_shift_std = np.std(color_shifts)
            motion_inconsistency_avg = np.mean(motion_inconsistencies) if motion_inconsistencies else 0.0
            
            # Scoring
            temporal_score = 0
            explanations = []
            
            # Sudden changes indicate AI (lowered thresholds for higher sensitivity)
            # Stricter thresholds to reduce false positives (require larger temporal anomalies)
            if frame_diff_std > 6.0:
                temporal_score += 20
                explanations.append(f"Inconsistent frame changes ({frame_diff_std:.1f})")
            
            if color_shift_std > 3.0:
                temporal_score += 15
                explanations.append(f"Unstable color grading ({color_shift_std:.1f})")
            
            if motion_inconsistency_avg > 4.0:
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
            
            # Sample up to 10 frames evenly
            sample_count = min(10, max(1, len(frames)))
            indices = np.linspace(0, len(frames)-1, sample_count, dtype=int)
            for idx in indices:
                frame = frames[idx]
                # Work on a downscaled grayscale to speed up processing
                small = cv2.resize(frame, (min(320, frame.shape[1]), min(240, frame.shape[0])))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                
                # Noise analysis
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                noise_diff = gray.astype(float) - denoised.astype(float)
                noise_level = np.std(noise_diff)
                
                # Edge analysis (faster at smaller scale)
                edges = cv2.Canny(gray, 50, 150)
                edge_ratio = np.sum(edges > 0) / edges.size
                
                # Pixel variance check to avoid flagging uniform frames (e.g., static test patterns)
                frame_variance = float(np.var(gray))

                # Calculate AI score for this frame
                frame_ai_score = 0

                # Adjusted artifact thresholds to be less permissive and avoid false positives
                if frame_variance < 5.0 and edge_ratio < 0.02:
                    # Very uniform frame — ambiguous, do not mark as AI by artifact heuristics
                    frame_ai_score = 0
                else:
                    if noise_level < 5.0:
                        frame_ai_score += 30

                    # Require more extreme edge-ratios to consider frame anomalous
                    if edge_ratio < 0.05 or edge_ratio > 0.25:
                        frame_ai_score += 20

                # Append per-frame AI score
                ai_scores.append(frame_ai_score)
            
            avg_ai_score = np.mean(ai_scores) if ai_scores else 0.0
            score_variance = np.std(ai_scores) if ai_scores else 0.0
            
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
        """Predict using deep learning model on sampled frames (batched inference)"""
        if self.model is None:
            return None
        
        try:
            # Determine sample size and indices
            sample_count = min(self.model_frame_count, max(1, len(frames)))
            sample_interval = max(1, len(frames) // sample_count)
            sampled_frames = frames[::sample_interval][:sample_count]

            tensors = []
            for frame in sampled_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_tensor = self.transform(img_pil)
                tensors.append(img_tensor)

            if not tensors:
                return None

            batch = torch.stack(tensors, dim=0).to(self.device)

            with torch.no_grad():
                # Use mixed precision if GPU available for faster inference
                if self.device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            # Use the mean AI-class probability across sampled frames (smoother, more precise)
            frame_probs = [float(p[1]) for p in probs]
            ai_ratio = float(np.mean(frame_probs))  # average probability for AI class (0..1)
            confidence = float(ai_ratio * 100.0)
            is_ai = ai_ratio > 0.5

            # Preds (for diagnostic purposes) still provided
            preds = np.argmax(probs, axis=1)

            return {
                'is_ai': is_ai,
                'confidence': confidence,
                'ai_frame_ratio': ai_ratio,
                'frame_probs': frame_probs,
                'preds': preds.tolist()
            }

        except Exception as e:
            print(f"❌ Error in model prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, video_path):
        """
        Main prediction method
        
        Returns:
            Dictionary with detection results
        """
        import time
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
            start_total = time.perf_counter()

            # Check file size (max ~100MB allowed)
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            if file_size > 100:
                results['verdict'] = f"❌ Video file too large ({file_size:.1f}MB). Maximum: 100MB"
                return results
            
            print(f"\n🎥 Analyzing video: {video_path}")
            print(f"   File size: {file_size:.1f}MB")
            
            # Extract frames (uses self.max_frames)
            t0 = time.perf_counter()
            frames, fps, duration = self.extract_frames(video_path)
            t1 = time.perf_counter()
            print(f"   Frame extraction: {(t1 - t0):.3f}s")
            
            if len(frames) == 0:
                results['verdict'] = "❌ Could not extract frames from video"
                return results
            
            results['duration'] = duration
            results['frames_analyzed'] = len(frames)
            
            # === ANALYSIS 1: Temporal Consistency ===
            print("🔍 Analyzing temporal consistency...")
            t0 = time.perf_counter()
            temporal_result = self.analyze_temporal_consistency(frames)
            t1 = time.perf_counter()
            print(f"   Temporal analysis: {(t1 - t0):.3f}s")
            
            # === ANALYSIS 2: Frame Artifacts ===
            print("🔍 Analyzing frame artifacts...")
            t0 = time.perf_counter()
            artifact_result = self.analyze_frame_artifacts(frames)
            t1 = time.perf_counter()
            print(f"   Frame artifact analysis: {(t1 - t0):.3f}s")
            
            # === ANALYSIS 3: Model Prediction (if available) ===
            model_result = None
            if self.model is not None:
                print("🔍 Running deep learning model (batched)...")
                t0 = time.perf_counter()
                model_result = self.predict_frames_with_model(frames)
                t1 = time.perf_counter()
                print(f"   Model inference: {(t1 - t0):.3f}s")

            # === ANALYSIS 4: Temporal ML Model (if available) ===
            temporal_ml_prob = None
            try:
                # Try both import styles to support running from project root or as package
                try:
                    from temporal_model import TemporalModel
                except Exception:
                    from src.temporal_model import TemporalModel

                temporal_model_ckpt = os.path.join('checkpoints', 'temporal_model.joblib')
                if os.path.exists(temporal_model_ckpt):
                    t0 = time.perf_counter()
                    tm = TemporalModel()
                    tm.load(temporal_model_ckpt)
                    temporal_ml_prob = tm.predict_proba(video_path)
                    t1 = time.perf_counter()
                    print(f"   Temporal ML inference: {(t1 - t0):.3f}s (AI_prob: {temporal_ml_prob:.3f})")
            except Exception as e:
                # Log for debugging so failures are visible
                print(f"⚠️ Temporal model could not be loaded or executed: {e}")
                temporal_ml_prob = None
            
            # === COMBINE RESULTS ===
            final_score = 0
            explanations = []
            
            # Temporal analysis weight: 25%
            if temporal_result:
                final_score += temporal_result['score'] * 0.25
                explanations.extend(temporal_result.get('explanations', []))
                results['metrics'].update({
                    'temporal_score': temporal_result['score'],
                    'frame_diff_std': temporal_result['frame_diff_std'],
                    'color_shift_std': temporal_result['color_shift_std'],
                    'motion_inconsistency': temporal_result['motion_inconsistency']
                })

            # If we have a learned temporal ML model, use its probability (weighted) as additional evidence
            if temporal_ml_prob is not None:
                # weight configurable by future env var; default 30% of final scoring
                temporal_weight = float(os.getenv('AI_VIDEO_TEMPORAL_WEIGHT', 0.30))
                final_score += float(temporal_ml_prob * 100.0) * temporal_weight
                explanations.append(f"Temporal-ML model AI probability: {temporal_ml_prob*100:.1f}%")
                results['metrics']['temporal_ml_ai_prob'] = float(temporal_ml_prob)
            
            # Artifact analysis weight: 30%
            if artifact_result:
                final_score += artifact_result['score'] * 0.30
                results['metrics'].update({
                    'artifact_score': artifact_result['score'],
                    'avg_frame_score': artifact_result['avg_frame_score'],
                    'score_variance': artifact_result['score_variance']
                })
            
            # Prepare "Top 3 Models" equivalent for UI display
            # We map our 3 analysis components to this structure so the frontend can display them nicely
            top_3_models = []
            
            # 1. Temporal Analysis
            if temporal_result:
                temp_conf = temporal_result['score']
                temp_label = "AI-Generated" if temp_conf > 50 else "Real Video"
                top_3_models.append({
                    "model": "Temporal Analysis",
                    "confidence": float(temp_conf),
                    "prediction": temp_label
                })

            # 2. Artifact Analysis
            if artifact_result:
                art_conf = artifact_result['score']
                art_label = "AI-Generated" if art_conf > 50 else "Real Video"
                top_3_models.append({
                    "model": "Frame Artifacts",
                    "confidence": float(art_conf),
                    "prediction": art_label
                })

            # 3. Deep Learning Model
            
            # Model prediction weight: 35%
            if model_result:
                ai_ratio = float(model_result.get('ai_frame_ratio', 0.0))
                results['metrics']['ai_frame_ratio'] = ai_ratio

                # Only count model evidence if a minimum fraction of frames are predicted AI (helps avoid false positives)
                model_score = ai_ratio * 100.0  # proportion -> percent
                
                if ai_ratio >= self.model_min_ai_ratio:
                    final_score += model_score * 0.35
                    explanations.append(f"Model prediction: {model_score:.1f}% AI-frame ratio")
                    results['metrics']['model_confidence'] = model_score
                else:
                    # Model did not show sufficient consensus to be strong evidence
                    explanations.append(f"Model: low AI-frame ratio ({model_score:.1f}%)")
                    results['metrics']['model_confidence'] = model_score

                # Additional rule: if any single sampled frame has very high AI probability, treat as a hotspot and boost score
                max_frame_ai_prob = max(model_result.get('frame_probs', [0.0]))
                results['metrics']['model_max_ai_prob'] = float(max_frame_ai_prob)
                # Hotspot rule: configurable threshold + boost to catch localized strong AI evidence
                if max_frame_ai_prob > self.hotspot_threshold:
                    # Strong per-frame evidence — add configurable boost
                    final_score += self.hotspot_boost
                    explanations.append(f"High AI-probability frame detected ({max_frame_ai_prob*100:.1f}%)")
                # Include config in metrics for traceability
                    results['metrics']['hotspot_threshold'] = float(self.hotspot_threshold)
                    results['metrics']['hotspot_boost'] = float(self.hotspot_boost)

                # Add to Top 3
                dl_label = "AI-Generated" if ai_ratio > 0.5 else "Real Video"
                top_3_models.append({
                    "model": "Deep Learning Model",
                    "confidence": float(model_score if ai_ratio >= self.model_min_ai_ratio else (100 - model_score)),
                    "prediction": dl_label
                })
            
            # Determine final result
            final_score = min(100.0, final_score)
            is_ai = final_score >= 50
            confidence = final_score if is_ai else (100 - final_score)
            
            results['is_ai_generated'] = is_ai
            results['confidence'] = float(confidence)
            results['label'] = 'AI Generated Video' if is_ai else 'Real Video'
            results['explanation'] = ' | '.join(explanations) if explanations else 'Analysis complete'
            
            # Include synthesized Top 3 Models for frontend compatibility
            # Sort by confidence
            top_3_models.sort(key=lambda x: x['confidence'], reverse=True)
            results['Top 3 Models'] = top_3_models

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
            
            t_end = time.perf_counter()
            print(f"\n🎯 Analysis complete! Total time: {(t_end - start_total):.3f}s")
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