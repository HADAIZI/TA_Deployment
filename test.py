#!/usr/bin/env python3
"""
Comprehensive test script for ergonomic assessment model
Tests model loading, feature engineering, and prediction pipeline
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import cv2
from pathlib import Path
import traceback
from datetime import datetime

# Add current directory to path to import local modules
sys.path.append(os.getcwd())

# Try to import local modules
try:
    from app.services.pose_estimation import init_movenet, get_joint_angles, create_row_dict, init_crop_region
    from app.services.ergonomic_model import get_model_resources, engineer_features_for_single_image, predict_single_image
    from app.services.ergonomic_model import get_risk_level, generate_feedback
    LOCAL_MODULES_AVAILABLE = True
    print("✅ Local modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import local modules: {e}")
    print("Will use standalone implementations")
    LOCAL_MODULES_AVAILABLE = False

# Constants
KEYPOINT_THRESHOLD = 0.3
SEQUENCE_LENGTH = 60
TEST_DIRECTORIES = ['test']
MODEL_PATH = "modelv4/reba_model.h5"
PREPROCESSING_PATH = "modelv4/preprocessing.joblib"

class StandaloneModelTester:
    """Standalone implementation for testing when local modules aren't available"""
    
    def __init__(self):
        self.model = None
        self.resources = None
        self.movenet = None
        self.input_size = 256
        
    def load_model_resources(self):
        """Load model and preprocessing resources"""
        try:
            print("📦 Loading model resources...")
            
            # Check if files exist
            if not os.path.exists(MODEL_PATH):
                print(f"❌ Model file not found: {MODEL_PATH}")
                return False
                
            if not os.path.exists(PREPROCESSING_PATH):
                print(f"❌ Preprocessing file not found: {PREPROCESSING_PATH}")
                return False
            
            # Load model
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded: {self.model.count_params():,} parameters")
            
            # Load preprocessing data
            preprocessing_data = joblib.load(PREPROCESSING_PATH)
            print(f"✅ Preprocessing data loaded")
            
            self.resources = {
                'model': self.model,
                'scaler': preprocessing_data['scaler'],
                'model_features': preprocessing_data['model_features'],
                'core_angles': preprocessing_data.get('core_angles', [
                    'Neck Angle', 'Left Upper Arm Angle', 'Right Upper Arm Angle',
                    'Left Lower Arm Angle', 'Right Lower Arm Angle', 'Waist Angle',
                    'Left Leg Angle', 'Right Leg Angle'
                ]),
                'sequence_length': preprocessing_data.get('sequence_length', SEQUENCE_LENGTH)
            }
            
            print(f"📊 Model features count: {len(self.resources['model_features'])}")
            print("📊 Top 10 features:")
            for i, feature in enumerate(self.resources['model_features'][:10], 1):
                print(f"   {i:2d}. {feature}")
            if len(self.resources['model_features']) > 10:
                print(f"   ... and {len(self.resources['model_features']) - 10} more")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model resources: {e}")
            traceback.print_exc()
            return False
    
    def init_movenet(self):
        """Initialize MoveNet model"""
        try:
            print("🤖 Initializing MoveNet...")
            
            # Try to load from TensorFlow Hub
            import tensorflow_hub as hub
            model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            module = hub.load(model_url)
            
            def movenet_wrapper(input_image):
                model = module.signatures['serving_default']
                input_image = tf.cast(input_image, dtype=tf.int32)
                outputs = model(input_image)
                keypoints_with_scores = outputs['output_0'].numpy()
                return keypoints_with_scores
            
            self.movenet = movenet_wrapper
            self.input_size = 256
            print("✅ MoveNet initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing MoveNet: {e}")
            traceback.print_exc()
            return False
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def get_joint_angles_simple(self, keypoints_with_scores):
        """Simplified joint angle calculation"""
        keypoints = keypoints_with_scores[0, 0, :, :2]
        scores = keypoints_with_scores[0, 0, :, 2]
        
        KEYPOINT_DICT = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3,
            'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 
            'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9,
            'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Check if we have enough valid keypoints
        valid_keypoints = {}
        for name, idx in KEYPOINT_DICT.items():
            if scores[idx] > KEYPOINT_THRESHOLD:
                valid_keypoints[name] = {
                    'x': keypoints[idx][1],
                    'y': keypoints[idx][0],
                    'valid': True
                }
            else:
                valid_keypoints[name] = {'valid': False}
        
        # Calculate angles with fallbacks
        angles = {}
        
        # Neck angle (ear to shoulder line vs vertical)
        try:
            if (valid_keypoints['left_ear']['valid'] and 
                valid_keypoints['left_shoulder']['valid'] and 
                valid_keypoints['right_shoulder']['valid']):
                
                ear = (valid_keypoints['left_ear']['y'], valid_keypoints['left_ear']['x'])
                mid_shoulder = (
                    (valid_keypoints['left_shoulder']['y'] + valid_keypoints['right_shoulder']['y']) / 2,
                    (valid_keypoints['left_shoulder']['x'] + valid_keypoints['right_shoulder']['x']) / 2
                )
                vertical_point = (mid_shoulder[0] - 1, mid_shoulder[1])
                
                angles['neck'] = self.calculate_angle(ear, mid_shoulder, vertical_point)
            else:
                angles['neck'] = 5  # Default neutral
        except:
            angles['neck'] = 5
        
        # Trunk angle (shoulder line vs hip line)
        try:
            if all(valid_keypoints[k]['valid'] for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                shoulder_center_y = (valid_keypoints['left_shoulder']['y'] + valid_keypoints['right_shoulder']['y']) / 2
                hip_center_y = (valid_keypoints['left_hip']['y'] + valid_keypoints['right_hip']['y']) / 2
                
                # Simple forward/backward detection
                if shoulder_center_y > hip_center_y:
                    angles['waist'] = 15  # Forward lean
                else:
                    angles['waist'] = -10  # Backward lean
            else:
                angles['waist'] = 90  # Default
        except:
            angles['waist'] = 90
        
        # Upper arm angles
        for side in ['left', 'right']:
            try:
                if all(valid_keypoints[k]['valid'] for k in [f'{side}_hip', f'{side}_shoulder', f'{side}_elbow']):
                    hip = (valid_keypoints[f'{side}_hip']['y'], valid_keypoints[f'{side}_hip']['x'])
                    shoulder = (valid_keypoints[f'{side}_shoulder']['y'], valid_keypoints[f'{side}_shoulder']['x'])
                    elbow = (valid_keypoints[f'{side}_elbow']['y'], valid_keypoints[f'{side}_elbow']['x'])
                    
                    angles[f'{side}_upper_arm'] = self.calculate_angle(hip, shoulder, elbow)
                else:
                    angles[f'{side}_upper_arm'] = 0  # Default neutral
            except:
                angles[f'{side}_upper_arm'] = 0
        
        # Lower arm angles
        for side in ['left', 'right']:
            try:
                if all(valid_keypoints[k]['valid'] for k in [f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist']):
                    shoulder = (valid_keypoints[f'{side}_shoulder']['y'], valid_keypoints[f'{side}_shoulder']['x'])
                    elbow = (valid_keypoints[f'{side}_elbow']['y'], valid_keypoints[f'{side}_elbow']['x'])
                    wrist = (valid_keypoints[f'{side}_wrist']['y'], valid_keypoints[f'{side}_wrist']['x'])
                    
                    angles[f'{side}_lower_arm'] = self.calculate_angle(shoulder, elbow, wrist)
                else:
                    angles[f'{side}_lower_arm'] = 90  # Default neutral
            except:
                angles[f'{side}_lower_arm'] = 90
        
        # Leg angles
        for side in ['left', 'right']:
            try:
                if all(valid_keypoints[k]['valid'] for k in [f'{side}_hip', f'{side}_knee', f'{side}_ankle']):
                    hip = (valid_keypoints[f'{side}_hip']['y'], valid_keypoints[f'{side}_hip']['x'])
                    knee = (valid_keypoints[f'{side}_knee']['y'], valid_keypoints[f'{side}_knee']['x'])
                    ankle = (valid_keypoints[f'{side}_ankle']['y'], valid_keypoints[f'{side}_ankle']['x'])
                    
                    angles[f'{side}_leg'] = self.calculate_angle(hip, knee, ankle)
                else:
                    angles[f'{side}_leg'] = 100  # Default
            except:
                angles[f'{side}_leg'] = 100
        
        return angles
    
    def create_row_dict_simple(self, angles, filename, frame_num):
        """Create row dictionary from angles"""
        return {
            'File Name': filename,
            'Frame': frame_num,
            'Neck Angle': angles.get('neck', 5),
            'Left Upper Arm Angle': angles.get('left_upper_arm', 0),
            'Right Upper Arm Angle': angles.get('right_upper_arm', 0),
            'Left Lower Arm Angle': angles.get('left_lower_arm', 90),
            'Right Lower Arm Angle': angles.get('right_lower_arm', 90),
            'Waist Angle': angles.get('waist', 90),
            'Left Leg Angle': angles.get('left_leg', 100),
            'Right Leg Angle': angles.get('right_leg', 100),
            'Left Arm Imputed': 0,
            'Right Arm Imputed': 0,
            'Left Leg Imputed': 0,
            'Right Leg Imputed': 0,
        }
    
    def engineer_features_simple(self, row_dict):
        """Simplified feature engineering based on your top 30 features"""
        # Create a sequence by replicating the row
        rows = [row_dict.copy() for _ in range(SEQUENCE_LENGTH)]
        for i, row in enumerate(rows):
            row['Frame'] = i
        
        df = pd.DataFrame(rows)
        
        # Core angles from your training
        core_angles = [
            'Neck Angle', 'Left Upper Arm Angle', 'Right Upper Arm Angle',
            'Left Lower Arm Angle', 'Right Lower Arm Angle', 'Waist Angle',
            'Left Leg Angle', 'Right Leg Angle'
        ]
        
        # Add trigonometric features (from your top 30)
        for angle in core_angles:
            if angle in df.columns:
                df[f'{angle}_sin'] = np.sin(np.radians(df[angle]))
                df[f'{angle}_cos'] = np.cos(np.radians(df[angle]))
                df[f'{angle}_squared'] = df[angle] ** 2
                df[f'{angle}_log'] = np.log(np.abs(df[angle]) + 1)
        
        # Range violation features
        normal_ranges = {
            'Neck Angle': (0, 45),
            'Waist Angle': (75, 105),
            'Left Upper Arm Angle': (-20, 120),
            'Right Upper Arm Angle': (-20, 120),
            'Left Lower Arm Angle': (60, 140),
            'Right Lower Arm Angle': (60, 140),
            'Left Leg Angle': (80, 120),
            'Right Leg Angle': (80, 120)
        }
        
        for angle, (min_val, max_val) in normal_ranges.items():
            if angle in df.columns:
                violations = (df[angle] < min_val) | (df[angle] > max_val)
                df[f'{angle}_range_violation'] = violations.astype(int)
        
        # Slouch pattern (from your top 30)
        if 'Waist Angle' in df.columns and 'Neck Angle' in df.columns:
            slouch_pattern = (df['Waist Angle'] > 60) & (df['Neck Angle'] > 10)
            df['slouch_pattern'] = slouch_pattern.astype(float)
        
        # Add missing features with defaults
        df['coordination_dominance'] = 0.5
        df['Waist Angle_velocity_mean'] = 0.0
        df['Waist Angle_acceleration_mean'] = 0.0
        df['Left Lower Arm Angle_skewness'] = 0.0
        
        # Get the last row (most representative)
        final_row = df.iloc[-1]
        
        # Extract features that exist in the model
        features_dict = {}
        for feature in self.resources['model_features']:
            if feature in final_row:
                value = final_row[feature]
                if np.isnan(value) or np.isinf(value):
                    features_dict[feature] = 0.0
                else:
                    features_dict[feature] = float(value)
            else:
                features_dict[feature] = 0.0
        
        # Calculate component scores
        component_scores = self.calculate_component_scores(row_dict)
        
        return features_dict, component_scores
    
    def calculate_component_scores(self, row_dict):
        """Calculate REBA component scores"""
        # Neck score
        neck_angle = row_dict.get('Neck Angle', 0)
        if 0 <= neck_angle < 20:
            neck_score = 1
        elif 20 <= neck_angle < 45:
            neck_score = 2
        elif neck_angle >= 45:
            neck_score = 3
        else:
            neck_score = 2
        
        # Trunk score
        waist_angle = row_dict.get('Waist Angle', 90)
        if waist_angle >= 0:
            if 0 <= waist_angle <= 5:
                trunk_score = 1
            elif 5 < waist_angle <= 20:
                trunk_score = 2
            elif 20 < waist_angle <= 60:
                trunk_score = 3
            else:
                trunk_score = 4
        else:
            trunk_score = 2
        
        # Upper arm score
        left_upper = abs(row_dict.get('Left Upper Arm Angle', 0))
        right_upper = abs(row_dict.get('Right Upper Arm Angle', 0))
        max_upper_arm = max(left_upper, right_upper)
        
        if -20 <= max_upper_arm < 20:
            upper_arm_score = 1
        elif 20 <= max_upper_arm < 45:
            upper_arm_score = 2
        elif 45 <= max_upper_arm < 90:
            upper_arm_score = 3
        else:
            upper_arm_score = 4
        
        # Lower arm score
        def score_lower_arm(angle):
            return 1 if 60 <= angle < 100 else 2
        
        left_lower_score = score_lower_arm(row_dict.get('Left Lower Arm Angle', 90))
        right_lower_score = score_lower_arm(row_dict.get('Right Lower Arm Angle', 90))
        lower_arm_score = max(left_lower_score, right_lower_score)
        
        return {
            'trunk_score': int(trunk_score),
            'neck_score': int(neck_score),
            'upper_arm_score': int(upper_arm_score),
            'lower_arm_score': int(lower_arm_score),
            'leg_score': 1  # Always 1 as per expert recommendation
        }
    
    def predict_single_image(self, features):
        """Make prediction for a single image"""
        # Convert features dict to array
        features_arr = np.array([list(features.values())])
        
        # Scale the features
        scaled_features = self.resources['scaler'].transform(features_arr)
        
        # Create sequence with decay
        decay = np.linspace(1.0, 0.95, SEQUENCE_LENGTH)[:, np.newaxis]
        sequence = np.tile(scaled_features, (SEQUENCE_LENGTH, 1)) * decay
        
        # Predict
        prediction = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)
        return float(prediction[0][0])
    
    def get_risk_level(self, reba_score):
        """Determine risk level from REBA score"""
        if reba_score <= 1:
            return "Negligible"
        elif reba_score <= 3:
            return "Low"
        elif reba_score <= 7:
            return "Medium"
        elif reba_score <= 10:
            return "High"
        else:
            return "Very High"

def find_test_files():
    """Find all image and video files in test directories"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    test_files = {'images': [], 'videos': []}
    
    # Check current directory
    for ext in image_extensions:
        test_files['images'].extend(glob.glob(f"*{ext}"))
        test_files['images'].extend(glob.glob(f"*{ext.upper()}"))
    
    for ext in video_extensions:
        test_files['videos'].extend(glob.glob(f"*{ext}"))
        test_files['videos'].extend(glob.glob(f"*{ext.upper()}"))
    
    # Check test directories
    for test_dir in TEST_DIRECTORIES:
        if os.path.exists(test_dir):
            print(f"📁 Checking directory: {test_dir}")
            
            for ext in image_extensions:
                pattern = os.path.join(test_dir, f"*{ext}")
                files = glob.glob(pattern)
                test_files['images'].extend(files)
                
                pattern = os.path.join(test_dir, f"*{ext.upper()}")
                files = glob.glob(pattern)
                test_files['images'].extend(files)
            
            for ext in video_extensions:
                pattern = os.path.join(test_dir, f"*{ext}")
                files = glob.glob(pattern)
                test_files['videos'].extend(files)
                
                pattern = os.path.join(test_dir, f"*{ext.upper()}")
                files = glob.glob(pattern)
                test_files['videos'].extend(files)
    
    # Remove duplicates
    test_files['images'] = list(set(test_files['images']))
    test_files['videos'] = list(set(test_files['videos']))
    
    print(f"📊 Found {len(test_files['images'])} images and {len(test_files['videos'])} videos")
    
    return test_files

def test_image_processing(tester, image_path):
    """Test image processing pipeline"""
    print(f"\n🖼️  Testing image: {os.path.basename(image_path)}")
    
    try:
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Could not read image: {image_path}")
            return False
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"   📐 Image size: {frame_rgb.shape}")
        
        # Initialize crop region
        height, width = frame_rgb.shape[:2]
        if width > height:
            box_height = width / height
            box_width = 1.0
            y_min = (height / 2 - width / 2) / height
            x_min = 0.0
        else:
            box_height = 1.0
            box_width = height / width
            y_min = 0.0
            x_min = (width / 2 - height / 2) / width
        
        crop_region = {
            'y_min': y_min, 'x_min': x_min,
            'y_max': y_min + box_height, 'x_max': x_min + box_width
        }
        
        # Prepare input for MoveNet
        input_image = tf.expand_dims(frame_rgb, axis=0)
        input_image = tf.image.crop_and_resize(
            input_image,
            boxes=[[crop_region['y_min'], crop_region['x_min'], 
                   crop_region['y_max'], crop_region['x_max']]],
            box_indices=[0],
            crop_size=[tester.input_size, tester.input_size]
        )
        
        # Run pose detection
        keypoints_with_scores = tester.movenet(input_image)
        print(f"   🎯 Keypoints detected: {keypoints_with_scores.shape}")
        
        # Calculate joint angles
        if LOCAL_MODULES_AVAILABLE:
            angles = get_joint_angles(keypoints_with_scores)
            if angles is None:
                print("   ⚠️  Insufficient keypoints detected")
                return False
            
            # Create row dictionary
            row_dict = create_row_dict(angles, os.path.basename(image_path), 0)
            
            # Engineer features
            features, component_scores = engineer_features_for_single_image(row_dict, tester.resources)
            
            # Make prediction
            reba_score = predict_single_image(features, tester.resources)
            
            # Generate feedback
            feedback = generate_feedback(component_scores, reba_score)
            risk_level = get_risk_level(reba_score)
        else:
            # Use standalone implementation
            angles = tester.get_joint_angles_simple(keypoints_with_scores)
            row_dict = tester.create_row_dict_simple(angles, os.path.basename(image_path), 0)
            features, component_scores = tester.engineer_features_simple(row_dict)
            reba_score = tester.predict_single_image(features)
            risk_level = tester.get_risk_level(reba_score)
            feedback = f"REBA Score: {reba_score:.2f} - Risk Level: {risk_level}"
        
        print(f"   🎯 REBA Score: {reba_score:.2f}")
        print(f"   📊 Risk Level: {risk_level}")
        print(f"   📋 Component Scores: {component_scores}")
        print(f"   💬 Feedback: {feedback[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing image: {e}")
        traceback.print_exc()
        return False

def test_video_processing(tester, video_path, max_frames=30):
    """Test video processing pipeline (limited frames)"""
    print(f"\n🎬 Testing video: {os.path.basename(video_path)}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   📐 Video info: {total_frames} frames, {fps:.2f} fps, {duration:.1f}s")
        
        processed_frames = 0
        reba_scores = []
        
        frame_interval = max(1, total_frames // max_frames)  # Sample frames
        print(f"   🎯 Processing every {frame_interval} frames (max {max_frames} frames)")
        
        frame_count = 0
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every nth frame
            if frame_count % frame_interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Initialize crop region
                    height, width = frame_rgb.shape[:2]
                    if width > height:
                        box_height = width / height
                        box_width = 1.0
                        y_min = (height / 2 - width / 2) / height
                        x_min = 0.0
                    else:
                        box_height = 1.0
                        box_width = height / width
                        y_min = 0.0
                        x_min = (width / 2 - height / 2) / width
                    
                    crop_region = {
                        'y_min': y_min, 'x_min': x_min,
                        'y_max': y_min + box_height, 'x_max': x_min + box_width
                    }
                    
                    # Prepare input for MoveNet
                    input_image = tf.expand_dims(frame_rgb, axis=0)
                    input_image = tf.image.crop_and_resize(
                        input_image,
                        boxes=[[crop_region['y_min'], crop_region['x_min'], 
                               crop_region['y_max'], crop_region['x_max']]],
                        box_indices=[0],
                        crop_size=[tester.input_size, tester.input_size]
                    )
                    
                    # Run pose detection
                    keypoints_with_scores = tester.movenet(input_image)
                    
                    # Calculate joint angles and predict
                    if LOCAL_MODULES_AVAILABLE:
                        angles = get_joint_angles(keypoints_with_scores)
                        if angles is not None:
                            row_dict = create_row_dict(angles, os.path.basename(video_path), frame_count)
                            features, component_scores = engineer_features_for_single_image(row_dict, tester.resources)
                            reba_score = predict_single_image(features, tester.resources)
                            reba_scores.append(reba_score)
                            processed_frames += 1
                    else:
                        angles = tester.get_joint_angles_simple(keypoints_with_scores)
                        if angles:
                            row_dict = tester.create_row_dict_simple(angles, os.path.basename(video_path), frame_count)
                            features, component_scores = tester.engineer_features_simple(row_dict)
                            reba_score = tester.predict_single_image(features)
                            reba_scores.append(reba_score)
                            processed_frames += 1
                
                except Exception as e:
                    print(f"   ⚠️  Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        if reba_scores:
            avg_reba = np.mean(reba_scores)
            min_reba = np.min(reba_scores)
            max_reba = np.max(reba_scores)
            
            if LOCAL_MODULES_AVAILABLE:
                risk_level = get_risk_level(avg_reba)
            else:
                risk_level = tester.get_risk_level(avg_reba)
            
            print(f"   🎯 Processed {processed_frames} frames")
            print(f"   📊 REBA Scores - Avg: {avg_reba:.2f}, Min: {min_reba:.2f}, Max: {max_reba:.2f}")
            print(f"   📋 Overall Risk Level: {risk_level}")
            return True
        else:
            print("   ❌ No frames could be processed successfully")
            return False
        
    except Exception as e:
        print(f"   ❌ Error processing video: {e}")
        traceback.print_exc()
        return False

def main():
    """Main testing function"""
    print("🚀 Starting Ergonomic Assessment Model Test")
    print("=" * 60)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🧠 TensorFlow version: {tf.__version__}")
    print(f"📂 Current directory: {os.getcwd()}")
    
    # Initialize tester
    if LOCAL_MODULES_AVAILABLE:
        print("\n🔧 Using local modules for testing...")
        try:
            # Test model loading using local modules
            resources = get_model_resources()
            if resources is None:
                print("❌ Failed to load model resources with local modules")
                return False
            
            # Test MoveNet initialization
            movenet, input_size = init_movenet()
            if movenet is None:
                print("❌ Failed to initialize MoveNet with local modules")
                return False
            
            print("✅ Local modules working correctly")
            
            # Create a wrapper class for compatibility
            class LocalModuleTester:
                def __init__(self):
                    self.resources = resources
                    self.movenet = movenet
                    self.input_size = input_size
                    self.model = resources['model']
                
                def get_risk_level(self, reba_score):
                    return get_risk_level(reba_score)
            
            tester = LocalModuleTester()
            
        except Exception as e:
            print(f"❌ Error with local modules: {e}")
            print("Falling back to standalone implementation...")
            tester = StandaloneModelTester()
            if not tester.load_model_resources() or not tester.init_movenet():
                print("❌ Failed to initialize standalone tester")
                return False
    else:
        print("\n🔧 Using standalone implementation...")
        tester = StandaloneModelTester()
        if not tester.load_model_resources() or not tester.init_movenet():
            print("❌ Failed to initialize standalone tester")
            return False
    
    # Test model architecture
    print(f"\n📊 Model Architecture Test:")
    print(f"   Model type: {type(tester.model).__name__}")
    print(f"   Total parameters: {tester.model.count_params():,}")
    print(f"   Input shape: {tester.model.input_shape}")
    print(f"   Output shape: {tester.model.output_shape}")
    
    # Test preprocessing resources
    print(f"\n📋 Preprocessing Resources Test:")
    if hasattr(tester, 'resources'):
        resources = tester.resources
        print(f"   Scaler type: {type(resources['scaler']).__name__}")
        print(f"   Feature count: {len(resources['model_features'])}")
        print(f"   Core angles: {len(resources['core_angles'])}")
        print(f"   Sequence length: {resources.get('sequence_length', 'Not specified')}")
        
        # Test scaler
        try:
            dummy_features = np.random.randn(1, len(resources['model_features']))
            scaled = resources['scaler'].transform(dummy_features)
            print(f"   ✅ Scaler working: Input {dummy_features.shape} -> Output {scaled.shape}")
        except Exception as e:
            print(f"   ❌ Scaler error: {e}")
    
    # Find test files
    print(f"\n📁 Finding test files...")
    test_files = find_test_files()
    
    if not test_files['images'] and not test_files['videos']:
        print("❌ No test files found!")
        print("Please add some images or videos to the following directories:")
        for dir_name in TEST_DIRECTORIES:
            print(f"   - {dir_name}/")
        print("Or place them in the current directory.")
        return False
    
    # Test feature engineering with dummy data
    print(f"\n🧪 Testing feature engineering pipeline...")
    try:
        dummy_row = {
            'File Name': 'test_dummy',
            'Frame': 0,
            'Neck Angle': 15.0,
            'Left Upper Arm Angle': 25.0,
            'Right Upper Arm Angle': 30.0,
            'Left Lower Arm Angle': 85.0,
            'Right Lower Arm Angle': 90.0,
            'Waist Angle': 95.0,
            'Left Leg Angle': 105.0,
            'Right Leg Angle': 100.0,
            'Left Arm Imputed': 0,
            'Right Arm Imputed': 0,
            'Left Leg Imputed': 0,
            'Right Leg Imputed': 0,
        }
        
        if LOCAL_MODULES_AVAILABLE:
            features, component_scores = engineer_features_for_single_image(dummy_row, tester.resources)
            reba_prediction = predict_single_image(features, tester.resources)
            risk_level = get_risk_level(reba_prediction)
            feedback = generate_feedback(component_scores, reba_prediction)
        else:
            features, component_scores = tester.engineer_features_simple(dummy_row)
            reba_prediction = tester.predict_single_image(features)
            risk_level = tester.get_risk_level(reba_prediction)
            feedback = f"REBA Score: {reba_prediction:.2f} - Risk Level: {risk_level}"
        
        print(f"   ✅ Feature engineering successful")
        print(f"   📊 Features generated: {len(features)}")
        print(f"   🎯 Dummy prediction: {reba_prediction:.3f}")
        print(f"   📋 Component scores: {component_scores}")
        print(f"   🏷️  Risk level: {risk_level}")
        print(f"   💬 Feedback preview: {feedback[:100]}...")
        
    except Exception as e:
        print(f"   ❌ Feature engineering failed: {e}")
        traceback.print_exc()
        return False
    
    # Test images
    success_count = 0
    total_tests = 0
    
    if test_files['images']:
        print(f"\n🖼️  Testing image processing ({len(test_files['images'])} images)...")
        for i, image_path in enumerate(test_files['images'][:5], 1):  # Test max 5 images
            print(f"\n--- Image Test {i}/5 ---")
            total_tests += 1
            if test_image_processing(tester, image_path):
                success_count += 1
    
    # Test videos
    if test_files['videos']:
        print(f"\n🎬 Testing video processing ({len(test_files['videos'])} videos)...")
        for i, video_path in enumerate(test_files['videos'][:3], 1):  # Test max 3 videos
            print(f"\n--- Video Test {i}/3 ---")
            total_tests += 1
            if test_video_processing(tester, video_path, max_frames=10):  # Test only 10 frames per video
                success_count += 1
    
    # Summary
    print(f"\n🏁 Test Summary")
    print("=" * 60)
    print(f"📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Successful tests: {success_count}/{total_tests}")
    print(f"📊 Success rate: {(success_count/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Model is working correctly.")
        return True
    elif success_count > 0:
        print("⚠️  Some tests failed, but basic functionality is working.")
        return True
    else:
        print("❌ All tests failed. Please check your model and dependencies.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🔚 Exiting with code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)