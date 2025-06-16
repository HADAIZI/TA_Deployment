#!/usr/bin/env python3
"""
Comprehensive test script for ergonomic assessment model
Tests model loading, feature engineering, and prediction pipeline
WITH VISUALIZATION SUPPORT - USING ALL APP MODULES
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

# Add visualization imports
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add current directory to path to import app modules
sys.path.append(os.getcwd())

# Import ALL your app modules
try:
    # Import from app.services.pose_estimation
    from app.services.pose_estimation import (
        init_movenet, get_joint_angles, create_row_dict, init_crop_region,
        run_inference, should_flip_image, crop_and_resize, KEYPOINT_DICT,
        calculate_angle, process_pose_from_bytes, AngleSmoother,
        get_keypoint_if_valid, calculate_angle_with_fallback
    )
    
    # Import from app.services.ergonomic_model
    from app.services.ergonomic_model import (
        get_model_resources, engineer_features_for_single_image, predict_single_image,
        get_risk_level, generate_feedback, get_action_level, predict_video,
        prepare_sequences, calculate_component_scores_from_angles
    )
    
    # Import from app.services.image_visualizer
    from app.services.image_visualizer import generate_pose_visualization
    
    LOCAL_MODULES_AVAILABLE = True
    print("✅ ALL app modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Failed to import app modules: {e}")
    print("Please ensure you're running from the root directory with app/ folder")
    LOCAL_MODULES_AVAILABLE = False

# Constants
KEYPOINT_THRESHOLD = 0.3
SEQUENCE_LENGTH = 60
TEST_DIRECTORIES = ['test']
OUTPUT_VIZ_DIR = "test_visualizations"

def save_visualization_with_app_modules(processed_image, keypoints_with_scores, component_scores, 
                                       reba_score, image_filename, angle_values=None, 
                                       original_image=None, flip_applied=False):
    """Save visualization using your app modules"""
    if not LOCAL_MODULES_AVAILABLE:
        print("❌ App modules not available for visualization")
        return None
        
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    
    # Add REBA score to component scores for your visualizer
    component_scores_with_reba = component_scores.copy()
    component_scores_with_reba['reba_score'] = reba_score
    
    # Use your app's generate_pose_visualization function
    try:
        vis_image = generate_pose_visualization(
            processed_image, keypoints_with_scores, component_scores_with_reba, 
            original_image, flip_applied
        )
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(image_filename)[0]
        viz_filename = f"{base_name}_reba_{reba_score:.1f}_{timestamp}.png"
        viz_path = os.path.join(OUTPUT_VIZ_DIR, viz_filename)
        
        # Convert RGB to BGR for cv2 saving
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(viz_path, vis_image_bgr)
        
        return viz_path
        
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")
        traceback.print_exc()
        return None

def find_test_files():
    """Find all image and video files in test directories"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    test_files = {'images': [], 'videos': []}
    
    # Check current directory
    for ext in image_extensions:
        test_files['images'].extend(glob.glob(f"*{ext}"))
        test_files['images'].extend(glob.glob(f"*{ext.upper()}"))
    
    # Check test directories
    for test_dir in TEST_DIRECTORIES:
        if os.path.exists(test_dir):
            print(f"📁 Checking directory: {test_dir}")
            
            for ext in image_extensions + video_extensions:
                pattern = os.path.join(test_dir, f"*{ext}")
                files = glob.glob(pattern)
                if ext in image_extensions:
                    test_files['images'].extend(files)
                else:
                    test_files['videos'].extend(files)
                
                pattern = os.path.join(test_dir, f"*{ext.upper()}")
                files = glob.glob(pattern)
                if ext in image_extensions:
                    test_files['images'].extend(files)
                else:
                    test_files['videos'].extend(files)
    
    # Remove duplicates
    test_files['images'] = list(set(test_files['images']))
    test_files['videos'] = list(set(test_files['videos']))
    
    print(f"📊 Found {len(test_files['images'])} images and {len(test_files['videos'])} videos")
    
    return test_files

def test_image_processing_with_app_modules(image_path):
    """Test image processing using ALL your app modules"""
    print(f"\n🖼️  Testing image: {os.path.basename(image_path)}")
    
    if not LOCAL_MODULES_AVAILABLE:
        print("❌ App modules not available")
        return False
    
    try:
        # Method 1: Use your process_pose_from_bytes function (RECOMMENDED)
        print("   🚀 Using your process_pose_from_bytes function")
        
        # Read image as bytes (like your deployment)
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Process using your complete pipeline
        result = process_pose_from_bytes(image_bytes, output_visualization=False)
        
        print(f"   📐 Image processed successfully")
        print(f"   🎯 REBA Score: {result['reba_score']:.2f}")
        print(f"   📊 Risk Level: {result['risk_level']}")
        print(f"   📋 Component Scores: {result['component_scores']}")
        print(f"   💬 Feedback: {result['feedback'][:100]}...")
        
        # For visualization, we need to get the processed images
        # Let's also run the individual steps to get the images
        print("   🎨 Generating visualization with individual steps...")
        
        # Read image for visualization
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"   ❌ Could not read image for visualization")
            return True  # Still success since processing worked
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize and run inference
        movenet, input_size = init_movenet()
        crop_region = init_crop_region(frame_rgb.shape[0], frame_rgb.shape[1])
        
        # Run your superior inference
        keypoints_with_scores, processed_image, original_image, flip_required = run_inference(
            movenet, frame_rgb, crop_region, crop_size=[input_size, input_size]
        )
        
        print(f"   🔄 Flip applied: {'Yes' if flip_required else 'No'}")
        
        # Generate visualization using your app modules
        viz_path = save_visualization_with_app_modules(
            processed_image, keypoints_with_scores, result['component_scores'],
            result['reba_score'], os.path.basename(image_path), 
            result['angle_values'], original_image, flip_required
        )
        
        if viz_path:
            print(f"   🎨 Visualization saved: {viz_path}")
        else:
            print(f"   ⚠️  Visualization could not be saved")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing image: {e}")
        traceback.print_exc()
        return False

def test_video_processing_with_app_modules(video_path, max_frames=30):
    """Test video processing using your app modules"""
    print(f"\n🎬 Testing video: {os.path.basename(video_path)}")
    
    if not LOCAL_MODULES_AVAILABLE:
        print("❌ App modules not available")
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   📐 Video info: {total_frames} frames, {fps:.2f} fps, {duration:.1f}s")
        
        # Initialize your modules
        movenet, input_size = init_movenet()
        resources = get_model_resources()
        
        processed_frames = 0
        reba_scores = []
        best_frame_data = None
        best_reba_score = 0
        
        frame_interval = max(1, total_frames // max_frames)
        print(f"   🎯 Processing every {frame_interval} frames (max {max_frames} frames)")
        print("   🚀 Using your complete app modules")
        
        frame_count = 0
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every nth frame
            if frame_count % frame_interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Use your complete inference pipeline
                    crop_region = init_crop_region(frame_rgb.shape[0], frame_rgb.shape[1])
                    keypoints_with_scores, processed_image, original_image, flip_required = run_inference(
                        movenet, frame_rgb, crop_region, crop_size=[input_size, input_size]
                    )
                    
                    # Calculate joint angles using your function
                    angles = get_joint_angles(keypoints_with_scores)
                    if angles is not None:
                        # Create row dictionary using your function
                        row_dict = create_row_dict(angles, os.path.basename(video_path), frame_count)
                        
                        # Engineer features using your function
                        features, component_scores = engineer_features_for_single_image(row_dict, resources)
                        
                        # Make prediction using your function
                        reba_score = predict_single_image(features, resources)
                        
                        reba_scores.append(reba_score)
                        processed_frames += 1
                        
                        # Keep track of highest risk frame for visualization
                        if reba_score > best_reba_score:
                            best_reba_score = reba_score
                            best_frame_data = {
                                'processed_image': processed_image.copy(),
                                'original_image': original_image.copy(),
                                'keypoints': keypoints_with_scores.copy(),
                                'component_scores': component_scores.copy(),
                                'reba_score': reba_score,
                                'angles': angles.copy(),
                                'frame_number': frame_count,
                                'flip_required': flip_required
                            }
                
                except Exception as e:
                    print(f"   ⚠️  Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        if reba_scores:
            avg_reba = np.mean(reba_scores)
            min_reba = np.min(reba_scores)
            max_reba = np.max(reba_scores)
            risk_level = get_risk_level(avg_reba)
            
            print(f"   🎯 Processed {processed_frames} frames")
            print(f"   📊 REBA Scores - Avg: {avg_reba:.2f}, Min: {min_reba:.2f}, Max: {max_reba:.2f}")
            print(f"   📋 Overall Risk Level: {risk_level}")
            
            # Save visualization of the highest risk frame using your app modules
            if best_frame_data:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_filename = f"{video_name}_frame_{best_frame_data['frame_number']}_highest_risk.jpg"
                
                # Convert angles dict to the format expected by save_visualization_with_app_modules
                angle_values = {}
                for key, value in best_frame_data['angles'].items():
                    if key != 'imputed_angles' and isinstance(value, (int, float)):
                        angle_values[key] = float(value)
                
                viz_path = save_visualization_with_app_modules(
                    best_frame_data['processed_image'], 
                    best_frame_data['keypoints'], 
                    best_frame_data['component_scores'],
                    best_frame_data['reba_score'], 
                    frame_filename,
                    angle_values,
                    best_frame_data['original_image'],
                    best_frame_data['flip_required']
                )
                
                if viz_path:
                    print(f"   🎨 Highest risk frame visualization saved: {viz_path}")
            
            return True
        else:
            print("   ❌ No frames could be processed successfully")
            return False
        
    except Exception as e:
        print(f"   ❌ Error processing video: {e}")
        traceback.print_exc()
        return False

def test_direct_bytes_processing():
    """Test your process_pose_from_bytes function directly"""
    print(f"\n🧪 Testing direct bytes processing...")
    
    if not LOCAL_MODULES_AVAILABLE:
        print("❌ App modules not available")
        return False
    
    try:
        # Create a simple test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', test_image)
        image_bytes = buffer.tobytes()
        
        print(f"   📦 Created test image bytes: {len(image_bytes)} bytes")
        
        # Process using your function
        result = process_pose_from_bytes(image_bytes, output_visualization=False)
        
        print(f"   ✅ Direct bytes processing successful!")
        print(f"   🎯 Test REBA Score: {result['reba_score']:.2f}")
        print(f"   📊 Test Risk Level: {result['risk_level']}")
        print(f"   📋 Test Component Scores: {result['component_scores']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Direct bytes processing failed: {e}")
        # This is expected to fail with random image, so don't print full traceback
        return False

def main():
    """Main testing function using ALL your app modules"""
    print("🚀 Starting Ergonomic Assessment Model Test with ALL APP MODULES")
    print("=" * 80)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🧠 TensorFlow version: {tf.__version__}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"🎨 Output directory: {OUTPUT_VIZ_DIR}")
    
    # Check if app modules are available
    if not LOCAL_MODULES_AVAILABLE:
        print("\n❌ App modules not available!")
        print("Please ensure:")
        print("1. You're running from the root directory")
        print("2. The app/ folder exists with all modules")
        print("3. All dependencies are installed")
        return False
    
    # Create output directory
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    print(f"✅ Visualization output directory created: {OUTPUT_VIZ_DIR}")
    
    # Test your app modules initialization
    print(f"\n🔧 Testing your app modules initialization...")
    try:
        # Test MoveNet initialization
        movenet, input_size = init_movenet()
        if movenet is None:
            print("❌ Failed to initialize MoveNet")
            return False
        print(f"✅ MoveNet initialized: input_size={input_size}")
        
        # Test model resources
        resources = get_model_resources()
        if resources is None:
            print("❌ Failed to load model resources")
            return False
        print(f"✅ Model resources loaded: {len(resources['model_features'])} features")
        
        # Test individual functions
        print(f"✅ Available functions:")
        print(f"   - init_crop_region: {callable(init_crop_region)}")
        print(f"   - run_inference: {callable(run_inference)}")
        print(f"   - get_joint_angles: {callable(get_joint_angles)}")
        print(f"   - create_row_dict: {callable(create_row_dict)}")
        print(f"   - engineer_features_for_single_image: {callable(engineer_features_for_single_image)}")
        print(f"   - predict_single_image: {callable(predict_single_image)}")
        print(f"   - generate_pose_visualization: {callable(generate_pose_visualization)}")
        print(f"   - process_pose_from_bytes: {callable(process_pose_from_bytes)}")
        
    except Exception as e:
        print(f"❌ Error testing app modules: {e}")
        traceback.print_exc()
        return False
    
    # Test direct bytes processing
    test_direct_bytes_processing()
    
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
    
    # Test images using ALL your app modules
    success_count = 0
    total_tests = 0
    
    if test_files['images']:
        print(f"\n🖼️  Testing image processing with ALL APP MODULES ({len(test_files['images'])} images)...")
        for i, image_path in enumerate(test_files['images'][:5], 1):  # Test max 5 images
            print(f"\n--- Image Test {i}/5 ---")
            total_tests += 1
            if test_image_processing_with_app_modules(image_path):
                success_count += 1
    
    # Test videos using ALL your app modules
    if test_files['videos']:
        print(f"\n🎬 Testing video processing with ALL APP MODULES ({len(test_files['videos'])} videos)...")
        for i, video_path in enumerate(test_files['videos'][:3], 1):  # Test max 3 videos
            print(f"\n--- Video Test {i}/3 ---")
            total_tests += 1
            if test_video_processing_with_app_modules(video_path, max_frames=10):
                success_count += 1
    
    # Summary
    print(f"\n🏁 Test Summary")
    print("=" * 80)
    print(f"📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Successful tests: {success_count}/{total_tests}")
    print(f"📊 Success rate: {(success_count/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    print(f"🎨 Visualizations saved in: {OUTPUT_VIZ_DIR}/")
    print(f"🚀 Used ALL YOUR APP MODULES throughout!")
    
    # List generated visualizations
    viz_files = glob.glob(os.path.join(OUTPUT_VIZ_DIR, "*.png"))
    if viz_files:
        print(f"\n📸 Generated visualizations ({len(viz_files)}):")
        for viz_file in sorted(viz_files):
            print(f"   - {os.path.basename(viz_file)}")
    
    # Final recommendations
    print(f"\n💡 Recommendations:")
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! Your app modules are working perfectly!")
        print("✅ Your deployment is ready with proper visualizations")
    elif success_count > 0:
        print("⚠️  Some tests failed, but your app modules are mostly working")
        print("🔍 Check the error messages above for specific issues")
    else:
        print("❌ All tests failed. Please check:")
        print("   1. App module imports and dependencies")
        print("   2. Model files in modelv4/ directory")
        print("   3. Test files in test/ directory")
    
    print(f"\n🔧 Your app modules used:")
    print(f"   ✅ app.services.pose_estimation")
    print(f"   ✅ app.services.ergonomic_model") 
    print(f"   ✅ app.services.image_visualizer")
    
    return success_count > 0

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