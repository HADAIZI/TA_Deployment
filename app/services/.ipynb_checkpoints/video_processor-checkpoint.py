import cv2
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import threading
import time
import tensorflow as tf

from app.services.pose_estimation import init_movenet, init_crop_region, get_joint_angles, create_row_dict
from app.services.ergonomic_model import get_model_resources, engineer_features_for_single_image, predict_single_image
from app.services.ergonomic_model import get_risk_level, generate_feedback
from app.services.job_manager import update_job
from app.utils.summarize_results import summarize_results

# Constants
FRAME_INTERVAL = 3  # Process every Nth frame

def process_video(job_folder, job_id, video_path):
    """
    Process a video file for ergonomic analysis
    
    Args:
        job_folder: Folder containing job files
        job_id: Unique job identifier
        video_path: Path to video file
    """
    try:
        print(f"Starting video processing job {job_id}")
        
        # Initialize MoveNet and resources
        movenet, input_size = init_movenet()
        if movenet is None:
            raise ValueError("Could not initialize MoveNet model")
            
        resources = get_model_resources()
        if resources is None:
            raise ValueError("Could not load model resources")
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {width}x{height}, {fps:.2f} fps, {duration:.2f} seconds, {total_frames} frames")
        print(f"Processing every {FRAME_INTERVAL} frames")
        
        # Initialize tracking variables
        frame_count = 0
        processed_count = 0
        rows = []
        
        # Create frames folder for debug images (if needed)
        frames_folder = os.path.join(job_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)
        
        # Create progress tracking file
        progress_file = os.path.join(job_folder, "progress.txt")
        with open(progress_file, 'w') as f:
            f.write("0.0")
            
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process only every Nth frame
            if frame_count % FRAME_INTERVAL == 0:
                # Update progress
                progress = min(100.0, 100.0 * frame_count / total_frames)
                with open(progress_file, 'w') as f:
                    f.write(f"{progress:.1f}")
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Initialize crop region
                crop_region = init_crop_region(frame_rgb.shape[0], frame_rgb.shape[1])
                
                # Prepare input for MoveNet
                input_image = tf.expand_dims(frame_rgb, axis=0)
                input_image = tf.image.crop_and_resize(
                    input_image,
                    boxes=[[
                        crop_region['y_min'],
                        crop_region['x_min'],
                        crop_region['y_max'],
                        crop_region['x_max']
                    ]],
                    box_indices=[0],
                    crop_size=[input_size, input_size]
                )
                
                # Run pose detection
                keypoints_with_scores = movenet(input_image)
                
                # Calculate joint angles
                angles = get_joint_angles(keypoints_with_scores)
                
                if angles is not None:
                    # Create row dictionary
                    row = create_row_dict(angles, os.path.basename(video_path), frame_count)
                    rows.append(row)
                    processed_count += 1
                    
                    # Save debug frame (optional)
                    if processed_count <= 5:  # Save only first few frames for debugging
                        # Create frame visualization
                        features, component_scores = engineer_features_for_single_image(row, resources)
                        reba_score = predict_single_image(features, resources)
                        component_scores['reba_score'] = reba_score
                        
                        keypoints_adjusted = keypoints_with_scores.copy()
                        # Adjust keypoints for crop region
                        for idx in range(17):
                            keypoints_adjusted[0, 0, idx, 0] = (
                                crop_region['y_min'] * frame_rgb.shape[0] +
                                crop_region['height'] * frame_rgb.shape[0] *
                                keypoints_adjusted[0, 0, idx, 0]) / frame_rgb.shape[0]
                            keypoints_adjusted[0, 0, idx, 1] = (
                                crop_region['x_min'] * frame_rgb.shape[1] +
                                crop_region['width'] * frame_rgb.shape[1] *
                                keypoints_adjusted[0, 0, idx, 1]) / frame_rgb.shape[1]
                        
                        # Generate visualization (import locally to avoid circular import)
                        from app.services.image_visualizer import generate_pose_visualization
                        visualization = generate_pose_visualization(
                            frame_rgb, keypoints_adjusted, component_scores
                        )
                        
                        # Save visualization
                        debug_path = os.path.join(frames_folder, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(debug_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                    
            frame_count += 1
            
            # Check if at least 3 frames have been processed after 50% of the video
            if frame_count > total_frames * 0.5 and processed_count >= 3:
                pass  # Continue processing
        
        cap.release()
        
        print(f"Video processing complete: {processed_count} frames analyzed out of {frame_count} total")
        
        # Check if we have enough data
        if len(rows) < 3:
            raise ValueError(f"Insufficient valid frames detected ({processed_count}). At least 3 frames are required for analysis.")
        
        # Create DataFrame from rows
        df = pd.DataFrame(rows)
        
        # Prepare results
        video_name = os.path.basename(video_path)
        
        # Calculate results for each frame
        results = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Engineer features for this frame
            features, component_scores = engineer_features_for_single_image(row_dict, resources)
            # Make prediction
            reba_score = predict_single_image(features, resources)
            # Create result
            result = {
                'frame': int(row['Frame']),
                'reba_score': float(reba_score),
                'component_scores': {
                    'trunk': int(component_scores['trunk_score']),
                    'neck': int(component_scores['neck_score']),
                    'upper_arm': int(component_scores['upper_arm_score']),
                    'lower_arm': int(component_scores['lower_arm_score']),
                    'leg': int(component_scores['leg_score'])
                },
                'angle_values': {
                    'neck': float(row['Neck Angle']),
                    'waist': float(row['Waist Angle']),
                    'left_upper_arm': float(row['Left Upper Arm Angle']),
                    'right_upper_arm': float(row['Right Upper Arm Angle']),
                    'left_lower_arm': float(row['Left Lower Arm Angle']),
                    'right_lower_arm': float(row['Right Lower Arm Angle']),
                    'left_leg': float(row['Left Leg Angle']),
                    'right_leg': float(row['Right Leg Angle'])
                }
            }
            results.append(result)
            
        # Summarize results
        final_summary = summarize_results(results)
        
        # Add timestamps and duration information
        final_summary['video_metadata'] = {
            'filename': video_name,
            'duration_seconds': duration,
            'total_frames': total_frames,
            'fps': float(fps),
            'processed_frames': processed_count,
            'processed_ratio': float(processed_count) / total_frames if total_frames > 0 else 0
        }
        
        # Generate feedback based on average component scores
        avg_component_scores = {
            'trunk_score': final_summary['avg_component_scores']['trunk'],
            'neck_score': final_summary['avg_component_scores']['neck'],
            'upper_arm_score': final_summary['avg_component_scores']['upper_arm'],
            'lower_arm_score': final_summary['avg_component_scores']['lower_arm'],
            'leg_score': final_summary['avg_component_scores']['leg']
        }
        
        final_summary['feedback'] = generate_feedback(avg_component_scores, final_summary['avg_reba_score'])
        
        # Update job with results
        update_job(job_id, final_summary)
        
        print(f"Video analysis completed for {video_name}")
        return final_summary
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Update job with error
        error_result = {
            "error": str(e),
            "status": "failed"
        }
        update_job(job_id, error_result)
        
        return error_result
    finally:
        # Set progress to 100% when finished
        try:
            progress_file = os.path.join(job_folder, "progress.txt")
            with open(progress_file, 'w') as f:
                f.write("100.0")
        except:
            pass
