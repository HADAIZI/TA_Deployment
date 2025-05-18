import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from app.services.ergonomic_model import get_model_resources, predict_single_image, get_risk_level, generate_feedback
from app.services.image_visualizer import generate_pose_visualization

# MoveNet model initialization
_movenet = None
_input_size = 256

# Constants
KEYPOINT_THRESHOLD = 0.3
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3,
    'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 
    'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9,
    'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def init_movenet():
    """Initialize the MoveNet model"""
    global _movenet, _input_size
    
    if _movenet is not None:
        return _movenet, _input_size
    
    try:
        import tensorflow_hub as hub
        model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        module = hub.load(model_url)
        
        def movenet_wrapper(input_image):
            """Runs MoveNet on an input image."""
            model = module.signatures['serving_default']
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = model(input_image)
            keypoints_with_scores = outputs['output_0'].numpy()
            return keypoints_with_scores
        
        _movenet = movenet_wrapper
        _input_size = 256
        
        print("MoveNet model initialized successfully")
        return _movenet, _input_size
        
    except Exception as e:
        print(f"Error initializing MoveNet: {e}")
        return None, None

class AngleSmoother:
    """Helper class to smooth angle measurements"""
    def __init__(self, window_size=3):
        self.history = deque(maxlen=window_size)
        
    def smooth(self, angle):
        if angle is not None:
            self.history.append(angle)
            if len(self.history) > 0:
                return np.mean(self.history)
        return None

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def init_crop_region(image_height, image_width):
    """Defines the default crop region."""
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }

def crop_and_resize(image, crop_region, crop_size):
    """Crops and resizes the image to prepare for the model input."""
    boxes = [[crop_region['y_min'], crop_region['x_min'],
             crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image

def should_flip_image(keypoints_with_scores):
    """Determines if the image should be flipped based on keypoint positions."""
    # Get relevant keypoints with confidence checks
    left_shoulder = keypoints_with_scores[0, 0, KEYPOINT_DICT['left_shoulder']]
    right_shoulder = keypoints_with_scores[0, 0, KEYPOINT_DICT['right_shoulder']]
    left_wrist = keypoints_with_scores[0, 0, KEYPOINT_DICT['left_wrist']]
    left_knee = keypoints_with_scores[0, 0, KEYPOINT_DICT['left_knee']]

    score = 0
    valid_keypoints = 0
    
    # Shoulder comparison
    if left_shoulder[2] > KEYPOINT_THRESHOLD and right_shoulder[2] > KEYPOINT_THRESHOLD:
        if left_shoulder[1] > right_shoulder[1]:
            score += 1  # Facing left
        else:
            score -= 1  # Facing right
        valid_keypoints += 1
    
    # Wrist position
    if left_wrist[2] > KEYPOINT_THRESHOLD and left_shoulder[2] > KEYPOINT_THRESHOLD:
        if left_wrist[1] < left_shoulder[1]:
            score += 1  # Facing left
        else:
            score -= 1  # Facing right
        valid_keypoints += 1
    
    # Knee position
    if left_knee[2] > KEYPOINT_THRESHOLD and left_shoulder[2] > KEYPOINT_THRESHOLD:
        if left_knee[1] < left_shoulder[1]:
            score += 1  # Facing left
        else:
            score -= 1  # Facing right
        valid_keypoints += 1
    
    return score > 0 if valid_keypoints >= 2 else False

def run_inference(movenet, image, crop_region, crop_size):
    """Runs model inference on the cropped region with proper flip handling."""
    image_height, image_width, _ = image.shape
    
    # First pass to determine orientation
    input_image = crop_and_resize(tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
    keypoints_with_scores = movenet(input_image)
    flip_required = should_flip_image(keypoints_with_scores)
    
    # Second pass if flipping is needed
    if flip_required:
        flipped_image = cv2.flip(image, 1)
        input_image = crop_and_resize(tf.expand_dims(flipped_image, axis=0), crop_region, crop_size=crop_size)
        keypoints_with_scores = movenet(input_image)
        original_image = image.copy()
        image = flipped_image
    else:
        original_image = image.copy()
    
    # Adjust keypoints for crop region
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0, 0, idx, 1]) / image_width

    return keypoints_with_scores, image, original_image, flip_required

def get_keypoint_if_valid(validated_keypoints, keypoint_name):
    """Get a valid keypoint if it exists"""
    kp = validated_keypoints[keypoint_name]
    return (kp['y'], kp['x']) if kp['valid'] else None

def calculate_angle_with_fallback(a_name, b_name, c_name, angle_name, validated_keypoints, imputed_angles, neutral_angles):
    """Calculate angle with fallback to neutral angles"""
    a = get_keypoint_if_valid(validated_keypoints, a_name)
    b = get_keypoint_if_valid(validated_keypoints, b_name)
    c = get_keypoint_if_valid(validated_keypoints, c_name)
    
    if a is not None and b is not None and c is not None:
        try:
            angle = calculate_angle(a, b, c)
            return angle
        except:
            pass
    
    # If we get here, use neutral angle and flag as imputed
    imputed_angles[angle_name] = True
    return neutral_angles[angle_name]

def get_joint_angles(keypoints_with_scores, keypoint_threshold=KEYPOINT_THRESHOLD):
    """
    Calculate joint angles from pose keypoints
    
    Args:
        keypoints_with_scores: Output from MoveNet model
        keypoint_threshold: Confidence threshold for valid keypoints
    
    Returns:
        dict: Contains angles and imputation flags
    """
    keypoints = keypoints_with_scores[0, 0, :, :2]
    scores = keypoints_with_scores[0, 0, :, 2]

    # Initialize smoothers if they don't exist
    if not hasattr(get_joint_angles, 'smoothers'):
        get_joint_angles.smoothers = {
            'left_leg': AngleSmoother(),
            'right_leg': AngleSmoother(),
            'neck': AngleSmoother(),
            'trunk': AngleSmoother(),
            'upper_arm': AngleSmoother(),
            'lower_arm': AngleSmoother(),
        }

    # Initialize tracking dictionaries
    imputed_angles = {
        'left_leg': False,
        'right_leg': False,
        'neck': False,
        'waist': False,
        'left_upper_arm': False,
        'right_upper_arm': False,
        'left_lower_arm': False,
        'right_lower_arm': False
    }

    neutral_angles = {
        'left_leg': 100,
        'right_leg': 100,
        'left_upper_arm': 0,
        'right_upper_arm': 0,
        'left_lower_arm': 90,
        'right_lower_arm': 90,
        'waist': 110,
        'neck': 5
    }

    # Create validated keypoints dictionary
    validated_keypoints = {}
    for name, idx in KEYPOINT_DICT.items():
        validated_keypoints[name] = {
            'x': keypoints[idx][1] if scores[idx] > keypoint_threshold else None,
            'y': keypoints[idx][0] if scores[idx] > keypoint_threshold else None,
            'valid': scores[idx] > keypoint_threshold
        }

    # Calculate all angles with fallback
    angles = {}
    
    # Calculate waist angle (with forward/backward detection)
    shoulder_left = get_keypoint_if_valid(validated_keypoints, 'left_shoulder')
    shoulder_right = get_keypoint_if_valid(validated_keypoints, 'right_shoulder')
    hip_left = get_keypoint_if_valid(validated_keypoints, 'left_hip')
    hip_right = get_keypoint_if_valid(validated_keypoints, 'right_hip')
    
    if all([shoulder_left, shoulder_right, hip_left, hip_right]):
        # Original angle calculation
        shoulder_vec = np.array([shoulder_left[0] - shoulder_right[0],
                                 shoulder_left[1] - shoulder_right[1]])
        hip_vec = np.array([hip_left[0] - hip_right[0],
                            hip_left[1] - hip_right[1]])
        
        dot_product = np.dot(shoulder_vec, hip_vec)
        shoulder_mag = np.linalg.norm(shoulder_vec)
        hip_mag = np.linalg.norm(hip_vec)
        
        if shoulder_mag > 0 and hip_mag > 0:
            cos_angle = dot_product / (shoulder_mag * hip_mag)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            unsigned_angle = np.degrees(np.arccos(cos_angle))
            
            # Trunk flexion (forward/backward lean)
            shoulder_center_y = (shoulder_left[1] + shoulder_right[1]) / 2
            hip_center_y = (hip_left[1] + hip_right[1]) / 2
            
            # Forward lean (shoulders lower than hips in image coordinates)
            if shoulder_center_y > hip_center_y:
                angles['waist'] = unsigned_angle  # Positive for forward
                angles['waist_direction'] = "forward"
            else:
                angles['waist'] = -unsigned_angle  # Negative for backward
                angles['waist_direction'] = "backward"
            
            imputed_angles['waist'] = False
        else:
            angles['waist'] = neutral_angles['waist']
            imputed_angles['waist'] = True
    else:
        angles['waist'] = neutral_angles['waist']
        imputed_angles['waist'] = True

    # Calculate neck angle using ear instead of nose
    # Try left ear first, then right ear if left isn't available
    ear_point = get_keypoint_if_valid(validated_keypoints, 'left_ear')
    if ear_point is None:
        ear_point = get_keypoint_if_valid(validated_keypoints, 'right_ear')
    
    if ear_point is not None and shoulder_left is not None and shoulder_right is not None:
        # Calculate midpoint between shoulders
        mid_shoulder = ((shoulder_left[0] + shoulder_right[0])/2, 
                       (shoulder_left[1] + shoulder_right[1])/2)
        
        # Calculate angle between ear and mid-shoulder point (vertical line)
        # Create a point directly above mid_shoulder (same x, lower y in image coordinates)
        vertical_point = (mid_shoulder[0] - 1, mid_shoulder[1])
        
        try:
            angle = calculate_angle(ear_point, mid_shoulder, vertical_point)
            angles['neck'] = angle
            imputed_angles['neck'] = False
        except:
            angles['neck'] = neutral_angles['neck']
            imputed_angles['neck'] = True
    else:
        angles['neck'] = neutral_angles['neck']
        imputed_angles['neck'] = True

    # Calculate other angles
    angle_mapping = {
        'left_upper_arm': ('left_hip', 'left_shoulder', 'left_elbow'),
        'right_upper_arm': ('right_hip', 'right_shoulder', 'right_elbow'),
        'left_lower_arm': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_lower_arm': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_leg': ('left_hip', 'left_knee', 'left_ankle'),
        'right_leg': ('right_hip', 'right_knee', 'right_ankle')
    }

    for angle_name, points in angle_mapping.items():
        angles[angle_name] = calculate_angle_with_fallback(
            points[0], points[1], points[2], angle_name,
            validated_keypoints, imputed_angles, neutral_angles)

    # Apply smoothing
    for angle_name in angles:
        if angle_name in get_joint_angles.smoothers:
            angles[angle_name] = get_joint_angles.smoothers[angle_name].smooth(angles[angle_name])

    # Check if we have minimum required angles (neck + waist + at least one other)
    has_minimum_angles = not imputed_angles['neck'] and not imputed_angles['waist']
    
    # Also store imputation information
    angles['imputed_angles'] = imputed_angles
    
    return angles if has_minimum_angles else None

def create_row_dict(angles, filename, frame_num):
    """Create a dictionary representing one row of data"""
    if angles is None:
        return None
        
    imputed_angles = angles.get('imputed_angles', {})
    
    row = {
        'File Name': filename,
        'Frame': frame_num,
        
        # Core Angles
        'Neck Angle': angles.get('neck', -1),
        'Left Upper Arm Angle': angles.get('left_upper_arm', -1),
        'Right Upper Arm Angle': angles.get('right_upper_arm', -1),
        'Left Lower Arm Angle': angles.get('left_lower_arm', -1),
        'Right Lower Arm Angle': angles.get('right_lower_arm', -1),
        'Waist Angle': angles.get('waist', -1),
        'Left Leg Angle': angles.get('left_leg', -1),
        'Right Leg Angle': angles.get('right_leg', -1),
        
        # Imputation Flags
        'Left Arm Imputed': int(imputed_angles.get('left_upper_arm', False)),
        'Right Arm Imputed': int(imputed_angles.get('right_upper_arm', False)),
        'Left Leg Imputed': int(imputed_angles.get('left_leg', False)),
        'Right Leg Imputed': int(imputed_angles.get('right_leg', False)),
    }
    
    return row

def engineer_features_for_single_image(row_dict, resources):
    """Engineer features for a single image"""
    core_angles = resources['core_angles']
    
    # Create DataFrame for single row
    df = pd.DataFrame([row_dict])
    
    # Calculate imputation penalty
    df['imputation_penalty'] = df[[
        'Left Arm Imputed', 'Right Arm Imputed',
        'Left Leg Imputed', 'Right Leg Imputed'
    ]].sum(axis=1) * 0.2
    
    # Neck flex score calculation
    df['neck_flex_degree'] = df['Neck Angle']  # Convert to flexion angle
    df['neck_flex_score'] = 0  # Initialize
    
    # Apply logic from NeckREBA class
    df.loc[df['neck_flex_degree'].between(0, 20, inclusive='left'), 'neck_flex_score'] = 1
    df.loc[df['neck_flex_degree'].between(20, 45, inclusive='left'), 'neck_flex_score'] = 2
    df.loc[df['neck_flex_degree'].between(45, 60, inclusive='left'), 'neck_flex_score'] = 3
    df.loc[df['neck_flex_degree'] >= 60, 'neck_flex_score'] = 4
    df.loc[df['neck_flex_degree'] < 0, 'neck_flex_score'] = np.where(
        df.loc[df['neck_flex_degree'] < 0, 'neck_flex_degree'] >= -20, 2, 3
    )

    # Trunk flex score calculation
    df['trunk_flex_score'] = 0  # Initialize
    waist_angle = df['Waist Angle']

    # Forward flexion cases (positive angle)
    df.loc[waist_angle.between(0, 5, inclusive='both'), 'trunk_flex_score'] = 1
    df.loc[waist_angle.between(5, 20, inclusive='right'), 'trunk_flex_score'] = 2
    df.loc[waist_angle.between(20, 60, inclusive='right'), 'trunk_flex_score'] = 3
    df.loc[waist_angle.between(60, 90, inclusive='right'), 'trunk_flex_score'] = 4
    df.loc[waist_angle.between(90, 120, inclusive='right'), 'trunk_flex_score'] = 5
    df.loc[waist_angle > 120, 'trunk_flex_score'] = 6

    # Extension cases (negative angle)
    abs_ext_angle = np.abs(df.loc[waist_angle < 0, 'Waist Angle'])
    df.loc[waist_angle < 0, 'trunk_flex_score'] = np.where(
        abs_ext_angle <= 5, 1,
        np.where(abs_ext_angle <= 20, 2,
                 np.where(abs_ext_angle < 45, 3, 4))
    )

    # Upper arm score calculation using UpperArmREBA logic
    df['max_upper_arm_angle'] = df[['Left Upper Arm Angle', 'Right Upper Arm Angle']].abs().max(axis=1)
    df['upper_arm_score'] = 0  # Initialize

    df.loc[df['max_upper_arm_angle'].between(-20, 20, inclusive='left'), 'upper_arm_score'] = 1
    df.loc[df['max_upper_arm_angle'].between(20, 45, inclusive='left'), 'upper_arm_score'] = 2
    df.loc[(df['max_upper_arm_angle'] < -20) | 
           (df['max_upper_arm_angle'].between(45, 90, inclusive='left')), 'upper_arm_score'] = 3
    df.loc[df['max_upper_arm_angle'] >= 90, 'upper_arm_score'] = 4

    # Lower arm score calculation using LAREBA logic
    def score_lower_arm(angle):
        return 1 if 60 <= angle < 100 else 2

    df['left_lower_arm_score'] = df['Left Lower Arm Angle'].apply(score_lower_arm)
    df['right_lower_arm_score'] = df['Right Lower Arm Angle'].apply(score_lower_arm)
    df['lower_arm_score'] = df[['left_lower_arm_score', 'right_lower_arm_score']].max(axis=1)

    # Leg score calculation
    def calc_leg_deviation(angle):
        return min(abs(angle - 90), abs(angle - 110))

    df['left_leg_deviation'] = df['Left Leg Angle'].apply(calc_leg_deviation)
    df['right_leg_deviation'] = df['Right Leg Angle'].apply(calc_leg_deviation)
    df['max_leg_deviation'] = df[['left_leg_deviation', 'right_leg_deviation']].max(axis=1)

    df['leg_score'] = 1  # Initialize
    df.loc[df['max_leg_deviation'].between(0, 20, inclusive='right'), 'leg_score'] = 2
    df.loc[df['max_leg_deviation'].between(20, 40, inclusive='right'), 'leg_score'] = 3
    df.loc[df['max_leg_deviation'] > 40, 'leg_score'] = 4
    
    # Simplified Activity Score (for images, set to default)
    df['static_posture'] = 1  # Assume static for single image
    df['rapid_movement'] = 0  # No movement in a single image
    df['activity_score'] = 1  # Default for static posture
    
    # Set all temporal features to 0 for single image
    df['Neck Angle_volatility_1s'] = 0
    df['Waist Angle_volatility_1s'] = 0
    df['Neck Angle_stability_2s'] = 0
    df['Waist Angle_stability_2s'] = 0
    df['neck_awkward_time_5s'] = 0
    df['trunk_awkward_time_5s'] = 0
    
    # Posture Symmetry
    df['arm_asymmetry'] = np.abs(df['Left Upper Arm Angle'] - df['Right Upper Arm Angle'])
    df['leg_asymmetry'] = np.abs(df['Left Leg Angle'] - df['Right Leg Angle'])
    
    # Return only the features needed for prediction
    features_dict = df[resources['model_features']].iloc[0].to_dict()
    component_scores = {
        'trunk_score': int(df['trunk_flex_score'].iloc[0]),
        'neck_score': int(df['neck_flex_score'].iloc[0]),
        'upper_arm_score': int(df['upper_arm_score'].iloc[0]),
        'lower_arm_score': int(df['lower_arm_score'].iloc[0]),
        'leg_score': int(df['leg_score'].iloc[0])
    }
    
    return features_dict, component_scores

def process_pose_from_bytes(image_bytes, output_visualization=True):
    """
    Process an image from bytes, detect pose, and generate predictions
    
    Args:
        image_bytes: Image data as bytes
        output_visualization: Whether to generate visualization
        
    Returns:
        dict: Results including REBA score, risk level, component scores, etc.
    """
    try:
        # Initialize MoveNet
        movenet, input_size = init_movenet()
        if movenet is None:
            raise ValueError("Could not initialize MoveNet model")
            
        # Load model resources
        resources = get_model_resources()
        if resources is None:
            raise ValueError("Could not load model resources")
            
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image data")
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        keypoints, processed_img, original_img, flip_required = run_inference(
            movenet, frame, init_crop_region(frame.shape[0], frame.shape[1]), 
            crop_size=[input_size, input_size])
        
        # Calculate joint angles
        angles = get_joint_angles(keypoints)
        if angles is None:
            raise ValueError("Insufficient keypoints detected in image")
        
        # Create row dictionary
        filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        row_dict = create_row_dict(angles, filename, 0)
        
        # Engineer features
        features, component_scores = engineer_features_for_single_image(row_dict, resources)
        
        # Make prediction
        reba_score = predict_single_image(features, resources)
        
        # Generate feedback
        feedback = generate_feedback(component_scores, reba_score)
        
        # Add REBA score to component scores for visualization
        component_scores['reba_score'] = reba_score
        
        visualization_path = None
        if output_visualization:
            # Create output directory if it doesn't exist
            date_str = datetime.now().strftime("%Y-%m-%d")
            folder_path = os.path.join("output_images", date_str)
            os.makedirs(folder_path, exist_ok=True)
            
            # Generate visualization
            visualization_filename = f"{filename}_{int(reba_score)}_{datetime.now().strftime('%H%M%S')}.png"
            visualization_path = os.path.join(folder_path, visualization_filename)
            
            # Generate and save visualization
            visualization = generate_pose_visualization(
                processed_img, keypoints, component_scores, original_img, flip_required
            )
            
            # Save visualization
            cv2.imwrite(visualization_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        # Store actual angle values
        angle_values = {
            'neck': float(angles['neck']),
            'waist': float(angles['waist']),
            'left_upper_arm': float(angles['left_upper_arm']),
            'right_upper_arm': float(angles['right_upper_arm']),
            'left_lower_arm': float(angles['left_lower_arm']),
            'right_lower_arm': float(angles['right_lower_arm']),
            'left_leg': float(angles['left_leg']),
            'right_leg': float(angles['right_leg'])
        }
        
        # Create result dictionary
        result = {
            'reba_score': float(reba_score),
            'risk_level': get_risk_level(reba_score),
            'component_scores': component_scores,
            'angle_values': angle_values,
            'feedback': feedback
        }
        
        if visualization_path:
            result['visualization_path'] = os.path.join(date_str, os.path.basename(visualization_path))
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
