import os
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd  # Make sure this import is present
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
from datetime import datetime
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
KEYPOINT_THRESHOLD = 0.3
SEQUENCE_LENGTH = 60
STRIDE = 30
MAX_GAP = 30
MODEL_PATH = "modelv4/reba_model.h5"
PREPROCESSING_PATH = "modelv4/preprocessing.joblib"

# Initialize model and resources
_model = None
_resources = None

def get_model_resources():
    """Load and return model resources, with lazy initialization"""
    global _model, _resources
    
    if _resources is not None:
        return _resources
    
    try:
        print("Loading ergonomic assessment model resources...")
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load preprocessing data
        preprocessing_data = joblib.load(PREPROCESSING_PATH)
        
        _resources = {
            'model': model,
            'scaler': preprocessing_data['scaler'],
            'model_features': preprocessing_data['model_features'],
            'core_angles': preprocessing_data['core_angles'],
            'sequence_length': preprocessing_data.get('sequence_length', SEQUENCE_LENGTH),
            'max_gap': preprocessing_data.get('max_gap', MAX_GAP),
            'static_window': preprocessing_data.get('static_window', 30),
            'imputation_columns': preprocessing_data.get('imputation_columns', 
                                                      ['Left Arm Imputed', 'Right Arm Imputed',
                                                       'Left Leg Imputed', 'Right Leg Imputed'])
        }
        
        print("Model resources loaded successfully")
        return _resources
        
    except Exception as e:
        print(f"Error loading model resources: {e}")
        return None

def predict_single_image(features, resources=None):
    """Make prediction for a single image"""
    if resources is None:
        resources = get_model_resources()
        
    if resources is None:
        raise ValueError("Could not load model resources")
        
    model = resources['model']
    scaler = resources['scaler']
    features_arr = np.array([list(features.values())])
    
    # Scale the features
    scaled_features = scaler.transform(features_arr)
    
    # For single image, repeat the features to create a sequence
    decay = np.linspace(1.0, 0.95, SEQUENCE_LENGTH)[:, np.newaxis]
    sequence = np.tile(scaled_features, (SEQUENCE_LENGTH, 1)) * decay
    
    # Predict
    prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
    return float(prediction[0][0])

def predict_video(engineered_df, resources=None):
    """Make predictions for video sequences"""
    if resources is None:
        resources = get_model_resources()
        
    if resources is None:
        raise ValueError("Could not load model resources")
        
    model = resources['model']
    scaler = resources['scaler']
    model_features = resources['model_features']
    
    # Scale features
    scaled_features = scaler.transform(engineered_df[model_features])
    engineered_df[model_features] = scaled_features
    
    # Prepare sequences
    sequences = prepare_sequences(
        engineered_df, 
        model_features, 
        sequence_length=resources['sequence_length'],
        max_gap=resources['max_gap']
    )
    
    if sequences is None or len(sequences) == 0:
        print("⚠ No valid sequences found for prediction")
        return None
    
    # Predict
    predictions = np.array([
        model.predict(np.expand_dims(seq, axis=0), verbose=0).flatten()
        for seq in sequences
    ])
    
    # Return prediction statistics
    return {
        'predictions': predictions.flatten(),
        'average': float(np.mean(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'std': float(np.std(predictions))
    }

def prepare_sequences(df, model_features, sequence_length=SEQUENCE_LENGTH, stride=STRIDE, max_gap=MAX_GAP):
    """Prepare sequences for prediction"""
    sequences = []
    frames = df['Frame'].values
    data = df[model_features].values
    
    i = 0
    while i < len(df) - sequence_length + 1:
        seq_frames = frames[i:i+sequence_length]
        gaps = np.diff(seq_frames)
        
        if np.any(gaps > max_gap):
            bad_pos = np.where(gaps > max_gap)[0][0]
            i += bad_pos + 1
            continue
            
        sequences.append(data[i:i+sequence_length])
        i += stride
    
    return np.array(sequences) if sequences else None

def get_risk_level(reba_score):
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

def get_action_level(reba_score):
    """Get action level based on REBA score"""
    if reba_score <= 1:
        return 0, "No action necessary"
    elif reba_score <= 3:
        return 1, "Action may be needed"
    elif reba_score <= 7:
        return 2, "Action necessary"
    elif reba_score <= 10:
        return 3, "Action necessary soon"
    else:
        return 4, "Action necessary NOW"

def generate_feedback(component_scores, reba_score):
    """Generate textual feedback based on component scores and REBA score"""
    feedback = f"Overall REBA Score: {reba_score:.1f} - "
    
    # Add risk level
    risk_level = get_risk_level(reba_score)
    action_level, action_text = get_action_level(reba_score)
    feedback += f"{risk_level} Risk. {action_text}.\n\n"
    
    # Identify highest risk components
    components = []
    
    if component_scores['trunk_score'] >= 3:  # Changed 'trunk' to 'trunk_score'
        components.append(f"Trunk posture (score {component_scores['trunk_score']})")
    
    if component_scores['neck_score'] >= 2:  # Changed 'neck' to 'neck_score'
        components.append(f"Neck posture (score {component_scores['neck_score']})")
    
    if component_scores['upper_arm_score'] >= 3:  # Changed 'upper_arm' to 'upper_arm_score'
        components.append(f"Upper arm position (score {component_scores['upper_arm_score']})")
    
    if component_scores['lower_arm_score'] == 2:  # Changed 'lower_arm' to 'lower_arm_score'
        components.append("Lower arm position")
    
    if component_scores['leg_score'] >= 2:  # Changed 'leg' to 'leg_score'
        components.append(f"Leg posture (score {component_scores['leg_score']})")
        
    # Add component-specific feedback
    if components:
        feedback += "Focus on improving: " + ", ".join(components) + ".\n\n"
        
        # Add specific recommendations
        feedback += "Recommended actions:\n"
        
        if "Trunk posture" in " ".join(components):
            feedback += "- Keep your back straight when possible\n"
            feedback += "- Avoid excessive bending or twisting\n"
            
        if "Neck posture" in " ".join(components):
            feedback += "- Position work at eye level to reduce neck flexion\n"
            feedback += "- Avoid looking down for extended periods\n"
            
        if "Upper arm" in " ".join(components):
            feedback += "- Lower your work surface to reduce shoulder strain\n"
            feedback += "- Keep elbows close to your body when possible\n"
            
        if "Lower arm" in " ".join(components):
            feedback += "- Position work to allow 90-110° elbow angles\n"
            
        if "Leg posture" in " ".join(components):
            feedback += "- Ensure even weight distribution between legs\n"
            feedback += "- Avoid prolonged static standing in awkward positions\n"
    else:
        feedback += "Your posture shows no major ergonomic concerns at this time."
    
    return feedback

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

# Initialize model resources on import
get_model_resources()
