import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Constants
KEYPOINT_THRESHOLD = 0.3
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3,
    'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 
    'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9,
    'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Edge colors for visualization
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (5, 3): 'r', (6, 4): 'r', (5, 7): 'b', (7, 9): 'b', (6, 8): 'b', (8, 10): 'b',
    (5, 6): 'b', (5, 11): 'orange', (6, 12): 'orange', (7, 5): 'g', (7, 9): 'g',
    (8, 6): 'g', (8, 10): 'g', (11, 13): 'purple', (13, 15): 'purple', (12, 14): 'purple', (14, 16): 'purple'
}

# Risk level colors
RISK_COLORS = {
    'trunk': {
        1: 'darkgreen',
        2: 'lightgreen',
        3: 'yellow',
        4: 'red',
        5: 'red',
        6: 'red'
    },
    'neck': {
        1: 'green',
        2: 'yellow',
        3: 'red',
        4: 'red'
    },
    'upper_arm': {
        1: 'green',
        2: 'yellow',
        3: 'yellow',
        4: 'red'
    },
    'lower_arm': {
        1: 'green',
        2: 'yellow'
    },
    'leg': {
        1: 'darkgreen',
        2: 'lightgreen',
        3: 'yellow',
        4: 'red'
    }
}

def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=KEYPOINT_THRESHOLD):
    """Prepare keypoints and edges for visualization"""
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape

    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if kpts_scores[edge_pair[0]] > keypoint_threshold and kpts_scores[edge_pair[1]] > keypoint_threshold:
                x_start, y_start = kpts_absolute_xy[edge_pair[0]]
                x_end, y_end = kpts_absolute_xy[edge_pair[1]]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)

    keypoints_xy = np.concatenate(keypoints_all, axis=0) if keypoints_all else np.zeros((0, 2))
    edges_xy = np.stack(keypoint_edges_all, axis=0) if keypoint_edges_all else np.zeros((0, 2, 2))

    return keypoints_xy, edges_xy, edge_colors

def generate_pose_visualization(image, keypoints_with_scores, risk_scores, original_image=None, 
                               flip_applied=False, crop_region=None, output_image_height=None):
    """Draws keypoint predictions with risk level coloring"""
    # Use original image if flip was applied
    vis_image = original_image if flip_applied else image
    
    if flip_applied:
        # Flip the keypoints back to match the flipped image
        width = vis_image.shape[1]
        keypoints_with_scores[0, 0, :, 1] = 1 - keypoints_with_scores[0, 0, :, 1]

    height, width, _ = vis_image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(vis_image)
    
    # Basic keypoints and edges
    (keypoint_locs, keypoint_edges, _) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)
    
    # Draw risk-colored edges
    # Define the body segments and their risk types
    # Updated to use ear instead of nose for neck connections
    risk_segments = {
        # Neck (connecting ear to shoulders)
        ((3, 5), (4, 6)): 'neck',  
        
        # Trunk (connecting shoulders to hips)
        ((5, 11), (6, 12)): 'trunk',
        
        # Upper arms
        ((5, 7), (6, 8)): 'upper_arm',
        
        # Lower arms
        ((7, 9), (8, 10)): 'lower_arm',
        
        # Legs
        ((11, 13), (12, 14), (13, 15), (14, 16)): 'leg'
    }
    
    # Draw connection between shoulders
    # This helps with visualization but doesn't have a risk color
    if (keypoints_with_scores[0, 0, 5, 2] > KEYPOINT_THRESHOLD and 
        keypoints_with_scores[0, 0, 6, 2] > KEYPOINT_THRESHOLD):
        x1 = keypoints_with_scores[0, 0, 5, 1] * width
        y1 = keypoints_with_scores[0, 0, 5, 0] * height
        x2 = keypoints_with_scores[0, 0, 6, 1] * width
        y2 = keypoints_with_scores[0, 0, 6, 0] * height
        
        shoulder_line = plt.Line2D([x1, x2], [y1, y2], lw=3, color='blue', zorder=2)
        ax.add_line(shoulder_line)
    
    # Draw edges with risk-based coloring
    for segment_pairs, risk_type in risk_segments.items():
        risk_level = risk_scores.get(f'{risk_type}_score', 1)
        color = RISK_COLORS[risk_type].get(risk_level, 'gray')
        
        for edge_pair in segment_pairs:
            idx1, idx2 = edge_pair
            
            # Check if keypoints exist and have sufficient confidence
            if (keypoints_with_scores[0, 0, idx1, 2] > KEYPOINT_THRESHOLD and 
                keypoints_with_scores[0, 0, idx2, 2] > KEYPOINT_THRESHOLD):
                
                x1 = keypoints_with_scores[0, 0, idx1, 1] * width
                y1 = keypoints_with_scores[0, 0, idx1, 0] * height
                x2 = keypoints_with_scores[0, 0, idx2, 1] * width
                y2 = keypoints_with_scores[0, 0, idx2, 0] * height
                
                line = plt.Line2D([x1, x2], [y1, y2], lw=4, color=color, zorder=2)
                ax.add_line(line)
    
    # Draw keypoints
    valid_keypoints = []
    for i, score in enumerate(keypoints_with_scores[0, 0, :, 2]):
        if score > KEYPOINT_THRESHOLD:
            x = keypoints_with_scores[0, 0, i, 1] * width
            y = keypoints_with_scores[0, 0, i, 0] * height
            valid_keypoints.append((x, y))
    
    if valid_keypoints:
        keypoints_x, keypoints_y = zip(*valid_keypoints)
        ax.scatter(keypoints_x, keypoints_y, s=60, color='white', edgecolor='black', zorder=3)
    
    # Add predicted REBA score text
    if 'reba_score' in risk_scores:
        # Determine risk level color
        reba_score = risk_scores['reba_score']
        if reba_score <= 2:
            reba_color = 'darkgreen'
        elif reba_score <= 4:
            reba_color = 'orange'
        elif reba_score <= 7:
            reba_color = 'darkorange'
        else:
            reba_color = 'darkred'
            
        text = f"REBA Score: {reba_score:.1f}"
        ax.text(width * 0.05, height * 0.05, text, fontsize=18, 
                color='white', bbox=dict(facecolor=reba_color, alpha=0.8))
    
    # Add risk legend
    legend_y = height * 0.95
    ax.text(width * 0.05, legend_y, "Risk Levels:", fontsize=14, color='black', 
            bbox=dict(facecolor='white', alpha=0.7))
    
    legend_items = [
        ("Neck", risk_scores.get('neck_score', 0)),
        ("Trunk", risk_scores.get('trunk_score', 0)),
        ("Upper Arms", risk_scores.get('upper_arm_score', 0)),
        ("Lower Arms", risk_scores.get('lower_arm_score', 0)),
        ("Legs", risk_scores.get('leg_score', 0))
    ]
    
    for i, (name, level) in enumerate(legend_items):
        y_pos = legend_y - (i+1) * height * 0.05
        color = RISK_COLORS[name.lower().replace(" ", "_").replace("s", "")].get(level, 'gray')
        ax.text(width * 0.07, y_pos, f"• {name}: {level}", fontsize=12, color='black',
                bbox=dict(facecolor=color, alpha=0.7))
    
    # Add crop region if provided
    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin, ymin), rec_width, rec_height,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Convert figure to image
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Resize if requested
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(image_from_plot, dsize=(output_image_width, output_image_height),
                                     interpolation=cv2.INTER_CUBIC)
    return image_from_plot
