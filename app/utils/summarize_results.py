import numpy as np
from app.services.ergonomic_model import get_risk_level, get_action_level

def calculate_reba_statistics(results):
    """Calculate statistics for REBA scores"""
    reba_scores = [result['reba_score'] for result in results]
    
    return {
        'min': float(np.min(reba_scores)),
        'max': float(np.max(reba_scores)),
        'avg': float(np.mean(reba_scores)),
        'median': float(np.median(reba_scores)),
        'std': float(np.std(reba_scores))
    }

def calculate_component_averages(results):
    """Calculate average component scores"""
    component_types = ['trunk', 'neck', 'upper_arm', 'lower_arm', 'leg']
    
    averages = {}
    for component in component_types:
        scores = [result['component_scores'][component] for result in results]
        averages[component] = float(np.mean(scores))
    
    return averages

def calculate_angle_statistics(results):
    """Calculate statistics for joint angles"""
    angle_types = [
        'neck', 'waist', 'left_upper_arm', 'right_upper_arm',
        'left_lower_arm', 'right_lower_arm', 'left_leg', 'right_leg'
    ]
    
    angle_stats = {}
    for angle_type in angle_types:
        angles = [result['angle_values'][angle_type] for result in results]
        angle_stats[angle_type] = {
            'min': float(np.min(angles)),
            'max': float(np.max(angles)),
            'avg': float(np.mean(angles)),
            'std': float(np.std(angles))
        }
    
    return angle_stats

def identify_high_risk_periods(results, threshold=7.0, min_duration=3):
    """Identify periods of high ergonomic risk"""
    if len(results) < 2:
        return []
        
    # Sort results by frame number
    sorted_results = sorted(results, key=lambda x: x['frame'])
    
    # Find continuous sequences above threshold
    high_risk_periods = []
    current_period = None
    
    for i, result in enumerate(sorted_results):
        if result['reba_score'] >= threshold:
            if current_period is None:
                current_period = {
                    'start_frame': result['frame'],
                    'scores': [result['reba_score']]
                }
            else:
                current_period['scores'].append(result['reba_score'])
        else:
            if current_period is not None:
                current_period['end_frame'] = sorted_results[i-1]['frame']
                # Check if period is long enough
                if len(current_period['scores']) >= min_duration:
                    current_period['avg_score'] = float(np.mean(current_period['scores']))
                    high_risk_periods.append(current_period)
                current_period = None
    
    # Handle the case where the video ends during a high risk period
    if current_period is not None:
        current_period['end_frame'] = sorted_results[-1]['frame']
        if len(current_period['scores']) >= min_duration:
            current_period['avg_score'] = float(np.mean(current_period['scores']))
            high_risk_periods.append(current_period)
    
    return high_risk_periods

def generate_recommendations(results):
    """Generate recommendations based on analysis results"""
    reba_scores = [result['reba_score'] for result in results]
    avg_reba = np.mean(reba_scores)
    
    component_averages = calculate_component_averages(results)
    
    recommendations = []
    
    # Trunk recommendations
    if component_averages['trunk'] >= 3:
        recommendations.append("Adjust your work height to avoid excessive trunk bending.")
        recommendations.append("Consider using a supportive chair that maintains proper spinal alignment.")
    
    # Neck recommendations
    if component_averages['neck'] >= 2:
        recommendations.append("Position your monitor at eye level to reduce neck strain.")
        recommendations.append("Take regular breaks to relieve neck tension.")
    
    # Upper arm recommendations
    if component_averages['upper_arm'] >= 3:
        recommendations.append("Lower your work surface to keep arms closer to your body.")
        recommendations.append("Use armrests when appropriate to reduce shoulder strain.")
    
    # Lower arm recommendations
    if component_averages['lower_arm'] >= 2:
        recommendations.append("Adjust workstation height to maintain 90-110° elbow angles.")
    
    # Leg recommendations
    if component_averages['leg'] >= 2:
        recommendations.append("Ensure even weight distribution between both legs.")
        recommendations.append("Use an anti-fatigue mat if standing for long periods.")
    
    # General recommendations based on overall REBA score
    if avg_reba <= 3:
        recommendations.append("Continue maintaining good posture with minor adjustments.")
    elif avg_reba <= 7:
        recommendations.append("Consider ergonomic adjustments to your workstation.")
        recommendations.append("Take regular breaks to change posture and reduce strain.")
    else:
        recommendations.append("Immediate action needed to redesign this work task.")
        recommendations.append("Consider ergonomic consultation to address high-risk factors.")
    
    return recommendations

def summarize_results(results):
    """
    Generate a comprehensive summary of ergonomic analysis results
    
    Args:
        results: List of frame-by-frame result dictionaries
        
    Returns:
        dict: Summary statistics and recommendations
    """
    if not results:
        return {"error": "No valid results to summarize"}
    
    # Calculate REBA score statistics
    reba_stats = calculate_reba_statistics(results)
    
    # Calculate average component scores
    avg_component_scores = calculate_component_averages(results)
    
    # Calculate angle statistics
    angle_stats = calculate_angle_statistics(results)
    
    # Identify high risk periods
    high_risk_periods = identify_high_risk_periods(results)
    
    # Generate recommendations
    recommendations = generate_recommendations(results)
    
    # Determine overall risk level
    risk_level = get_risk_level(reba_stats['avg'])
    action_level, action_text = get_action_level(reba_stats['avg'])
    
    # Create summary
    summary = {
        "avg_reba_score": reba_stats['avg'],
        "reba_statistics": reba_stats,
        "risk_level": risk_level,
        "action_level": action_level,
        "action_text": action_text,
        "avg_component_scores": avg_component_scores,
        "angle_statistics": angle_stats,
        "high_risk_periods_count": len(high_risk_periods),
        "high_risk_periods": high_risk_periods,
        "recommendations": recommendations
    }
    
    return summary
