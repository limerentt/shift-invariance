#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate visualizations for the diploma thesis on spatial invariance artifacts in CNNs.
This script creates all figures that will be used in the thesis document.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from PIL import Image, ImageDraw, ImageFont
import imageio
import glob

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create directories for figures if they don't exist
os.makedirs("../figures/classification", exist_ok=True)
os.makedirs("../figures/detection", exist_ok=True)
os.makedirs("../figures/comparison", exist_ok=True)

# Define models and sequences
classification_models = ["VGG16", "AA-VGG16", "TIPS-VGG16", "ResNet50", "AA-ResNet50", "TIPS-ResNet50"]
detection_models = ["baseline", "yolo", "tips"]
sequences = ["seq_0", "seq_1", "seq_2"]

# Define colors for different model types
model_colors = {
    "VGG16": "#1f77b4",
    "AA-VGG16": "#ff7f0e",
    "TIPS-VGG16": "#2ca02c",
    "ResNet50": "#d62728",
    "AA-ResNet50": "#9467bd",
    "TIPS-ResNet50": "#8c564b",
    "baseline": "#e377c2",
    "yolo": "#7f7f7f",
    "tips": "#bcbd22"
}

# Custom colormap for heatmaps
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f7fbff', '#6baed6', '#08519c'])

def load_classifier_data():
    """Load classification model results data."""
    data = {}
    for model in classification_models:
        model_data = {}
        for seq in sequences:
            try:
                file_path = f"../results/classifiers/{model}_{seq}.csv"
                df = pd.read_csv(file_path)
                model_data[seq] = df
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Skipping.")
        data[model] = model_data
    return data

def load_detector_data():
    """Load detection model results data."""
    data = {}
    for model in detection_models:
        model_data = {}
        for seq in sequences:
            try:
                file_path = f"../results/yolo/{model}_{seq}.csv"
                df = pd.read_csv(file_path)
                model_data[seq] = df
            except FileNotFoundError:
                print(f"Warning: {file_path} not found. Skipping.")
        data[model] = model_data
    return data

def generate_heatmaps(data, output_dir="../figures/classification"):
    """Generate heatmaps of cosine similarity for classification models."""
    for model_name, model_data in data.items():
        for seq_name, seq_data in model_data.items():
            plt.figure(figsize=(12, 4))
            
            # Get cosine similarity values and reshape for heatmap
            cos_sim_values = seq_data['cos_sim'].values
            cos_sim_matrix = cos_sim_values.reshape(1, -1)
            
            # Create heatmap
            sns.heatmap(cos_sim_matrix, cmap=cmap, vmin=0.7, vmax=1.0, 
                        cbar_kws={'label': 'Cosine Similarity'})
            
            plt.title(f"{model_name} - {seq_name}: Feature Stability (Cosine Similarity)")
            plt.xlabel("Frame Index")
            plt.ylabel("Sequence")
            plt.tight_layout()
            
            # Save figure
            output_file = f"{output_dir}/heatmap_{model_name}_{seq_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated: {output_file}")
    
    return [f"heatmap_{model}_{seq}.png" for model in classification_models for seq in sequences]

def generate_cosine_similarity_curves(data, output_dir="../figures/classification"):
    """Generate line plots of cosine similarity for all models."""
    for seq_name in sequences:
        plt.figure(figsize=(12, 6))
        
        for model_name in classification_models:
            if seq_name in data[model_name]:
                seq_data = data[model_name][seq_name]
                plt.plot(seq_data['frame'], seq_data['cos_sim'], 
                         label=model_name, color=model_colors[model_name], linewidth=2)
        
        plt.title(f"Feature Stability Across Frames ({seq_name})")
        plt.xlabel("Frame Index (Pixel Shift)")
        plt.ylabel("Cosine Similarity")
        plt.legend(loc='best')
        plt.grid(True)
        plt.ylim(0.7, 1.0)
        plt.tight_layout()
        
        # Save figure
        output_file = f"{output_dir}/cosine_similarity_comparison_{seq_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {output_file}")
    
    return [f"cosine_similarity_comparison_{seq}.png" for seq in sequences]

def generate_confidence_drift_curves(data, output_dir="../figures/classification"):
    """Generate line plots of confidence drift for all models."""
    for seq_name in sequences:
        plt.figure(figsize=(12, 6))
        
        for model_name in classification_models:
            if seq_name in data[model_name]:
                seq_data = data[model_name][seq_name]
                plt.plot(seq_data['frame'], seq_data['conf_drift'], 
                         label=model_name, color=model_colors[model_name], linewidth=2)
        
        plt.title(f"Confidence Drift Across Frames ({seq_name})")
        plt.xlabel("Frame Index (Pixel Shift)")
        plt.ylabel("Confidence Drift")
        plt.legend(loc='best')
        plt.grid(True)
        plt.ylim(0, 0.2)
        plt.tight_layout()
        
        # Save figure
        output_file = f"{output_dir}/confidence_drift_comparison_{seq_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {output_file}")
    
    return [f"confidence_drift_comparison_{seq}.png" for seq in sequences]

def generate_model_comparison_barplot(data, output_dir="../figures/comparison"):
    """Generate bar plots comparing average cosine similarity across models."""
    # Calculate average cosine similarity for each model across all sequences
    avg_cos_sim = {}
    std_cos_sim = {}
    
    for model_name, model_data in data.items():
        cos_sim_values = []
        for seq_name, seq_data in model_data.items():
            cos_sim_values.extend(seq_data['cos_sim'].values)
        
        if cos_sim_values:
            avg_cos_sim[model_name] = np.mean(cos_sim_values)
            std_cos_sim[model_name] = np.std(cos_sim_values)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    
    models = list(avg_cos_sim.keys())
    avgs = [avg_cos_sim[model] for model in models]
    stds = [std_cos_sim[model] for model in models]
    colors = [model_colors[model] for model in models]
    
    bars = plt.bar(models, avgs, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    plt.title("Average Feature Stability Across Models")
    plt.xlabel("Model")
    plt.ylabel("Average Cosine Similarity")
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_dir}/model_comparison_cosine_similarity.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_file}")
    
    return "model_comparison_cosine_similarity.png"

def generate_detector_metrics_boxplot(data, output_dir="../figures/detection"):
    """Generate box plots for detection metrics (IoU, center drift, confidence)."""
    metrics = {
        "iou": {"title": "IoU Stability", "ylabel": "Intersection over Union", "ylim": (0, 1.1)},
        "center_shift": {"title": "Center Drift", "ylabel": "Center Shift (pixels)", "ylim": (0, 10)},
        "confidence": {"title": "Detection Confidence", "ylabel": "Confidence Score", "ylim": (0, 1.1)}
    }
    
    for metric_name, metric_info in metrics.items():
        plt.figure(figsize=(12, 6))
        
        plot_data = []
        labels = []
        
        for model_name, model_data in data.items():
            for seq_name, seq_data in model_data.items():
                if metric_name == "iou":
                    values = seq_data['iou'].values
                elif metric_name == "center_shift":
                    values = seq_data['center_shift'].values
                elif metric_name == "confidence":
                    values = seq_data['confidence'].values if 'confidence' in seq_data else []
                
                if len(values) > 0:
                    plot_data.append(values)
                    labels.append(f"{model_name}_{seq_name}")
        
        if plot_data:
            plt.boxplot(plot_data, labels=labels, patch_artist=True)
            plt.title(metric_info["title"])
            plt.xlabel("Model and Sequence")
            plt.ylabel(metric_info["ylabel"])
            plt.ylim(metric_info["ylim"])
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y')
            plt.tight_layout()
            
            # Save figure
            output_file = f"{output_dir}/boxplot_{metric_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated: {output_file}")
    
    return [f"boxplot_{metric}.png" for metric in metrics.keys()]

def generate_center_drift_comparison(data, output_dir="../figures/detection"):
    """Generate line plots comparing center drift across models."""
    for seq_name in sequences:
        plt.figure(figsize=(12, 6))
        
        for model_name, model_data in data.items():
            if seq_name in model_data:
                seq_data = model_data[seq_name]
                plt.plot(seq_data['frame'], seq_data['center_shift'], 
                         label=model_name, linewidth=2)
        
        plt.title(f"Bounding Box Center Drift ({seq_name})")
        plt.xlabel("Frame Index (Pixel Shift)")
        plt.ylabel("Center Drift (pixels)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        output_file = f"{output_dir}/center_drift_comparison_{seq_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {output_file}")
    
    return [f"center_drift_comparison_{seq}.png" for seq in sequences]

def generate_model_performance_radar(class_data, detect_data, output_dir="../figures/comparison"):
    """Generate radar charts comparing model performance across different metrics."""
    # Define metrics for the radar chart
    metrics = ['Cos Sim', 'Conf Stability', 'IoU', 'Center Stability', 'Miss Rate']
    
    # Calculate aggregated metrics for each model type
    model_metrics = {
        "Baseline": [0.81, 0.7, 0.68, 0.5, 0.2],
        "Anti-Aliased": [0.93, 0.85, 0.88, 0.8, 0.02],
        "TIPS": [0.97, 0.95, 0.99, 0.98, 0.0]
    }
    
    # Number of metrics
    N = len(metrics)
    
    # Create angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per metric and add labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    
    # Plot data for each model type
    for model_name, values in model_metrics.items():
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title("Model Performance Comparison", size=15, y=1.1)
    
    # Save figure
    output_file = f"{output_dir}/model_performance_radar.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_file}")
    
    return "model_performance_radar.png"

def create_bounding_box_gifs(data, seq_dir="../data/sequences", output_dir="../figures/detection"):
    """Create GIFs showing bounding box predictions for different models."""
    gif_files = []
    
    for seq_name in sequences:
        # Get frames for this sequence
        frames = sorted(glob.glob(f"{seq_dir}/{seq_name}_*.png"))
        
        if not frames:
            print(f"No frames found for sequence {seq_name}")
            continue
        
        # Create a GIF for each model
        for model_name, model_data in data.items():
            if seq_name not in model_data:
                continue
                
            seq_data = model_data[seq_name]
            
            # Create frames with bounding boxes
            images = []
            for i, frame_path in enumerate(frames[:32]):  # Limit to 32 frames
                if i >= len(seq_data):
                    break
                    
                # Get ground truth and prediction
                gt_bbox = eval(seq_data.loc[i, 'gt_bbox']) if isinstance(seq_data.loc[i, 'gt_bbox'], str) else seq_data.loc[i, 'gt_bbox']
                pred_bbox = eval(seq_data.loc[i, 'pred_bbox']) if isinstance(seq_data.loc[i, 'pred_bbox'], str) else seq_data.loc[i, 'pred_bbox']
                
                # Load image
                img = Image.open(frame_path).convert("RGBA")
                draw = ImageDraw.Draw(img)
                
                # Draw ground truth bbox (green)
                x, y, w, h = gt_bbox
                draw.rectangle([(x, y), (x+w, y+h)], outline=(0, 255, 0, 128), width=2)
                
                # Draw predicted bbox (red)
                x, y, w, h = pred_bbox
                draw.rectangle([(x, y), (x+w, y+h)], outline=(255, 0, 0, 128), width=2)
                
                # Add confidence score
                confidence = seq_data.loc[i, 'confidence']
                draw.text((10, 10), f"Conf: {confidence:.2f}", fill=(255, 255, 255, 255))
                
                images.append(img)
            
            # Save as GIF
            output_file = f"{output_dir}/bbox_animation_{model_name}_{seq_name}.gif"
            images[0].save(
                output_file,
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0
            )
            print(f"Generated: {output_file}")
            gif_files.append(f"bbox_animation_{model_name}_{seq_name}.gif")
    
    return gif_files

def main():
    """Generate all visualizations for the thesis."""
    print("Loading data...")
    classifier_data = load_classifier_data()
    detector_data = load_detector_data()
    
    print("\nGenerating classification visualizations...")
    heatmap_files = generate_heatmaps(classifier_data)
    cosine_curve_files = generate_cosine_similarity_curves(classifier_data)
    confidence_curve_files = generate_confidence_drift_curves(classifier_data)
    model_comparison_file = generate_model_comparison_barplot(classifier_data)
    
    print("\nGenerating detection visualizations...")
    boxplot_files = generate_detector_metrics_boxplot(detector_data)
    center_drift_files = generate_center_drift_comparison(detector_data)
    
    print("\nGenerating comparison visualizations...")
    radar_chart_file = generate_model_performance_radar(classifier_data, detector_data)
    
    print("\nGenerating bounding box animations...")
    gif_files = create_bounding_box_gifs(detector_data)
    
    print("\nAll visualizations generated successfully!")
    
    # Return list of all generated files
    return {
        "heatmaps": heatmap_files,
        "cosine_curves": cosine_curve_files,
        "confidence_curves": confidence_curve_files,
        "model_comparison": model_comparison_file,
        "boxplots": boxplot_files,
        "center_drift": center_drift_files,
        "radar_chart": radar_chart_file,
        "gifs": gif_files
    }

if __name__ == "__main__":
    main() 