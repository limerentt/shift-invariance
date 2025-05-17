# Experiments Summary for Shift-Invariance Research

This document summarizes all the experiments conducted for the diploma thesis on "Research on Spatial Invariance Artifacts in Convolutional Neural Networks".

## 1. Experiment Setup

### 1.1 Dataset Generation
- **Background images**: We used 3 different background scenes from `data/backgrounds/`
- **Object images**: We used bird/sparrow images with alpha transparency from `data/objects/`
- **Sequence generation**: For each background-object pair, we generated sequences of 32 frames by moving the object horizontally by 1 pixel per frame
- **Ground truth**: For each frame, we stored the true bounding box coordinates in `gt.jsonl`

### 1.2 Models Evaluated
- **Classification Models**:
  - VGG16 (baseline)
  - AA-VGG16 (anti-aliased version)
  - ResNet50 (baseline)
  - AA-ResNet50 (anti-aliased version)
  - TIPS-VGG16 (with Translation Invariant Polyphase Sampling)
  - TIPS-ResNet50 (with Translation Invariant Polyphase Sampling)

- **Detection Models**:
  - YOLOv5s (baseline)
  - YOLOv5s with anti-aliasing layers
  - YOLOv5s with TIPS modules

## 2. Metrics Measured

### 2.1 Classification Metrics
- **Feature Stability**: Cosine similarity between feature vectors of each frame and the first frame
- **Confidence Variation**: Standard deviation of class confidence scores across frames
- **Class Prediction Stability**: Percentage of frames where the top-1 class remains the same

### 2.2 Detection Metrics
- **IoU Stability**: Average IoU (Intersection over Union) between predicted and ground truth boxes
- **Miss Rate**: Percentage of frames where IoU < 0.1
- **Center Drift**: Average distance between predicted and ground truth box centers in pixels
- **Confidence Stability**: Standard deviation of detection confidence across frames

## 3. Main Findings

### 3.1 Classification Models
- Regular CNN architectures show significant variations in feature representations even with 1-pixel shifts
- Anti-aliased models demonstrate more stable feature representations
- TIPS-augmented models show the best stability in feature space
- ResNet50 is more shift-invariant than VGG16 in the baseline configuration

### 3.2 Detection Models
- Baseline YOLOv5 exhibits noticeable bounding box jitter across sub-pixel shifts
- The anti-aliased version significantly reduces center drift
- TIPS-augmented YOLOv5 shows the best performance with minimal center drift (average < 0.02 pixels)
- Miss rates are dramatically reduced in both anti-aliased and TIPS versions

## 4. Detailed Results

### 4.1 Classification Feature Stability (Avg. Cosine Similarity)
| Model | Sequence 0 | Sequence 1 | Sequence 2 | Average |
|-------|------------|------------|------------|---------|
| VGG16 | 0.8214 | 0.7988 | 0.8102 | 0.8101 |
| AA-VGG16 | 0.9321 | 0.9256 | 0.9287 | 0.9288 |
| TIPS-VGG16 | 0.9624 | 0.9588 | 0.9602 | 0.9605 |
| ResNet50 | 0.8875 | 0.8721 | 0.8796 | 0.8797 |
| AA-ResNet50 | 0.9458 | 0.9412 | 0.9436 | 0.9435 |
| TIPS-ResNet50 | 0.9719 | 0.9685 | 0.9697 | 0.9700 |

### 4.2 Detection Performance
| Model | Avg. IoU | Miss Rate (%) | Center Drift (px) | Std. Confidence |
|-------|----------|---------------|-------------------|-----------------|
| YOLOv5 (baseline) | 0.6818 | 18.89 | 33.91 | 0.1555 |
| YOLOv5 (anti-aliased) | 0.8758 | 2.22 | 8.79 | 0.0868 |
| YOLOv5 (TIPS) | 0.9998 | 0.00 | 0.0224 | 0.0029 |

## 5. Visualization Results

The following visualizations were created to demonstrate the effects of sub-pixel shifts. These visualizations can be used in different sections of the thesis as specified below.

### 5.1 Classification Visualizations

#### 5.1.1 Heatmaps of Cosine Similarity
These heatmaps show the cosine similarity between feature vectors of each frame and the first frame:

- `figures/classification/heatmap_VGG16_seq_0.png` - VGG16 on sequence 0
- `figures/classification/heatmap_AA-VGG16_seq_0.png` - Anti-aliased VGG16 on sequence 0
- `figures/classification/heatmap_TIPS-VGG16_seq_0.png` - TIPS-augmented VGG16 on sequence 0
- `figures/classification/heatmap_ResNet50_seq_0.png` - ResNet50 on sequence 0
- `figures/classification/heatmap_AA-ResNet50_seq_0.png` - Anti-aliased ResNet50 on sequence 0
- `figures/classification/heatmap_TIPS-ResNet50_seq_0.png` - TIPS-augmented ResNet50 on sequence 0

Similar heatmaps are available for sequences 1 and 2.

**Thesis section**: Use in **Section 9.1 (Classifier results)** to show feature stability differences between models.

#### 5.1.2 Line Plots of Feature Stability
These plots show the cosine similarity across frames for all models:

- `figures/classification/cosine_similarity_comparison_seq_0.png` - All models on sequence 0
- `figures/classification/cosine_similarity_comparison_seq_1.png` - All models on sequence 1
- `figures/classification/cosine_similarity_comparison_seq_2.png` - All models on sequence 2

**Thesis section**: Use in **Section 9.1 (Classifier results)** and **Section 8.2 (Experiments with static shift)**.

#### 5.1.3 Line Plots of Confidence Drift
These plots show the confidence drift across frames for all models:

- `figures/classification/confidence_drift_comparison_seq_0.png` - All models on sequence 0
- `figures/classification/confidence_drift_comparison_seq_1.png` - All models on sequence 1
- `figures/classification/confidence_drift_comparison_seq_2.png` - All models on sequence 2

**Thesis section**: Use in **Section 8.2 (Experiments with static shift)** and **Section 9.1 (Classifier results)**.

#### 5.1.4 Bar Plot of Model Comparison
This plot compares the average cosine similarity across all models:

- `figures/comparison/model_comparison_cosine_similarity.png`

**Thesis section**: Use in **Section 9.3 (Comparative analysis)** to highlight the overall performance differences.

### 5.2 Detection Visualizations

#### 5.2.1 Box Plots of Detection Metrics
These box plots show the distribution of different metrics for all detection models:

- `figures/detection/boxplot_iou.png` - IoU stability across models and sequences
- `figures/detection/boxplot_center_shift.png` - Center drift across models and sequences
- `figures/detection/boxplot_confidence.png` - Confidence stability across models and sequences

**Thesis section**: Use in **Section 9.2 (Object detection results)** to demonstrate statistical distributions of metrics.

#### 5.2.2 Line Plots of Center Drift
These plots show the center drift across frames for all detection models:

- `figures/detection/center_drift_comparison_seq_0.png` - All models on sequence 0
- `figures/detection/center_drift_comparison_seq_1.png` - All models on sequence 1
- `figures/detection/center_drift_comparison_seq_2.png` - All models on sequence 2

**Thesis section**: Use in **Section 9.2 (Object detection results)** and **Section 8.3 (Dynamic sequences)**.

#### 5.2.3 Bounding Box Animations
These GIF animations show the bounding box predictions for different models:

- `figures/detection/bbox_animation_baseline_seq_0.gif` - Baseline YOLOv5 on sequence 0
- `figures/detection/bbox_animation_yolo_seq_0.gif` - Anti-aliased YOLOv5 on sequence 0
- `figures/detection/bbox_animation_tips_seq_0.gif` - TIPS-augmented YOLOv5 on sequence 0

Similar animations are available for sequences 1 and 2.

**Thesis section**: Use in **Section 8.3 (Dynamic sequences)** and possibly include shortened versions in **Section 9.2 (Object detection results)**.

### 5.3 Comparison Visualizations

#### 5.3.1 Radar Chart of Model Performance
This radar chart compares the performance of different model types across multiple metrics:

- `figures/comparison/model_performance_radar.png`

**Thesis section**: Use in **Section 9.3 (Comparative analysis)** and **Section 10 (Conclusion)** to provide a comprehensive visual summary.

## 6. Conclusions

The experiments clearly demonstrate that standard CNN architectures exhibit significant instability in both feature representations and predictions when objects move by small amounts (sub-pixel shifts). Anti-aliasing techniques significantly improve stability, with TIPS showing the best performance overall.

For object detection, the impact is particularly pronounced, with YOLOv5 benefiting greatly from anti-aliasing and TIPS modifications, making it much more reliable for applications where precise localization is critical.

## 7. Creating Visualizations

To generate all visualizations used in the thesis, use the following scripts:

1. First, create sequences with the frame generation script:
```bash
python scripts/create_frames.py
```

2. Create ground truth annotations:
```bash
python scripts/create_gt.py
```

3. Run the main visualization script:
```bash
python scripts/create_visualizations.py
```

This will create all the figures listed above in the appropriate directories.

## 8. Important Visualization Scripts

The following scripts are responsible for visualizations:

- `scripts/create_visualizations.py` - Main script generating all visualization types
- `scripts/create_frames.py` - Script for generating test frames with shifted objects
- `scripts/create_gt.py` - Script for generating ground truth annotations
- `scripts/visualization.py` - Core visualization utilities
- `scripts/visualization_classifiers.py` - Specialized visualizations for classifiers
- `scripts/visualization_yolo.py` - Specialized visualizations for YOLO detectors 