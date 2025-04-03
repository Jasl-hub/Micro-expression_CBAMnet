# Micro-expression_CBAMnet
# Micro-Expression Recognition Using Attention-Driven Deep Learning

## Overview
This project focuses on micro-expression recognition using a combination of Convolutional Neural Networks (CNN) and Convolutional Block Attention Module (CBAM). The model is trained on the **SAMM dataset** to capture subtle facial expressions, significantly improving classification accuracy.

## Features
- **Attention Mechanism:** Integrated CBAM with CNN to enhance feature extraction and improve recognition accuracy.
- **Model Performance:** Achieved a **14% accuracy improvement**, increasing from **80% (CNN) to 94% (CNN+CBAM)**.
- **Explainability:** Used **Grad-CAM** to interpret model decisions and visualize important features.
- **Optimized Training:** Trained on **256×256** images, with **8.41M parameters** over **15 epochs**.
- **Micro-Expression Classification:** Successfully classified **six micro-expressions**, contributing to advancements in facial emotion analysis.

## Dataset
- **Dataset Name:** SAMM (Spontaneous Micro-Expression Database)
- **Preprocessing:** Images resized to **256×256** for uniform input size.
- **Augmentation:** Applied transformations for better generalization.

## Model Architecture
- **Baseline Model:** CNN-based classifier.
- **Enhanced Model:** CNN + CBAM (Convolutional Block Attention Module) to refine feature extraction.
- **Loss Function:** Cross-entropy loss for multi-class classification.
- **Optimizer:** Adam optimizer for efficient convergence.

## Installation & Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy matplotlib opencv-python
```
### Running the Model
```python
python train.py --dataset_path <path_to_SAMM_dataset>
```

## Challenges & Solutions
- **Subtle Feature Detection:** Micro-expressions last for a very short duration, making them hard to detect. **Solution:** Used attention mechanisms (CBAM) to focus on critical facial regions.
- **Limited Dataset Size:** Small dataset size can lead to overfitting. **Solution:** Applied **data augmentation** techniques.
- **Explainability:** Understanding why the model made specific predictions. **Solution:** Integrated **Grad-CAM** for feature visualization.

## Results & Impact
- Achieved **94% accuracy** with CNN + CBAM model.
- Improved interpretability using **Grad-CAM**.
- Potential real-world applications in **lie detection, mental health monitoring, and human-computer interaction**.

## Future Enhancements
- Extend to **multi-dataset training** for robustness.
- Experiment with **Vision Transformers (ViTs)** for further performance improvements.
- Deploy as an **API** for real-time micro-expression detection.


## Contact
For any questions, feel free to reach out!
🚀 LinkedIn: [Jasleen Kaur](www.linkedin.com/in/jas03leen)

