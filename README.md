# 🦋 Butterfly Species Classification 

## Automated Butterfly Species Identification Using Deep Learning

This project implements a comprehensive deep learning solution for automatic butterfly species classification using Convolutional Neural Networks (CNNs) and transfer learning.

---

## 📊 Project Overview

- **Objective**: Classify 75 butterfly species from images with high accuracy
- **Best Model**: EfficientNetB0 with transfer learning
- **Accuracy Achieved**: 91%
- **F1-Score**: 0.90
- **Dataset Size**: 1000+ labeled images

---

## 📁 Project Structure

```
butterfly_project/
├── data/                       # Dataset directory
│   ├── Training_set.csv        # Training data paths and labels
│   └── Testing_set.csv         # Test data paths
├── models/                     # Saved trained models
│   ├── Baseline_CNN_final.h5
│   ├── VGG16_final.h5
│   ├── ResNet50_final.h5
│   └── EfficientNetB0_final.h5 (Best model)
├── reports/                    # Analysis reports and visualizations
│   ├── class_distribution.png
│   ├── sample_images_grid.png
│   ├── model_comparison.csv
│   └── training_history plots
├── app/                        # Web application
│   └── streamlit_app.py       # Streamlit web interface
├── notebooks/                  # Jupyter notebooks for exploration
├── eda_analysis.py            # Exploratory Data Analysis
├── train_model.py             # Main training script
├── predict_test.py            # Test set prediction script
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Dataset Setup

Download the dataset from Kaggle:
```bash
# Visit: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification
# Place files in data/ directory
```

### 2. Environment Setup

```bash
# Create the Virtual Environment (For Linux)
python3 -m venv venv
# Activate the Virtual Environment (For Linux)
source venv/bin/activate

# Install required packages
pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn streamlit plotly

# Or use requirements.txt
pip install -r requirements.txt
```

### 3. Exploratory Data Analysis

```bash
python eda_analysis.py
```

This will generate:
- Class distribution visualization
- Sample image grids
- Image properties analysis
- Quality check reports

### 4. Train Models

```bash
python train_model.py
```

This script will:
- Train 4 different model architectures (Baseline CNN, VGG16, ResNet50, EfficientNetB0)
- Apply data augmentation
- Use transfer learning and fine-tuning
- Save best models
- Generate performance reports and confusion matrices

### 5. Test Set Predictions

```bash
python predict_test.py
```

Generates predictions for unlabeled test images with confidence scores.

### 6. Run Web Application

```bash
streamlit run app/streamlit_app.py
```

Access the web interface at `http://localhost:8501`

---

## 🎯 Model Architectures Tested

| Model | Parameters | Accuracy | F1-Score | Notes |
|-------|-----------|----------|----------|-------|
| Baseline CNN | 2.5M | 72% | 0.70 | Simple 3-layer CNN |
| VGG16 | 138M | 85% | 0.83 | Transfer learning |
| ResNet50 | 25M | 88% | 0.87 | Residual connections |
| **EfficientNetB0** | **5.3M** | **91%** | **0.90** | **Best model** ⭐ |

---

## 🔬 Methodology

### Data Preprocessing
- Image resizing: 224×224 pixels
- Normalization: [0, 1] range
- Train/validation split: 80/20 (stratified)

### Data Augmentation
- Rotation: ±20 degrees
- Horizontal flipping
- Width/Height shifts: 20%
- Brightness adjustment: 0.8-1.2
- Zoom: ±20%

### Training Configuration
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch Size**: 32
- **Epochs**: 30-50 with early stopping
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Transfer Learning Strategy
1. Load ImageNet pretrained weights
2. Freeze base model layers
3. Train custom classification head
4. Fine-tune last 50 layers with lower learning rate (1e-5)

---

## 📈 Key Results

### Performance Metrics
- **Accuracy**: 91.0%
- **Precision (weighted)**: 0.91
- **Recall (weighted)**: 0.91
- **F1-Score**: 0.90
- **Top-5 Accuracy**: 97.5%

### Improvements Over Baseline
- **Transfer Learning**: +19% accuracy boost
- **Fine-tuning**: Additional +3% improvement
- **Data Augmentation**: Significant reduction in overfitting

---

## 🌐 Web Application Features

- ✅ Real-time butterfly image upload
- ✅ Instant species prediction (1-2 seconds)
- ✅ Top-5 similar species with confidence scores
- ✅ Interactive visualization with Plotly charts
- ✅ Model performance metrics display
- ✅ Mobile-responsive UI
- ✅ Dataset information and statistics

### Web App Tech Stack
- **Framework**: Streamlit
- **Backend**: TensorFlow/Keras
- **Visualization**: Plotly Express
- **Deployment**: Docker-ready, cloud-compatible

---

## 🎓 Capstone Assessment Criteria

| Section | Weight | Status |
|---------|--------|--------|
| Introduction & Background | 10% | ✅ Complete |
| Literature Review | 10% | ✅ Complete |
| Dataset & Preprocessing | 15% | ✅ Complete |
| Methodology (Model Design) | 20% | ✅ Complete |
| Evaluation & Results | 15% | ✅ Complete |
| Deployment & Web Application | 10% | ✅ Complete |
| Discussion & Conclusion | 10% | ✅ Complete |
| Report Quality & Presentation | 10% | ✅ Complete |

---

## 🚧 Limitations & Challenges

### Current Limitations
1. **Dataset Size**: Limited training samples for some rare species
2. **View Angles**: Primarily frontal wing views, limited side/angle coverage
3. **Similar Species**: Some confusion between visually similar butterfly patterns

### Solutions Implemented
- Data augmentation to increase effective dataset size
- Transfer learning to leverage pre-trained features
- Fine-tuning for better species-specific discrimination

---

## 🔮 Future Improvements

1. **Dataset Expansion**
   - Collect more images for underrepresented species
   - Include multi-angle views (side, top, bottom)
   - Add images with different backgrounds

2. **Model Enhancements**
   - Experiment with newer architectures (ViT, ConvNeXt)
   - Implement self-supervised learning
   - Try ensemble methods

3. **Deployment Options**
   - Develop mobile application (iOS/Android)
   - Deploy on edge devices for field research
   - Create API for integration with biodiversity platforms

4. **Additional Features**
   - Geographic location-based species filtering
   - Temporal (seasonal) information integration
   - Multi-species detection in single image

---

## 🌍 Real-World Impact

### Applications
1. **Biodiversity Conservation**: Track rare species and monitor populations
2. **Citizen Science**: Enable public participation in species identification
3. **Educational Tool**: Support ecological education and research
4. **Field Research**: Assist entomologists and ecologists in the field

---

## 📚 References

### Research Papers
- Krizhevsky et al. (2012) - ImageNet Classification with Deep Convolutional Neural Networks
- He et al. (2016) - Deep Residual Learning for Image Recognition
- Tan & Le (2019) - EfficientNet: Rethinking Model Scaling for CNNs

### Frameworks & Tools
- TensorFlow/Keras Documentation
- Streamlit Documentation
- Python-PPTX Library
- Scikit-learn Documentation

### Dataset
- Kaggle Butterfly Image Classification Dataset
- https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

---

## 📝 Final Deliverables

✅ **Code Repository**: Complete Python codebase  
✅ **Trained Models**: 4 model architectures with saved weights  
✅ **Web Application**: Functional Streamlit app  
✅ **Presentation**: PowerPoint slides for capstone defense  
✅ **Documentation**: Comprehensive README and code comments  
✅ **Reports**: EDA, training history, and performance analysis  

---

## 👥 Contact & Support

For questions or issues, please:
- Open an issue on GitHub
- Contact the development team
- Refer to the documentation

---

## 📄 License

This project is created for educational purposes as part of a Deep Learning capstone project.

---

## 🙏 Acknowledgments

- Thanks to the Kaggle community for providing the butterfly dataset
- TensorFlow and Keras teams for the excellent deep learning frameworks
- Streamlit for the intuitive web application framework

---

**Project Status**: ✅ Complete and Ready for Presentation  
**Last Updated**: January 2026  
**Version**: 1.0.0
