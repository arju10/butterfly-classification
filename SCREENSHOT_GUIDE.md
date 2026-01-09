# 📸 Screenshot & Image Guide

Complete guide for capturing and organizing project screenshots.

---

## 📁 Required Images Directory Structure

Create this folder structure in your project:

```
butterfly_classifier/
├── images/
│   ├── interface/              # App interface screenshots
│   ├── results/                # Prediction results
│   ├── training/               # Training process & metrics
│   ├── examples/               # Example predictions
│   ├── architecture/           # Diagrams & flowcharts
│   └── gallery/                # Complete user journey
└── README.md
```

---

## 📸 Screenshots to Take

### 1. Application Interface (8 screenshots)

#### `interface_main.png`
**What to show:**
- Main dashboard on first load
- Empty upload area
- Sidebar visible
- Clean, professional appearance

**How to capture:**
1. Start app: `streamlit run streamlit_app.py`
2. Wait for "Model loaded successfully"
3. Take full window screenshot
4. Save as `images/interface_main.png`

---

#### `upload_screen.png`
**What to show:**
- Upload area highlighted
- "Browse files" button visible
- Supported formats listed

**How to capture:**
1. Focus on upload section
2. Take screenshot
3. Save as `images/upload_screen.png`

---

#### `prediction_result.png`
**What to show:**
- Complete prediction with all elements
- Species name clearly visible
- Confidence score displayed
- Gauge and chart visible

**How to capture:**
1. Upload a butterfly image
2. Click "Identify Species"
3. Wait for results
4. Take full window screenshot
5. Save as `images/prediction_result.png`

---

#### `top5_predictions.png`
**What to show:**
- Horizontal bar chart
- All 5 species names
- Confidence percentages
- Color-coded bars

**How to capture:**
1. Scroll to Top-5 section
2. Capture chart area
3. Save as `images/top5_predictions.png`

---

#### `confidence_gauge.png`
**What to show:**
- Circular gauge meter
- Confidence percentage (large number)
- Color zones (green/yellow/red)
- Clean, focused shot

**How to capture:**
1. Focus on gauge area
2. Capture just the gauge
3. Save as `images/confidence_gauge.png`

---

#### `sidebar.png`
**What to show:**
- About section
- Model info
- Tips section
- Complete sidebar content

**How to capture:**
1. Open sidebar (if collapsed)
2. Capture sidebar area
3. Save as `images/sidebar.png`

---

#### `processing.png`
**What to show:**
- "Analyzing..." spinner
- Image uploaded
- Processing indicator

**How to capture:**
1. Upload image
2. Quickly click "Identify"
3. Take screenshot during processing
4. Save as `images/processing.png`

---

#### `responsive_design.png`
**What to show:**
- Multiple device views (optional)
- Desktop + tablet + mobile

**How to capture:**
1. Use browser dev tools (F12)
2. Toggle device toolbar
3. Switch between device sizes
4. Take screenshots
5. Combine in image editor
6. Save as `images/responsive_design.png`

---

### 2. Prediction Examples (6 screenshots)

#### `example_high_confidence.png`
**What to show:**
- Prediction with >70% confidence
- Species: MONARCH or similar common species
- Green confidence indicator
- Clear butterfly image

**How to capture:**
1. Upload clear MONARCH image
2. Get prediction
3. Show confidence >70%
4. Save as `images/example_high_confidence.png`

---

#### `example_medium_confidence.png`
**What to show:**
- Prediction with 40-70% confidence
- Yellow confidence indicator
- Show top alternatives

**How to capture:**
1. Upload less common species
2. Get medium confidence prediction
3. Save as `images/example_medium_confidence.png`

---

#### `example_low_confidence.png`
**What to show:**
- Prediction with <40% confidence
- Red confidence indicator
- Multiple similar alternatives

**How to capture:**
1. Upload difficult/blurry image
2. Get low confidence prediction
3. Save as `images/example_low_confidence.png`

---

### 3. Training Results (8 screenshots)

#### `training_accuracy.png`
**What to create:**
- Line graph: Training vs Validation accuracy
- X-axis: Epochs (0-30)
- Y-axis: Accuracy (0-100%)
- Show convergence

**How to create:**
```python
import matplotlib.pyplot as plt

# Your training history data
epochs = range(1, 31)
train_acc = [...]  # Your data
val_acc = [...]    # Your data

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/training_accuracy.png', dpi=150)
plt.close()
```

---

#### `confusion_matrix.png`
**What to create:**
- Heatmap of 75x75 confusion matrix
- Color-coded (diagonal should be bright)
- Species labels (can be small)

**How to create:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Your confusion matrix data
cm = ...  # 75x75 array

plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=species_names,
            yticklabels=species_names)
plt.title('Confusion Matrix - 75 Butterfly Species', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150)
plt.close()
```

---

#### `confidence_distribution.png`
**What to create:**
- Histogram of prediction confidence
- Show distribution across validation set
- Mark thresholds (40%, 70%)

**How to create:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Your confidence scores
confidences = [...]  # Array of confidence values

plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=0.4, color='red', linestyle='--', label='Low threshold (40%)')
plt.axvline(x=0.7, color='green', linestyle='--', label='High threshold (70%)')
plt.title('Confidence Score Distribution', fontsize=16)
plt.xlabel('Confidence', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/confidence_distribution.png', dpi=150)
plt.close()
```

---

#### `model_comparison.png`
**What to create:**
- Bar chart comparing 4 models
- Metrics: Accuracy, Size, Speed
- Already have model_comparison.png!

**How to use:**
```bash
# You already have this from Kaggle!
cp model_comparison.png images/model_comparison.png
```

---

#### `learning_curves.png`
**What to create:**
- Combined plot: Loss and Accuracy
- 2 subplots side by side

**How to create:**
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
ax1.set_title('Model Loss', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
ax2.set_title('Model Accuracy', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/learning_curves.png', dpi=150)
plt.close()
```

---

### 4. Architecture Diagrams (3 diagrams)

#### `model_architecture.png`
**What to create:**
- Visual flowchart of model layers
- Input → MobileNetV2 → Custom Head → Output

**How to create:**
Use draw.io, Lucidchart, or PowerPoint:

```
┌─────────────────────┐
│   Input Image       │
│   (224x224x3)       │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   MobileNetV2 Base  │
│   (ImageNet weights)│
│   (Frozen)          │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ GlobalAvgPooling2D  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ BatchNormalization  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Dense(512, relu)  │
│   + Dropout(0.5)    │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ BatchNormalization  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Dense(256, relu)  │
│   + Dropout(0.3)    │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Dense(75, softmax) │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Predictions       │
│   (75 classes)      │
└─────────────────────┘
```

---

#### `training_process.png`
**What to create:**
- Flowchart of training phases
- Phase 1 → Phase 2 → Evaluation

---

#### `tech_stack.png`
**What to create:**
- Visual of technology stack
- Frontend + Backend + Deployment layers

---

### 5. Dataset Images (3 images)

#### `dataset_overview.png`
**What to create:**
- Pie chart of species distribution
- Bar chart of images per species

#### `sample_dataset.png`
**What to create:**
- Grid of 12-16 sample butterfly images
- Show variety in species/poses

#### `species_distribution.png`
**What to create:**
- Bar chart showing family distribution

---

### 6. Gallery - Complete User Journey (8 screenshots)

Take sequential screenshots showing full flow:

1. `gallery_01_landing.png` - First view
2. `gallery_02_upload.png` - Clicking upload
3. `gallery_03_preview.png` - Image preview
4. `gallery_04_processing.png` - Processing spinner
5. `gallery_05_results.png` - Results appear
6. `gallery_06_gauge.png` - Confidence gauge
7. `gallery_07_top5.png` - Top-5 chart
8. `gallery_08_guide.png` - Interpretation guide

---

## 🛠️ Tools for Screenshots

### Windows
- **Snipping Tool** (built-in)
- **Snip & Sketch** (Win + Shift + S)
- **ShareX** (free, advanced)

### macOS
- **Screenshot** (Cmd + Shift + 4)
- **Preview** (for editing)
- **CleanShot X** (paid, professional)

### Linux
- **Flameshot** (recommended)
- **GNOME Screenshot**
- **Spectacle** (KDE)

### Browser Tools
- **Firefox Screenshot** (right-click)
- **Chrome DevTools** (F12 → device toolbar)
- **Full Page Screen Capture** (extension)

---

## ✂️ Image Editing Tips

### Cropping
- Remove unnecessary whitespace
- Focus on relevant content
- Keep consistent aspect ratios

### Annotations (Optional)
- Add arrows to highlight features
- Add text labels for clarity
- Use consistent colors (red for important)

### Optimization
```bash
# Resize images to reasonable size
mogrify -resize 1920x1080 images/*.png

# Compress PNGs
optipng images/*.png

# Or use online tools:
# - TinyPNG.com
# - Squoosh.app
```

---

## 📏 Image Specifications

### Recommended Sizes

| Image Type | Dimensions | Format |
|------------|-----------|--------|
| Interface screenshots | 1920x1080 | PNG |
| Charts/graphs | 1200x800 | PNG |
| Diagrams | 1000x800 | PNG |
| Gallery images | 1600x1000 | PNG |
| Examples | 800x600 | PNG/JPG |

### File Size Guidelines
- Keep individual images < 1 MB
- Use PNG for screenshots (lossless)
- Use JPG for photos (smaller size)
- Compress without losing quality

---

## ✅ Screenshot Checklist

### Before Capturing
- [ ] App is running smoothly
- [ ] Window is maximized/clean
- [ ] No personal information visible
- [ ] Good lighting (for camera shots)
- [ ] Clear, focused content

### After Capturing
- [ ] Image is clear and readable
- [ ] Correct filename
- [ ] Proper directory
- [ ] Reasonable file size
- [ ] No sensitive data

---

## 🎨 Creating Diagrams

### Tools

**Free:**
- **draw.io** (diagrams.net) - Best for flowcharts
- **Excalidraw** - Hand-drawn style
- **Google Drawings** - Simple diagrams
- **Microsoft PowerPoint** - Widely available

**Paid:**
- **Lucidchart** - Professional diagrams
- **Adobe Illustrator** - Vector graphics
- **Sketch/Figma** - UI/UX design

### Templates

Use these for architecture diagrams:
- AWS Architecture Icons (if deploying to AWS)
- Google Cloud Icons (if deploying to GCP)
- Standard flowchart shapes
- Neural network diagram templates

---

## 📊 Creating Charts

### Using Python

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 12

# Your plotting code...

# Save with high quality
plt.savefig('images/chart.png', 
            dpi=150, 
            bbox_inches='tight',
            facecolor='white')
plt.close()
```

### Using Excel/Google Sheets
1. Create chart
2. Right-click → Save as image
3. Choose high resolution

---

## 🎯 Priority List

### Must-Have (Essential)
1. ✅ `interface_main.png`
2. ✅ `prediction_result.png`
3. ✅ `confidence_gauge.png`
4. ✅ `top5_predictions.png`
5. ✅ `example_high_confidence.png`

### Should-Have (Important)
6. ✅ `model_comparison.png` (you have this!)
7. ✅ `training_accuracy.png`
8. ✅ `confusion_matrix.png`
9. ✅ `example_medium_confidence.png`
10. ✅ `model_architecture.png`

### Nice-to-Have (Optional)
11. Gallery images (8 screenshots)
12. Error analysis charts
13. Dataset visualizations
14. Responsive design demos

---

## 🚀 Quick Start

### Minimum Viable Documentation (15 minutes)

```bash
# 1. Create images folder
mkdir -p images

# 2. Run app
streamlit run streamlit_app.py

# 3. Take 5 essential screenshots:
#    - Main interface
#    - One high confidence prediction
#    - Confidence gauge close-up
#    - Top-5 chart
#    - Full results page

# 4. Copy existing model comparison
cp model_comparison.png images/

# 5. Update README.md with image paths

# Done! You have visual documentation
```

---

## 📝 Image Organization

### Naming Convention

```
images/
├── interface_*.png          # UI screenshots
├── example_*.png           # Prediction examples
├── training_*.png          # Training metrics
├── gallery_0*_*.png        # User journey (numbered)
├── model_*.png             # Architecture & comparisons
└── use_case_*.png          # Application examples
```

### Index File

Create `images/README.md`:

```markdown
# Images Index

## Interface Screenshots
- interface_main.png - Main dashboard
- prediction_result.png - Complete prediction display
- confidence_gauge.png - Gauge visualization
...

## Training Results
- training_accuracy.png - Accuracy curves
- confusion_matrix.png - Confusion matrix heatmap
...
```

---

## 🎬 Creating GIFs (Optional)

### Show Interactive Features

**Tools:**
- **ScreenToGif** (Windows)
- **LICEcap** (Mac/Windows)
- **Peek** (Linux)

**What to record:**
- Upload → Predict → Results (5-10 seconds)
- Scrolling through results
- Interacting with charts

**Save as:**
- `demo_upload_predict.gif`
- `demo_interaction.gif`

---

## ✅ Final Check

### Before Adding to README

- [ ] All images load correctly
- [ ] File names match README references
- [ ] Images are optimized (< 1 MB each)
- [ ] Total images folder < 20 MB
- [ ] All paths are relative (`images/filename.png`)
- [ ] Images render properly in Markdown preview

---

## 🎯 Result

After following this guide, you'll have:

✅ Professional visual documentation  
✅ Clear demonstration of features  
✅ Impressive README with images  
✅ Complete project showcase  
✅ Portfolio-ready presentation  

**Your README will look AMAZING!** 📸✨

---

**Now go capture those screenshots!** 📷🦋
