# """
# 🦋 Butterfly Species Classifier - Streamlit Web App
# Production-ready web interface for butterfly identification

# Features:
# - Upload butterfly images OR use built-in sample images
# - Get instant predictions
# - View top-5 most likely species
# - Confidence visualization
# - Beautiful, user-friendly interface
# """

# import streamlit as st
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import json
# import os
# import plotly.graph_objects as go
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="🦋 Butterfly Classifier",
#     page_icon="🦋",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         color: #10b981;
#         margin-bottom: 0.5rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         text-align: center;
#         color: #6b7280;
#         margin-bottom: 2rem;
#     }
#     .prediction-card {
#         background-color: #f0fdf4;
#         padding: 1.5rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #10b981;
#         margin: 1rem 0;
#     }
#     .confidence-high {
#         color: #10b981;
#         font-weight: bold;
#     }
#     .confidence-medium {
#         color: #f59e0b;
#         font-weight: bold;
#     }
#     .confidence-low {
#         color: #ef4444;
#         font-weight: bold;
#     }
#     .stButton>button {
#         width: 100%;
#         background-color: #10b981;
#         color: white;
#         font-weight: bold;
#         padding: 0.75rem;
#         border-radius: 0.5rem;
#         border: none;
#         font-size: 1.1rem;
#     }
#     .stButton>button:hover {
#         background-color: #059669;
#     }
# </style>
# """, unsafe_allow_html=True)


# def create_butterfly_pattern(size=400, color=(255, 140, 0), pattern_type='monarch'):
#     """
#     Create a butterfly-like pattern image
#     Different patterns for different "species"
#     """
#     img = Image.new('RGB', (size, size), color='white')
#     draw = ImageDraw.Draw(img)
    
#     center_x, center_y = size // 2, size // 2
    
#     if pattern_type == 'monarch':
#         # Orange with black veins
#         # Background orange
#         draw.rectangle([0, 0, size, size], fill=(255, 140, 0))
#         # Black borders
#         draw.rectangle([0, 0, size, 20], fill=(20, 20, 20))
#         draw.rectangle([0, size-20, size, size], fill=(20, 20, 20))
#         draw.rectangle([0, 0, 20, size], fill=(20, 20, 20))
#         draw.rectangle([size-20, 0, size, size], fill=(20, 20, 20))
#         # Veins
#         for i in range(0, size, 40):
#             draw.line([i, 0, i, size], fill=(20, 20, 20), width=3)
#             draw.line([0, i, size, i], fill=(20, 20, 20), width=3)
#         # White spots
#         for i in range(50, size-50, 60):
#             for j in range(50, size-50, 60):
#                 draw.ellipse([i-8, j-8, i+8, j+8], fill=(255, 255, 255))
    
#     elif pattern_type == 'blue':
#         # Blue morpho - bright blue with black edges
#         draw.rectangle([0, 0, size, size], fill=(30, 144, 255))
#         # Dark edges
#         draw.ellipse([50, 50, size-50, size-50], fill=(30, 144, 255))
#         draw.rectangle([0, 0, size, 60], fill=(20, 20, 40))
#         draw.rectangle([0, size-60, size, size], fill=(20, 20, 40))
#         # Shimmer effect
#         for i in range(0, size, 30):
#             alpha = int(255 * (1 - i/size))
#             lighter = (60, 180, 255)
#             draw.ellipse([i, i, size-i, size-i], outline=lighter, width=2)
    
#     elif pattern_type == 'pink':
#         # Painted lady - pink/orange with patterns
#         draw.rectangle([0, 0, size, size], fill=(255, 192, 203))
#         # Brown patterns
#         for i in range(0, size, 50):
#             draw.polygon([
#                 (i, 0), (i+25, 50), (i+50, 0)
#             ], fill=(139, 69, 19))
#             draw.polygon([
#                 (i, size), (i+25, size-50), (i+50, size)
#             ], fill=(139, 69, 19))
#         # White spots
#         for i in range(80, size-80, 100):
#             draw.ellipse([i-15, center_y-15, i+15, center_y+15], fill=(255, 255, 255))
    
#     elif pattern_type == 'red':
#         # Red admiral - black with red bands
#         draw.rectangle([0, 0, size, size], fill=(20, 20, 20))
#         # Red bands
#         band_width = 60
#         draw.rectangle([0, center_y-band_width, size, center_y+band_width], fill=(220, 20, 60))
#         draw.rectangle([center_x-band_width, 0, center_x+band_width, size], fill=(220, 20, 60))
#         # White spots on tips
#         for i in range(50, size-50, 80):
#             draw.ellipse([i-10, 50-10, i+10, 50+10], fill=(255, 255, 255))
#             draw.ellipse([i-10, size-50-10, i+10, size-50+10], fill=(255, 255, 255))
    
#     elif pattern_type == 'purple':
#         # Purple emperor
#         draw.rectangle([0, 0, size, size], fill=(148, 0, 211))
#         # Darker edges
#         draw.rectangle([0, 0, size, 30], fill=(75, 0, 130))
#         draw.rectangle([0, size-30, size, size], fill=(75, 0, 130))
#         draw.rectangle([0, 0, 30, size], fill=(75, 0, 130))
#         draw.rectangle([size-30, 0, size, size], fill=(75, 0, 130))
#         # Lighter center
#         draw.ellipse([100, 100, size-100, size-100], fill=(186, 85, 211))
#         # Eye spots
#         for x, y in [(120, 120), (size-120, 120), (120, size-120), (size-120, size-120)]:
#             draw.ellipse([x-25, y-25, x+25, y+25], fill=(255, 255, 0))
#             draw.ellipse([x-15, y-15, x+15, y+15], fill=(0, 0, 0))
#             draw.ellipse([x-5, y-5, x+5, y+5], fill=(255, 255, 255))
    
#     return img


# def create_sample_images():
#     """
#     Create sample butterfly images with patterns
#     """
#     sample_images = {}
    
#     patterns = [
#         ('monarch', "Monarch Butterfly"),
#         ('blue', "Blue Morpho"),
#         ('pink', "Painted Lady"),
#         ('red', "Red Admiral"),
#         ('purple', "Purple Emperor")
#     ]
    
#     for pattern_type, name in patterns:
#         try:
#             # Get color based on pattern
#             colors = {
#                 'monarch': (255, 140, 0),
#                 'blue': (30, 144, 255),
#                 'pink': (255, 192, 203),
#                 'red': (220, 20, 60),
#                 'purple': (148, 0, 211)
#             }
#             color = colors.get(pattern_type, (255, 140, 0))
            
#             # Create patterned image
#             img = create_butterfly_pattern(400, color, pattern_type)
#             sample_images[name] = img
            
#         except Exception as e:
#             st.warning(f"Could not create sample {name}: {e}")
    
#     return sample_images


# def load_sample_images_from_folder():
#     """
#     Load sample images from samples/ folder if it exists
#     Otherwise, create patterned samples
#     """
#     sample_folder = "samples"
#     sample_images = {}
    
#     # Check if samples folder exists
#     if os.path.exists(sample_folder) and os.path.isdir(sample_folder):
#         # Load images from folder
#         image_files = [f for f in os.listdir(sample_folder) 
#                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
#         if image_files:
#             for filename in image_files:
#                 filepath = os.path.join(sample_folder, filename)
#                 try:
#                     img = Image.open(filepath).convert('RGB')
#                     # Use filename without extension as name
#                     name = os.path.splitext(filename)[0].replace('_', ' ').title()
#                     sample_images[name] = img
#                 except Exception as e:
#                     st.warning(f"Could not load {filename}: {e}")
    
#     # If no images found, create patterned samples
#     if not sample_images:
#         sample_images = create_sample_images()
#         st.info("💡 **Tip:** Add real butterfly images to `samples/` folder for better demo!")
    
#     return sample_images


# @st.cache_resource(show_spinner=False)
# def load_model_and_classes():
#     """Load the trained model and class indices with caching (silent)"""
#     try:
#         # Model path - using .keras format
#         model_path = 'models/butterfly_model_WORKING.keras'
        
#         # Check if model exists
#         if not os.path.exists(model_path):
#             return None, None, None, "Model file not found"
        
#         # Load model silently
#         model = None
#         try:
#             # Load with compile=False for faster loading
#             model = keras.models.load_model(model_path, compile=False)
            
#             # Compile for predictions
#             model.compile(
#                 optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy']
#             )
            
#         except Exception as e:
#             return None, None, None, f"Model loading error: {str(e)}"
        
#         # Load class indices
#         class_indices_path = 'class_indices.json'
#         if not os.path.exists(class_indices_path):
#             return None, None, None, "class_indices.json not found"
            
#         with open(class_indices_path, 'r') as f:
#             class_indices = json.load(f)
        
#         # Create reverse mapping (index -> class name)
#         idx_to_class = {v: k for k, v in class_indices.items()}
        
#         return model, class_indices, idx_to_class, None
        
#     except Exception as e:
#         return None, None, None, f"Unexpected error: {str(e)}"


# @st.cache_data(show_spinner=False)
# def get_sample_images():
#     """Cache sample images so they only load once"""
#     return load_sample_images_from_folder()


# def preprocess_image(image, target_size=(224, 224)):
#     """Preprocess image for model prediction"""
#     # Resize image
#     image = image.resize(target_size)
    
#     # Convert to array and normalize to [0, 1]
#     img_array = np.array(image, dtype=np.float32) / 255.0
    
#     # Add batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
    
#     return img_array


# def get_confidence_color(confidence):
#     """Return CSS class based on confidence level"""
#     if confidence >= 0.7:
#         return "confidence-high"
#     elif confidence >= 0.4:
#         return "confidence-medium"
#     else:
#         return "confidence-low"


# def get_confidence_interpretation(confidence):
#     """Return human-readable confidence interpretation"""
#     if confidence >= 0.9:
#         return "Very High Confidence"
#     elif confidence >= 0.7:
#         return "High Confidence"
#     elif confidence >= 0.5:
#         return "Medium Confidence"
#     elif confidence >= 0.3:
#         return "Low Confidence"
#     else:
#         return "Very Low Confidence"


# def create_confidence_gauge(confidence, species_name):
#     """Create a beautiful confidence gauge using Plotly"""
#     # Determine color based on confidence
#     if confidence >= 0.7:
#         bar_color = "#10b981"  # Green
#     elif confidence >= 0.4:
#         bar_color = "#f59e0b"  # Yellow
#     else:
#         bar_color = "#ef4444"  # Red
    
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=confidence * 100,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': f"Confidence", 'font': {'size': 20}},
#         number={'suffix': "%", 'font': {'size': 40}},
#         gauge={
#             'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
#             'bar': {'color': bar_color, 'thickness': 0.75},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "gray",
#             'steps': [
#                 {'range': [0, 40], 'color': '#fee2e2'},
#                 {'range': [40, 70], 'color': '#fef3c7'},
#                 {'range': [70, 100], 'color': '#d1fae5'}
#             ],
#             'threshold': {
#                 'line': {'color': "red", 'width': 4},
#                 'thickness': 0.75,
#                 'value': 50
#             }
#         }
#     ))
    
#     fig.update_layout(
#         height=300,
#         margin=dict(l=20, r=20, t=60, b=20),
#         paper_bgcolor="rgba(0,0,0,0)",
#         font={'family': "Arial, sans-serif"}
#     )
    
#     return fig


# def create_top_predictions_chart(predictions, idx_to_class, top_k=5):
#     """Create horizontal bar chart for top predictions"""
#     # Get top k predictions
#     top_indices = np.argsort(predictions[0])[-top_k:][::-1]
#     top_species = [idx_to_class[i] for i in top_indices]
#     top_confidences = predictions[0][top_indices] * 100
    
#     # Create color scale based on confidence
#     colors = []
#     for c in top_confidences:
#         if c >= 70:
#             colors.append('#10b981')  # Green
#         elif c >= 40:
#             colors.append('#f59e0b')  # Yellow
#         else:
#             colors.append('#ef4444')  # Red
    
#     fig = go.Figure(go.Bar(
#         x=top_confidences,
#         y=top_species,
#         orientation='h',
#         marker=dict(color=colors),
#         text=[f'{c:.1f}%' for c in top_confidences],
#         textposition='auto',
#         textfont=dict(size=14, color='white', family='Arial Black')
#     ))
    
#     fig.update_layout(
#         title=f"Top {top_k} Most Likely Species",
#         xaxis_title="Confidence (%)",
#         yaxis_title="Species",
#         height=300,
#         margin=dict(l=20, r=20, t=60, b=20),
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font={'family': "Arial, sans-serif", 'size': 12},
#         xaxis=dict(gridcolor='lightgray', range=[0, 100]),
#         yaxis=dict(autorange="reversed")
#     )
    
#     return fig


# def main():
#     # Header
#     st.markdown('<p class="main-header">🦋 Butterfly Species Classifier</p>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Upload a butterfly image to identify its species using AI</p>', unsafe_allow_html=True)
    
#     # Load model silently
#     model, class_indices, idx_to_class, error = load_model_and_classes()
    
#     # Check if model loaded
#     if model is None:
#         st.error(f"❌ Failed to load model: {error}")
#         st.info("""
#         **Setup Instructions:**
#         1. Place `butterfly_model_best.keras` in `models/` directory
#         2. Place `class_indices.json` in the project root
#         3. Restart the Streamlit app
#         """)
#         st.stop()
    
#     # Load sample images
#     sample_images = get_sample_images()
    
#     # Sidebar
#     with st.sidebar:
#         st.header("ℹ️ About")
#         st.write(f"""
#         This AI-powered app can identify **{len(class_indices)} different butterfly species** with high accuracy!
        
#         **How to use:**
#         1. Upload your own image OR try sample images
#         2. Click 'Identify Species'
#         3. Get instant predictions!
        
#         **Best results:**
#         - Clear, well-lit photos
#         - Butterfly in focus
#         - Minimal background clutter
#         """)
        
#         st.divider()
        
#         st.header("📊 Model Info")
#         if os.path.exists('model_info.json'):
#             try:
#                 with open('model_info.json', 'r') as f:
#                     model_info = json.load(f)
#                 st.write(f"**Model:** {model_info.get('best_model', 'MobileNetV2')}")
#                 st.write(f"**Accuracy:** {model_info.get('best_model_metrics', {}).get('accuracy', 0.85)*100:.1f}%")
#                 st.write(f"**Species:** {model_info.get('num_classes', len(class_indices))}")
#             except:
#                 st.write(f"**Species:** {len(class_indices)}")
#         else:
#             st.write(f"**Architecture:** MobileNetV2")
#             st.write(f"**Species:** {len(class_indices)}")
#             st.write(f"**Accuracy:** 88.6%")
        
#         st.divider()
        
#         st.header("🎯 Confidence Guide")
#         st.write("""
#         - **High (>70%)** 🟢: Very reliable
#         - **Medium (40-70%)** 🟡: Generally good
#         - **Low (<40%)** 🔴: Needs verification
#         """)
        
#         st.divider()
        
#         st.header("👨‍💻 Developer")
#         st.write("**Created by:** Arju")
#         st.write("**Year:** 2026")
#         st.write("**Framework:** TensorFlow & Streamlit")
    
#     # Main content
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("📤 Upload Image")
        
#         # Tab for upload vs sample images
#         tab1, tab2 = st.tabs(["📁 Upload Your Image", "🖼️ Try Sample Images"])
        
#         image_to_predict = None
#         image_source = None
        
#         with tab1:
#             uploaded_file = st.file_uploader(
#                 "Choose a butterfly image...",
#                 type=['jpg', 'jpeg', 'png'],
#                 help="Upload a clear image of a butterfly"
#             )
            
#             if uploaded_file is not None:
#                 image_to_predict = Image.open(uploaded_file).convert('RGB')
#                 image_source = "uploaded"
#                 st.image(image_to_predict, caption='Your Uploaded Image', use_container_width=True)
#                 st.info(f"📐 Image size: {image_to_predict.size[0]} x {image_to_predict.size[1]} pixels")
        
#         with tab2:
#             st.write("**Click on any sample below to test:**")
            
#             # Display sample images in a grid
#             if len(sample_images) > 0:
#                 num_cols = 2
#                 sample_list = list(sample_images.items())
                
#                 for i in range(0, len(sample_list), num_cols):
#                     cols = st.columns(num_cols)
#                     for j in range(num_cols):
#                         idx = i + j
#                         if idx < len(sample_list):
#                             name, img = sample_list[idx]
#                             with cols[j]:
#                                 # Show thumbnail
#                                 st.image(img, caption=name, use_container_width=True)
#                                 if st.button(f"Use This", key=f"sample_{idx}", use_container_width=True):
#                                     st.session_state['sample_image'] = img
#                                     st.session_state['sample_name'] = name
#                                     st.rerun()
#             else:
#                 st.error("No sample images available")
            
#             # Show selected sample image
#             if 'sample_image' in st.session_state:
#                 st.divider()
#                 st.success(f"✅ Selected: **{st.session_state.get('sample_name', 'Sample')}**")
#                 image_to_predict = st.session_state['sample_image']
#                 image_source = "sample"
        
#         # Predict button
#         if image_to_predict is not None:
#             st.divider()
#             if st.button("🔍 Identify Species", type="primary", use_container_width=True):
#                 with st.spinner("🤔 Analyzing butterfly..."):
#                     try:
#                         # Preprocess image
#                         processed_image = preprocess_image(image_to_predict)
                        
#                         # Make prediction
#                         predictions = model.predict(processed_image, verbose=0)
                        
#                         # Get top prediction
#                         top_class_idx = np.argmax(predictions[0])
#                         top_species = idx_to_class[top_class_idx]
#                         top_confidence = float(predictions[0][top_class_idx])
                        
#                         # Store in session state
#                         st.session_state['predictions'] = predictions
#                         st.session_state['top_species'] = top_species
#                         st.session_state['top_confidence'] = top_confidence
#                         st.session_state['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
#                         st.success("✅ Prediction complete!")
#                         st.balloons()
                        
#                     except Exception as e:
#                         st.error(f"❌ Prediction failed: {e}")
#         else:
#             st.info("👆 Upload or select a sample image to begin")
    
#     with col2:
#         st.header("🎯 Results")
        
#         if 'predictions' in st.session_state:
#             predictions = st.session_state['predictions']
#             top_species = st.session_state['top_species']
#             top_confidence = st.session_state['top_confidence']
            
#             # Main prediction card
#             confidence_class = get_confidence_color(top_confidence)
#             confidence_text = get_confidence_interpretation(top_confidence)
            
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h2 style="margin-top: 0; color: #10b981;">✨ Predicted Species</h2>
#                 <h1 style="margin: 0.5rem 0; color: #1f2937; font-size: 2rem;">{top_species}</h1>
#                 <p style="margin: 0; font-size: 1.5rem;" class="{confidence_class}">
#                     {top_confidence*100:.1f}% - {confidence_text}
#                 </p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Confidence gauge
#             st.plotly_chart(
#                 create_confidence_gauge(top_confidence, top_species),
#                 use_container_width=True
#             )
            
#             st.info(f"🕐 Predicted at: {st.session_state['prediction_time']}")
            
#             # Quick action
#             if st.button("🔄 Try Another Image", use_container_width=True):
#                 for key in ['predictions', 'top_species', 'top_confidence', 'prediction_time', 'sample_image', 'sample_name']:
#                     if key in st.session_state:
#                         del st.session_state[key]
#                 st.rerun()
#         else:
#             st.info("👆 Results will appear here after classification")
    
#     # Top predictions chart
#     if 'predictions' in st.session_state:
#         st.divider()
#         st.header("📊 Top 5 Most Likely Species")
        
#         col_chart1, col_chart2 = st.columns([2, 1])
        
#         with col_chart1:
#             st.plotly_chart(
#                 create_top_predictions_chart(st.session_state['predictions'], idx_to_class, top_k=5),
#                 use_container_width=True
#             )
        
#         with col_chart2:
#             st.subheader("🔍 Interpretation")
#             top_conf = st.session_state['top_confidence']
            
#             if top_conf >= 0.7:
#                 st.success("✅ **High Confidence**")
#                 st.write("Very reliable prediction!")
#             elif top_conf >= 0.4:
#                 st.warning("⚠️ **Medium Confidence**")
#                 st.write("Check alternatives below.")
#             else:
#                 st.error("❌ **Low Confidence**")
#                 st.write("Try a clearer photo.")
    
#     # Footer
#     st.divider()
#     st.markdown(f"""
#     <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
#         <p style="font-size: 1.1rem;">🦋 <strong>Butterfly Species Classifier</strong></p>
#         <p style="font-size: 0.9rem;">Created by <strong>Arju</strong> | 2026</p>
#         <p style="font-size: 0.85rem;">Trained on {len(class_indices) if class_indices else 75} species</p>
#     </div>
#     """, unsafe_allow_html=True)


# if __name__ == "__main__":
#     main()

















"""
🦋 Butterfly Species Classifier - Streamlit Web App
Production-ready web interface for butterfly identification

Features:
- Upload butterfly images OR use sample images from samples/ folder
- Get instant predictions
- View top-5 most likely species
- Confidence visualization
- Beautiful, user-friendly interface
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🦋 Butterfly Classifier",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #10b981;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0fdf4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #10b981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .confidence-low {
        color: #ef4444;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #10b981;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #059669;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_sample_images_from_folder():
    """
    Load sample images from samples/ folder
    Returns dictionary of {name: PIL Image}
    """
    sample_folder = "samples"
    sample_images = {}
    
    # Check if samples folder exists
    if not os.path.exists(sample_folder):
        return sample_images
    
    if not os.path.isdir(sample_folder):
        return sample_images
    
    # Get all image files
    try:
        all_files = os.listdir(sample_folder)
    except Exception as e:
        st.warning(f"Could not read samples folder: {e}")
        return sample_images
    
    # Filter for image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
    
    if not image_files:
        return sample_images
    
    # Load each image
    for filename in image_files:
        filepath = os.path.join(sample_folder, filename)
        try:
            # Load image
            img = Image.open(filepath)
            
            # Convert to RGB (handles RGBA, grayscale, etc.)
            img = img.convert('RGB')
            
            # Create clean name from filename
            name = os.path.splitext(filename)[0]  # Remove extension
            name = name.replace('_', ' ')          # Replace underscores with spaces
            name = name.replace('-', ' ')          # Replace dashes with spaces
            name = name.title()                    # Capitalize words
            
            # Store image
            sample_images[name] = img
            
        except Exception as e:
            # Skip files that can't be loaded
            st.warning(f"⚠️ Could not load {filename}: {str(e)}")
            continue
    
    return sample_images


@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    """Load the trained model and class indices with caching (silent)"""
    try:
        model_path = 'models/butterfly_model_WORKING.keras'
        
        if not os.path.exists(model_path):
            return None, None, None, "Model file not found"
        
        try:
            model = keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            return None, None, None, f"Model loading error: {str(e)}"
        
        class_indices_path = 'class_indices.json'
        if not os.path.exists(class_indices_path):
            return None, None, None, "class_indices.json not found"
            
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        return model, class_indices, idx_to_class, None
        
    except Exception as e:
        return None, None, None, f"Unexpected error: {str(e)}"


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_confidence_color(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_confidence_interpretation(confidence):
    """Return human-readable confidence interpretation"""
    if confidence >= 0.9:
        return "Very High Confidence"
    elif confidence >= 0.7:
        return "High Confidence"
    elif confidence >= 0.5:
        return "Medium Confidence"
    elif confidence >= 0.3:
        return "Low Confidence"
    else:
        return "Very Low Confidence"


def create_confidence_gauge(confidence, species_name):
    """Create a beautiful confidence gauge using Plotly"""
    if confidence >= 0.7:
        bar_color = "#10b981"
    elif confidence >= 0.4:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkgray"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},
                {'range': [40, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig


def create_top_predictions_chart(predictions, idx_to_class, top_k=5):
    """Create horizontal bar chart for top predictions"""
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_species = [idx_to_class[i] for i in top_indices]
    top_confidences = predictions[0][top_indices] * 100
    
    colors = []
    for c in top_confidences:
        if c >= 70:
            colors.append('#10b981')
        elif c >= 40:
            colors.append('#f59e0b')
        else:
            colors.append('#ef4444')
    
    fig = go.Figure(go.Bar(
        x=top_confidences,
        y=top_species,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{c:.1f}%' for c in top_confidences],
        textposition='auto',
        textfont=dict(size=14, color='white', family='Arial Black')
    ))
    
    fig.update_layout(
        title=f"Top {top_k} Most Likely Species",
        xaxis_title="Confidence (%)",
        yaxis_title="Species",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif", 'size': 12},
        xaxis=dict(gridcolor='lightgray', range=[0, 100]),
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🦋 Butterfly Species Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a butterfly image to identify its species using AI</p>', unsafe_allow_html=True)
    
    # Load model silently
    model, class_indices, idx_to_class, error = load_model_and_classes()
    
    if model is None:
        st.error(f"❌ Failed to load model: {error}")
        st.info("""
        **Setup Instructions:**
        1. Place `butterfly_model_best.keras` in `models/` directory
        2. Place `class_indices.json` in the project root
        3. Restart the Streamlit app
        """)
        st.stop()
    
    # Load sample images from folder
    sample_images = load_sample_images_from_folder()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(f"""
        This AI-powered app can identify **{len(class_indices)} different butterfly species** with high accuracy!
        
        **How to use:**
        1. Upload your own image OR try sample images
        2. Click 'Identify Species'
        3. Get instant predictions!
        
        **Best results:**
        - Clear, well-lit photos
        - Butterfly in focus
        - Minimal background clutter
        """)
        
        st.divider()
        
        st.header("📊 Model Info")
        if os.path.exists('model_info.json'):
            try:
                with open('model_info.json', 'r') as f:
                    model_info = json.load(f)
                st.write(f"**Model:** {model_info.get('best_model', 'MobileNetV2')}")
                st.write(f"**Accuracy:** {model_info.get('best_model_metrics', {}).get('accuracy', 0.85)*100:.1f}%")
                st.write(f"**Species:** {model_info.get('num_classes', len(class_indices))}")
            except:
                st.write(f"**Species:** {len(class_indices)}")
        else:
            st.write(f"**Architecture:** MobileNetV2")
            st.write(f"**Species:** {len(class_indices)}")
            st.write(f"**Accuracy:** 88.6%")
        
        st.divider()
        
        st.header("🎯 Confidence Guide")
        st.write("""
        - **High (>70%)** 🟢: Very reliable
        - **Medium (40-70%)** 🟡: Generally good
        - **Low (<40%)** 🔴: Needs verification
        """)
        
        st.divider()
        
        st.header("👨‍💻 Developer")
        st.write("**Created by:** Arju")
        st.write("**Year:** 2026")
        st.write("**Framework:** TensorFlow & Streamlit")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        tab1, tab2 = st.tabs(["📁 Upload Your Image", "🖼️ Try Sample Images"])
        
        image_to_predict = None
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Choose a butterfly image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of a butterfly"
            )
            
            if uploaded_file is not None:
                image_to_predict = Image.open(uploaded_file).convert('RGB')
                st.image(image_to_predict, caption='Your Uploaded Image', use_container_width=True)
                st.info(f"📐 Image size: {image_to_predict.size[0]} x {image_to_predict.size[1]} pixels")
        
        with tab2:
            if len(sample_images) > 0:
                st.write(f"**{len(sample_images)} sample images available. Click to test:**")
                
                # Display in 2 columns
                num_cols = 2
                sample_list = list(sample_images.items())
                
                for i in range(0, len(sample_list), num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i + j
                        if idx < len(sample_list):
                            name, img = sample_list[idx]
                            with cols[j]:
                                # Show image thumbnail
                                st.image(img, caption=name, use_container_width=True)
                                
                                # Button to select this image
                                if st.button(f"Use This", key=f"sample_{idx}", use_container_width=True):
                                    st.session_state['sample_image'] = img
                                    st.session_state['sample_name'] = name
                                    st.rerun()
                
                # Show selected sample
                if 'sample_image' in st.session_state:
                    st.divider()
                    st.success(f"✅ Selected: **{st.session_state.get('sample_name')}**")
                    image_to_predict = st.session_state['sample_image']
            else:
                st.info("""
                **No sample images found in `samples/` folder.**
                
                To add samples:
                1. Create a `samples/` folder in your project
                2. Add butterfly images (JPG, JPEG, PNG)
                3. Restart the app
                """)
        
        # Predict button
        if image_to_predict is not None:
            st.divider()
            if st.button("🔍 Identify Species", type="primary", use_container_width=True):
                with st.spinner("🤔 Analyzing butterfly..."):
                    try:
                        processed_image = preprocess_image(image_to_predict)
                        predictions = model.predict(processed_image, verbose=0)
                        
                        top_class_idx = np.argmax(predictions[0])
                        top_species = idx_to_class[top_class_idx]
                        top_confidence = float(predictions[0][top_class_idx])
                        
                        st.session_state['predictions'] = predictions
                        st.session_state['top_species'] = top_species
                        st.session_state['top_confidence'] = top_confidence
                        st.session_state['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        st.success("✅ Prediction complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
        else:
            st.info("👆 Upload or select a sample image to begin")
    
    with col2:
        st.header("🎯 Results")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            top_species = st.session_state['top_species']
            top_confidence = st.session_state['top_confidence']
            
            confidence_class = get_confidence_color(top_confidence)
            confidence_text = get_confidence_interpretation(top_confidence)
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="margin-top: 0; color: #10b981;">✨ Predicted Species</h2>
                <h1 style="margin: 0.5rem 0; color: #1f2937; font-size: 2rem;">{top_species}</h1>
                <p style="margin: 0; font-size: 1.5rem;" class="{confidence_class}">
                    {top_confidence*100:.1f}% - {confidence_text}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(
                create_confidence_gauge(top_confidence, top_species),
                use_container_width=True
            )
            
            st.info(f"🕐 Predicted at: {st.session_state['prediction_time']}")
            
            if st.button("🔄 Try Another Image", use_container_width=True):
                for key in ['predictions', 'top_species', 'top_confidence', 'prediction_time', 'sample_image', 'sample_name']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            st.info("👆 Results will appear here after classification")
            
            st.markdown("""
            **What you'll see:**
            - 🦋 Predicted species name
            - 📊 Confidence percentage  
            - 🎯 Visual confidence gauge
            - 🕐 Prediction timestamp
            """)
    
    # Top predictions chart
    if 'predictions' in st.session_state:
        st.divider()
        st.header("📊 Top 5 Most Likely Species")
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            st.plotly_chart(
                create_top_predictions_chart(st.session_state['predictions'], idx_to_class, top_k=5),
                use_container_width=True
            )
        
        with col_chart2:
            st.subheader("🔍 Interpretation")
            top_conf = st.session_state['top_confidence']
            
            if top_conf >= 0.7:
                st.success("✅ **High Confidence**")
                st.write("Very reliable prediction!")
            elif top_conf >= 0.4:
                st.warning("⚠️ **Medium Confidence**")
                st.write("Check alternatives below.")
            else:
                st.error("❌ **Low Confidence**")
                st.write("Try a clearer photo.")
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p style="font-size: 1.1rem;">🦋 <strong>Butterfly Species Classifier</strong></p>
        <p style="font-size: 0.9rem;">Created by <strong>Arju</strong> | 2026</p>
        <p style="font-size: 0.85rem;">Trained on {len(class_indices) if class_indices else 75} species</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()








