"""
🦋 Butterfly Species Classifier - Streamlit Web App
Production-ready web interface for butterfly identification

Features:
- Upload butterfly images
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


@st.cache_resource
def load_model_and_classes():
    """Load the trained model and class indices with caching"""
    try:
        # Model path - using .keras format
        model_path = 'models/butterfly_model_WORKING.keras'
        
        # Check if model exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at: {model_path}")
            st.info("""
            **Setup Instructions:**
            1. Place `butterfly_model_best.keras` in `models/` directory
            2. Place `class_indices.json` in the project root
            3. Restart the Streamlit app
            """)
            return None, None, None
        
        st.info(f"📂 Loading model from: {model_path}")
        
        # Load model
        model = None
        try:
            # Load with compile=False for faster loading
            model = keras.models.load_model(model_path, compile=False)
            
            # Compile for predictions
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.info("""
            **Troubleshooting:**
            1. Make sure you have the .keras file (not .h5)
            2. File should be ~173 MB
            3. Run: `ls -lh models/butterfly_model_best.keras`
            """)
            return None, None, None
        
        # Load class indices
        class_indices_path = 'class_indices.json'
        if not os.path.exists(class_indices_path):
            st.error(f"❌ Class indices not found: {class_indices_path}")
            st.info("Run: `python generate_json_files.py` to create it")
            return None, None, None
            
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        # Create reverse mapping (index -> class name)
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        st.success(f"✅ Loaded {len(class_indices)} butterfly species")
        
        return model, class_indices, idx_to_class
        
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return None, None, None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
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
    # Determine color based on confidence
    if confidence >= 0.7:
        bar_color = "#10b981"  # Green
    elif confidence >= 0.4:
        bar_color = "#f59e0b"  # Yellow
    else:
        bar_color = "#ef4444"  # Red
    
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
    # Get top k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_species = [idx_to_class[i] for i in top_indices]
    top_confidences = predictions[0][top_indices] * 100
    
    # Create color scale based on confidence
    colors = []
    for c in top_confidences:
        if c >= 70:
            colors.append('#10b981')  # Green
        elif c >= 40:
            colors.append('#f59e0b')  # Yellow
        else:
            colors.append('#ef4444')  # Red
    
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
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, class_indices, idx_to_class = load_model_and_classes()
    
    # Check if model loaded
    if model is None:
        st.error("❌ Failed to load model. Please check the setup instructions above.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(f"""
        This AI-powered app can identify **{len(class_indices)} different butterfly species** with high accuracy!
        
        **How to use:**
        1. Upload a clear butterfly image
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
            st.write(f"**Format:** Keras 3.x (.keras)")
        
        st.divider()
        
        st.header("🎯 Tips")
        st.write("""
        - **High confidence (>70%)**: Very reliable
        - **Medium (40-70%)**: Generally good
        - **Low (<40%)**: May need verification
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a butterfly image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a butterfly"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Show image info
            st.info(f"📐 Image size: {image.size[0]} x {image.size[1]} pixels")
            
            # Predict button
            if st.button("🔍 Identify Species", type="primary"):
                with st.spinner("🤔 Analyzing butterfly..."):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        predictions = model.predict(processed_image, verbose=0)
                        
                        # Get top prediction
                        top_class_idx = np.argmax(predictions[0])
                        top_species = idx_to_class[top_class_idx]
                        top_confidence = float(predictions[0][top_class_idx])
                        
                        # Store in session state
                        st.session_state['predictions'] = predictions
                        st.session_state['top_species'] = top_species
                        st.session_state['top_confidence'] = top_confidence
                        st.session_state['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        st.success("✅ Prediction complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        st.info("Please try uploading a different image.")
    
    with col2:
        st.header("🎯 Results")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            top_species = st.session_state['top_species']
            top_confidence = st.session_state['top_confidence']
            
            # Main prediction card
            confidence_class = get_confidence_color(top_confidence)
            confidence_text = get_confidence_interpretation(top_confidence)
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="margin-top: 0; color: #10b981;">Predicted Species</h2>
                <h1 style="margin: 0.5rem 0; color: #1f2937;">{top_species}</h1>
                <p style="margin: 0; font-size: 1.5rem;" class="{confidence_class}">
                    {top_confidence*100:.1f}% - {confidence_text}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.plotly_chart(
                create_confidence_gauge(top_confidence, top_species),
                use_container_width=True
            )
            
            # Additional info
            st.info(f"🕐 Predicted at: {st.session_state['prediction_time']}")
        else:
            st.info("👆 Upload an image and click 'Identify Species' to see results")
    
    # Top predictions chart (full width)
    if 'predictions' in st.session_state:
        st.divider()
        st.header("📊 Top 5 Predictions")
        
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
                st.write("The model is very sure about this prediction!")
            elif top_conf >= 0.4:
                st.warning("⚠️ **Medium Confidence**")
                st.write("The prediction is likely correct, but consider the alternatives.")
            else:
                st.error("❌ **Low Confidence**")
                st.write("The model is uncertain. This might not be in the training dataset.")
            
            st.write("**What to do:**")
            if top_conf >= 0.7:
                st.write("- ✅ Trust this prediction")
                st.write("- 📚 Use for identification")
            elif top_conf >= 0.4:
                st.write("- 👀 Check top alternatives")
                st.write("- 🔍 Verify with expert")
            else:
                st.write("- ⚠️ Image may be unclear")
                st.write("- 🔄 Try a different photo")
                st.write("- 👤 Consult an expert")
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p>🦋 <strong>Butterfly Species Classifier</strong> | Powered by Deep Learning</p>
        <p style="font-size: 0.9rem;">Trained on {len(class_indices) if class_indices else 75} species | Built with TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()