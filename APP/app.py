import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import logging
from datetime import datetime
import dotenv
import json

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model_utils import load_image
from utils.inference_engine import InferenceEngine, MetricsCalculator
from utils.weight_manager import WeightManager
from utils.data_utils import DataManager, ResultsSaver

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== UTILITY FUNCTIONS ====================
def load_benchmark_metrics(dataset_type):
    """Load benchmark metrics from JSON file"""
    try:
        # Try multiple possible paths
        possible_paths = [
            f"weights/model_metrics_{dataset_type}.json",
            f"./weights/model_metrics_{dataset_type}.json",
            os.path.join(os.getcwd(), f"weights/model_metrics_{dataset_type}.json")
        ]
        
        for metrics_path in possible_paths:
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded metrics from: {metrics_path}")
                    return data
        
        logger.warning(f"Metrics file not found for {dataset_type}. Tried paths: {possible_paths}")
        
        # Return default metrics if file not found
        default_metrics = get_default_metrics(dataset_type)
        return default_metrics
        
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return get_default_metrics(dataset_type)

def get_default_metrics(dataset_type):
    """Return default metrics when JSON files are not available"""
    if dataset_type == "DRIVE":
        return {
            "model_name": "TransFuse DRIVE",
            "dataset": "DRIVE",
            "model_statistics": {
                "total_parameters": 57810,
                "model_size_mb": 0.2205,
                "flops_gflops": 2.5961,
                "macs_gmacs": 0.0000,
                "activation_memory_mb": 22.0009
            },
            "validation_metrics": {
                "f1_score": 0.9890,
                "dice_coefficient": 0.9890,
                "jaccard_index": 0.9783,
                "sensitivity": 0.9949,
                "specificity": 0.9627,
                "accuracy": 0.9800,
                "precision": 0.9835
            }
        }
    else:  # IOSTAR
        return {
            "model_name": "TransFuse IOSTAR",
            "dataset": "IOSTAR",
            "model_statistics": {
                "total_parameters": 66194,
                "model_size_mb": 0.2525,
                "flops_gflops": 2.6640,
                "macs_gmacs": 0.0000,
                "activation_memory_mb": 22.0009
            },
            "validation_metrics": {
                "f1_score": 0.9729,
                "dice_coefficient": 0.9729,
                "jaccard_index": 0.9472,
                "sensitivity": 0.9977,
                "specificity": 0.8826,
                "accuracy": 0.9650,
                "precision": 0.9487
            }
        }

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Retinal Vessel Segmentation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%);
        color: #e8eef5;
    }
    
    p, span, div, label {
        color: #e8eef5 !important;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMarkdownContainer"] h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 28px !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #e8eef5 !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    
    .main {
        background: transparent;
    }
    
    /* ==================== HEADER ==================== */
    .header-wrapper {
        background: linear-gradient(135deg, #1f4788 0%, #2d5eb8 50%, #1a3a6e 100%);
        padding: 40px 50px;
        border-radius: 16px;
        margin-bottom: 40px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .header-wrapper::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    }
    
    .header-wrapper h1 {
        font-size: 42px;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -1px;
        margin: 0;
    }
    
    /* ==================== SECTIONS & TITLES ==================== */
    .section-header {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 20px !important;
        margin-top: 0 !important;
        padding: 0 !important;
        padding-bottom: 12px !important;
        border-bottom: 2px solid rgba(100, 150, 220, 0.5) !important;
        letter-spacing: -0.5px !important;
        line-height: 1.2 !important;
    }
    
    .subsection-header {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #e0e8f5 !important;
        margin-bottom: 16px !important;
        letter-spacing: -0.3px !important;
    }
    
    h2 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    h4 {
        color: #ffffff !important;
    }
    
    /* ==================== UPLOAD CARD ==================== */
    .upload-card {
        background: linear-gradient(135deg, rgba(45, 94, 184, 0.1) 0%, rgba(26, 58, 110, 0.1) 100%);
        border: 2px solid rgba(45, 94, 184, 0.3);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .upload-card:hover {
        border-color: rgba(45, 94, 184, 0.6);
        background: linear-gradient(135deg, rgba(45, 94, 184, 0.15) 0%, rgba(26, 58, 110, 0.15) 100%);
        box-shadow: 0 8px 24px rgba(45, 94, 184, 0.15);
    }
    
    /* ==================== BUTTONS ==================== */
    .stButton > button {
        background: linear-gradient(135deg, #2d5eb8 0%, #1f4788 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(45, 94, 184, 0.3);
        text-transform: uppercase;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3568c8 0%, #2a5199 100%);
        box-shadow: 0 6px 20px rgba(45, 94, 184, 0.5);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(45, 94, 184, 0.3);
    }
    
    /* ==================== SUCCESS BOX ==================== */
    .success-container {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 2px solid rgba(52, 211, 153, 0.4);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(52, 211, 153, 0.1);
    }
    
    .success-container h4 {
        font-size: 17px;
        font-weight: 700;
        color: #34d399;
        margin: 0;
        letter-spacing: -0.3px;
    }
    
    /* ==================== METRICS CARDS ==================== */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(26, 58, 110, 0.2) 100%);
        border: 1px solid rgba(45, 94, 184, 0.3);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(8px);
    }
    
    .metric-card:hover {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.5) 0%, rgba(26, 58, 110, 0.3) 100%);
        border-color: rgba(45, 94, 184, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(45, 94, 184, 0.15);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        font-weight: 600;
        color: #a8bfe6;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    
    /* ==================== IMAGE CONTAINER ==================== */
    .image-container {
        background: linear-gradient(135deg, rgba(20, 40, 90, 0.5) 0%, rgba(15, 30, 70, 0.5) 100%);
        border: 1px solid rgba(45, 94, 184, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .image-label {
        font-size: 14px;
        font-weight: 700;
        color: #d4e6f1;
        margin-bottom: 12px;
        letter-spacing: 0.2px;
        text-transform: uppercase;
    }
    
    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.2) 0%, rgba(26, 58, 110, 0.1) 100%);
        border: 1px solid rgba(45, 94, 184, 0.2);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background: transparent;
        border: none;
        color: #a8bfe6;
        font-weight: 600;
        font-size: 14px;
        padding: 10px 18px;
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.2px;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #2d5eb8 0%, #1f4788 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(45, 94, 184, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background: rgba(45, 94, 184, 0.15);
        color: #e0e8f5;
    }
    
    /* ==================== INFO BOX ==================== */
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(8px);
    }
    
    .stInfo > div {
        color: #d4e6f1;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* ==================== ERROR BOX ==================== */
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(8px);
    }
    
    .stError > div {
        color: #fca5a5;
    }
    
    /* ==================== SUCCESS BOX ==================== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(8px);
    }
    
    .stSuccess > div {
        color: #a7f3d0;
    }
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(45, 94, 184, 0.15) 0%, rgba(26, 58, 110, 0.1) 100%);
        border: 2px dashed rgba(45, 94, 184, 0.4);
        border-radius: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(45, 94, 184, 0.7);
        background: linear-gradient(135deg, rgba(45, 94, 184, 0.25) 0%, rgba(26, 58, 110, 0.15) 100%);
    }
    
    /* ==================== DIVIDER ==================== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(45, 94, 184, 0.3), transparent);
        margin: 32px 0;
    }
    
    /* ==================== FOOTER ==================== */
    .footer-text {
        text-align: center;
        opacity: 0.6;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(45, 94, 184, 0.2);
        letter-spacing: 0.5px;
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(45, 94, 184, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(45, 94, 184, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(45, 94, 184, 0.6);
    }
    
    /* ==================== RESPONSIVE ==================== */
    @media (max-width: 768px) {
        .header-wrapper {
            padding: 30px 25px;
        }
        
        .header-wrapper h1 {
            font-size: 28px;
        }
        
        .section-header {
            font-size: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = 'DRIVE'
if 'inference_results' not in st.session_state:
    st.session_state.inference_results = None
if 'weight_manager' not in st.session_state:
    st.session_state.weight_manager = WeightManager(
        os.getenv('WEIGHTS_DIR', './weights')
    )
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

# ==================== HEADER ====================
st.markdown("""
<div class="header-wrapper">
    <h1>Retinal Vessel Segmentation</h1>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    dataset_type = st.radio(
        "Select Model Type:",
        options=['DRIVE', 'IOSTAR'],
        help="Choose between DRIVE or IOSTAR trained models"
    )
    
    st.markdown("---")
    
    # Check weight availability
    st.markdown("### 📦 Model Weights Status")
    available_weights = st.session_state.weight_manager.get_available_weights()
    
    for dataset, exists in available_weights.items():
        status = "✅ Available" if exists else "⬇️ Not Downloaded"
        st.write(f"{dataset}: {status}")
    
    if not available_weights.get(dataset_type, False):
        st.warning(f"⚠️ {dataset_type} weights not found. Will download on first use.")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This application performs segmentation of:
    - **Blood Vessels (BV)**
    - **Optic Disc (OD)**
    
    Using a TransFuse architecture with CBAM attention.
    """)

# ==================== MAIN CONTENT ====================
col1, col2 = st.columns([1, 1], gap="large")

# ==================== LEFT COLUMN: UPLOAD & INPUT ====================
with col1:
    st.write("")
    st.write("")
    st.markdown('<h2 class="section-header" style="margin-top: 0;">Upload Image</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'tif', 'bmp'],
        help="Supported formats: JPG, PNG, TIF, BMP"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<div class="image-label">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display image info
        st.markdown('<div class="subsection-header">Image Information</div>', unsafe_allow_html=True)
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Dimensions", f"{img.width} × {img.height}")
        with col_info2:
            st.metric("Format", img.format)

# ==================== RIGHT COLUMN: RESULTS ====================
with col2:
    st.write("")
    st.write("")
    st.markdown('<h2 class="section-header" style="margin-top: 0;">Processing</h2>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process button
        if st.button("Run Segmentation", use_container_width=True, type="primary"):
            try:
                with st.spinner("Loading model..."):
                    # Ensure weights are downloaded
                    weight_path = st.session_state.weight_manager.ensure_weights(
                        dataset_type
                    )
                    
                    # Get correct key_dim for this dataset
                    key_dim = st.session_state.weight_manager.get_key_dim(dataset_type)
                    
                    # Load model if different dataset
                    if st.session_state.model is None or st.session_state.current_dataset != dataset_type:
                        st.session_state.model = InferenceEngine(
                            weight_path,
                            input_size=int(os.getenv('MODEL_INPUT_SIZE', 512)),
                            key_dim=key_dim,
                            dataset_type=dataset_type
                        )
                        st.session_state.current_dataset = dataset_type
                
                with st.spinner("Running inference..."):
                    # Prepare image
                    img_array = load_image(uploaded_file)
                    if img_array is None:
                        st.error("Failed to load image. Please try another file.")
                    else:
                        # Run inference
                        results = st.session_state.model.predict(img_array)
                        
                        # Store results
                        st.session_state.inference_results = {
                            'bv': results['bv_prediction'],
                            'od': results['od_prediction'],
                            'time': results['inference_time'],
                            'image': img_array[0],
                            'filename': uploaded_file.name,
                            'dataset_type': dataset_type
                        }
                
                # Display success message
                st.markdown("""
                <div class="success-container">
                    <h4>Segmentation Complete</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display inference time
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    st.metric(
                        "Inference Time",
                        f"{results['inference_time']:.3f}s"
                    )
                with col_time2:
                    st.metric(
                        "Model",
                        f"{dataset_type}.h5"
                    )
                
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
                logger.error(f"Inference error: {str(e)}")

# ==================== VISUALIZATION SECTION ====================
if st.session_state.inference_results is not None:
    st.markdown("---")
    st.markdown('<h2 class="section-header">Segmentation Results</h2>', unsafe_allow_html=True)
    
    results = st.session_state.inference_results
    
    # Extract images
    image = results['image']
    bv = results['bv']
    od = results['od']

    # Convert grayscale → RGB if needed
    if bv.ndim == 2:  
        bv = np.stack([bv] * 3, axis=-1)

    if od.ndim == 2:
        od = np.stack([od] * 3, axis=-1)

    # Display in 3 columns with enhanced styling
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<div class="image-label">Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<div class="image-label">Blood Vessel Segmentation</div>', unsafe_allow_html=True)
        st.image(bv, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<div class="image-label">Optic Disc Segmentation</div>', unsafe_allow_html=True)
        st.image(od, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    # ==================== METRICS SECTION ====================
    st.markdown("---")
    st.markdown('<h2 class="section-header">Segmentation Quality Metrics</h2>', unsafe_allow_html=True)
    
    # Calculate metrics dynamically from predictions
    metrics_bv = MetricsCalculator.calculate_metrics(results['bv'])
    metrics_od = MetricsCalculator.calculate_metrics(results['od'])
    
    # Load benchmark metrics
    benchmark_metrics = load_benchmark_metrics(results['dataset_type'])
    
    # Display model statistics
    if benchmark_metrics and 'model_statistics' in benchmark_metrics:
        st.markdown('<div style="color: #a8bfe6; font-size: 13px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px;">Model Statistics</div>', unsafe_allow_html=True)
        
        stats = benchmark_metrics['model_statistics']
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        
        with col_s1:
            st.metric(
                "Parameters",
                f"{stats['total_parameters']:,}",
                help="Total model parameters"
            )
        
        with col_s2:
            st.metric(
                "Model Size",
                f"{stats['model_size_mb']:.4f} MB",
                help="Model size in float32"
            )
        
        with col_s3:
            st.metric(
                "FLOPs",
                f"{stats['flops_gflops']:.4f} G",
                help="Floating point operations"
            )
        
        with col_s4:
            st.metric(
                "MACs",
                f"{stats['macs_gmacs']:.4f} G",
                help="Multiply-accumulate operations"
            )
        
        with col_s5:
            st.metric(
                "Activation Memory",
                f"{stats['activation_memory_mb']:.4f} MB",
                help="Memory needed for activations"
            )
        
        st.markdown("---")
    
    # Create tabs for metrics
    tab1, tab2 = st.tabs(["Blood Vessel", "Optic Disc"])
    
    with tab1:
        st.markdown('<div class="subsection-header">Blood Vessel Analysis</div>', unsafe_allow_html=True)
        
        # ==================== PREDICTION QUALITY METRICS ====================
        st.markdown('<div style="color: #a8bfe6; font-size: 13px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px;">Prediction Quality Metrics</div>', unsafe_allow_html=True)
        
        if metrics_bv:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Mean Confidence",
                    f"{metrics_bv['mean_confidence']:.4f}",
                    help="Average prediction confidence (0-1)"
                )
            
            with col_m2:
                st.metric(
                    "Std Deviation",
                    f"{metrics_bv['std_confidence']:.4f}",
                    help="Consistency of predictions"
                )
            
            with col_m3:
                st.metric(
                    "Foreground Ratio",
                    f"{metrics_bv['foreground_ratio']:.2%}",
                    help="Percentage of vessel pixels detected"
                )
            
            with col_m4:
                st.metric(
                    "Edge Intensity",
                    f"{metrics_bv['edge_intensity']:.4f}",
                    help="Boundary sharpness"
                )
            
            col_m5, col_m6, col_m7, col_m8 = st.columns(4)
            
            with col_m5:
                st.metric(
                    "Min Value",
                    f"{metrics_bv['min_value']:.4f}",
                    help="Minimum prediction value"
                )
            
            with col_m6:
                st.metric(
                    "Max Value",
                    f"{metrics_bv['max_value']:.4f}",
                    help="Maximum prediction value"
                )
            
            with col_m7:
                st.metric(
                    "Num Components",
                    f"{int(metrics_bv['num_components'])}",
                    help="Number of connected vessel segments"
                )
            
            with col_m8:
                st.metric(
                    "Threshold",
                    f"{metrics_bv['threshold_used']:.4f}",
                    help="Decision threshold for binarization"
                )
        else:
            st.error("Could not compute metrics for Blood Vessel segmentation")
        
        # ==================== REFERENCE BENCHMARKS ====================
        st.markdown("---")
        st.markdown('<div style="color: #a8bfe6; font-size: 13px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px;">Reference Benchmarks (Validation Set)</div>', unsafe_allow_html=True)
        
        if benchmark_metrics and 'validation_metrics' in benchmark_metrics:
            val_metrics = benchmark_metrics['validation_metrics']
            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
            
            with col_b1:
                st.metric(
                    "F1 Score",
                    f"{val_metrics['f1_score']:.4f}",
                    help="Harmonic mean of precision and recall"
                )
            
            with col_b2:
                st.metric(
                    "Dice",
                    f"{val_metrics['dice_coefficient']:.4f}",
                    help="Dice coefficient from validation set"
                )
            
            with col_b3:
                st.metric(
                    "Jaccard",
                    f"{val_metrics['jaccard_index']:.4f}",
                    help="IoU (Jaccard Index)"
                )
            
            with col_b4:
                st.metric(
                    "Sensitivity",
                    f"{val_metrics['sensitivity']:.4f}",
                    help="True Positive Rate"
                )
            
            with col_b5:
                st.metric(
                    "Specificity",
                    f"{val_metrics['specificity']:.4f}",
                    help="True Negative Rate"
                )
        else:
            st.info("Benchmark metrics not available for this model")
    
    with tab2:
        st.markdown('<div class="subsection-header">Optic Disc Analysis</div>', unsafe_allow_html=True)
        
        # ==================== PREDICTION QUALITY METRICS ====================
        st.markdown('<div style="color: #a8bfe6; font-size: 13px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px;">Prediction Quality Metrics</div>', unsafe_allow_html=True)
        
        if metrics_od:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "Mean Confidence",
                    f"{metrics_od['mean_confidence']:.4f}",
                    help="Average prediction confidence (0-1)"
                )
            
            with col_m2:
                st.metric(
                    "Std Deviation",
                    f"{metrics_od['std_confidence']:.4f}",
                    help="Consistency of predictions"
                )
            
            with col_m3:
                st.metric(
                    "Foreground Ratio",
                    f"{metrics_od['foreground_ratio']:.2%}",
                    help="Percentage of disc pixels detected"
                )
            
            with col_m4:
                st.metric(
                    "Edge Intensity",
                    f"{metrics_od['edge_intensity']:.4f}",
                    help="Boundary sharpness"
                )
            
            col_m5, col_m6, col_m7, col_m8 = st.columns(4)
            
            with col_m5:
                st.metric(
                    "Min Value",
                    f"{metrics_od['min_value']:.4f}",
                    help="Minimum prediction value"
                )
            
            with col_m6:
                st.metric(
                    "Max Value",
                    f"{metrics_od['max_value']:.4f}",
                    help="Maximum prediction value"
                )
            
            with col_m7:
                st.metric(
                    "Num Components",
                    f"{int(metrics_od['num_components'])}",
                    help="Number of connected disc segments"
                )
            
            with col_m8:
                st.metric(
                    "Threshold",
                    f"{metrics_od['threshold_used']:.4f}",
                    help="Decision threshold for binarization"
                )
        else:
            st.error("Could not compute metrics for Optic Disc segmentation")
        
        # ==================== REFERENCE BENCHMARKS ====================
        st.markdown("---")
        st.markdown('<div style="color: #a8bfe6; font-size: 13px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px;">Reference Benchmarks (Validation Set)</div>', unsafe_allow_html=True)
        
        if benchmark_metrics and 'validation_metrics' in benchmark_metrics:
            val_metrics = benchmark_metrics['validation_metrics']
            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
            
            with col_b1:
                st.metric(
                    "F1 Score",
                    f"{val_metrics['f1_score']:.4f}",
                    help="Harmonic mean of precision and recall"
                )
            
            with col_b2:
                st.metric(
                    "Dice",
                    f"{val_metrics['dice_coefficient']:.4f}",
                    help="Dice coefficient from validation set"
                )
            
            with col_b3:
                st.metric(
                    "Jaccard",
                    f"{val_metrics['jaccard_index']:.4f}",
                    help="IoU (Jaccard Index)"
                )
            
            with col_b4:
                st.metric(
                    "Sensitivity",
                    f"{val_metrics['sensitivity']:.4f}",
                    help="True Positive Rate"
                )
            
            with col_b5:
                st.metric(
                    "Specificity",
                    f"{val_metrics['specificity']:.4f}",
                    help="True Negative Rate"
                )
        else:
            st.info("Benchmark metrics not available for this model")
    
    st.info("""
**Prediction Quality Metrics:**
- **Mean Confidence**: Average prediction confidence across all pixels (closer to 1 = more confident)
- **Std Deviation**: How varied the predictions are (higher = more contrast in segmentation)
- **Foreground Ratio**: Percentage of pixels classified as vessel/disc
- **Edge Intensity**: How sharp the boundaries are between structures
- **Num Components**: Number of connected regions (1 is ideal for single structure)

**Reference Benchmarks (Validation Set):**
- **F1 Score**: Harmonic mean of precision and recall
- **Dice**: Dice coefficient showing overlap with ground truth
- **Jaccard**: IoU (Intersection over Union) index
- **Sensitivity**: True Positive Rate (recall) from validation
- **Specificity**: True Negative Rate from validation
    """)
    
    # ==================== SAVE RESULTS ====================
    st.markdown("---")
    st.markdown('<h2 class="section-header">Save Results</h2>', unsafe_allow_html=True)
    
    col_save1, col_save2 = st.columns([2, 1])
    
    with col_save1:
        if st.button("Save Segmentation Results", use_container_width=True):
            try:
                saver = ResultsSaver(
                    st.session_state.data_manager.get_results_path()
                )
                result_folder = saver.save_prediction(
                    results['image'],
                    results['bv'],
                    results['od'],
                    results['filename'].split('.')[0],
                    results['dataset_type']
                )
                
                # Log metadata
                st.session_state.data_manager.save_result_metadata(
                    results['filename'],
                    {
                        'model': results['dataset_type'],
                        'inference_time': results['time']
                    },
                    results['dataset_type'],
                    results['time']
                )
                
                st.success(f"Results saved to: {result_folder}")
            except Exception as e:
                st.error(f"Error saving results: {str(e)}")
    
    with col_save2:
        st.metric(
            "Status",
            "Ready"
        )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="footer-text">
    <p>Retinal Segmentation Pipeline</p>
</div>
""", unsafe_allow_html=True)