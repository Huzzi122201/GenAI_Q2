import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import os
import time

# Import custom modules
from utils import load_model, generate_image, create_sketch_from_image, preprocess_image
from model import GeneratorUNet

# Page configuration
st.set_page_config(
    page_title="Anime Sketch Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        transition: 0.3s;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header"><h1>🎨 Anime Sketch Colorization</h1><p>Transform your sketches into vibrant anime art using AI</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Model Configuration")
    
    # Model selection
    model_path = st.text_input(
        "Model Path",
        value="models/generator_final.pth",
        help="Path to your trained .pth file"
    )
    
    # Device selection
    device_option = st.selectbox(
        "Device",
        ["Auto", "CPU", "CUDA"],
        help="Select processing device"
    )
    
    if device_option == "Auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_option.lower()
    
    st.info(f"🖥️ Using device: {device.upper()}")
    
    # Load model button
    if st.button("🚀 Load Model", use_container_width=True):
        with st.spinner("Loading model..."):
            try:
                if os.path.exists(model_path):
                    model = load_model(model_path, device)
                    st.session_state['model'] = model
                    st.session_state['model_loaded'] = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error(f"❌ Model not found at {model_path}")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. **Load Model**: Enter the path to your .pth file and click 'Load Model'
    2. **Upload Sketch**: Upload a sketch or draw one
    3. **Generate**: Click 'Generate Colored Image'
    4. **Download**: Save the result
    
    **Tips:**
    - Use clear, well-defined sketches
    - Best results with 256x256 images
    - Try the sketch converter for photos
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    - 🎨 Real-time colorization
    - 📸 Sketch from photo conversion
    - 💾 Download results
    - 🚀 GPU acceleration
    """)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'generated_image' not in st.session_state:
    st.session_state['generated_image'] = None

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📤 Input")
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Draw Sketch", "Example Sketch"],
        horizontal=True
    )
    
    input_image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a sketch or image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        )
        
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            input_image = np.array(input_image)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)
    
    elif input_method == "Draw Sketch":
        st.info("Click the 'Draw' button below to start sketching")
        # Simple drawing canvas using streamlit's canvas component
        from streamlit_drawable_canvas import st_canvas
        
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
        )
        
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000000")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#FFFFFF")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            update_streamlit=True,
            height=400,
            width=400,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            input_image = canvas_result.image_data.astype(np.uint8)
            if input_image.shape[2] == 4:  # RGBA
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
            st.image(input_image, caption="Your Drawing", use_container_width=True)
    
    elif input_method == "Example Sketch":
        example_choice = st.selectbox(
            "Select example:",
            ["Anime Girl", "Samurai", "Dragon", "Flower", "Landscape"]
        )
        
        # Create some example sketches programmatically or load from files
        examples = {
            "Anime Girl": "examples/anime_girl_sketch.jpg",
            "Samurai": "examples/samurai_sketch.jpg",
            "Dragon": "examples/dragon_sketch.jpg",
            "Flower": "examples/flower_sketch.jpg",
            "Landscape": "examples/landscape_sketch.jpg"
        }
        
        # For demo, create a simple test pattern if example images don't exist
        if not os.path.exists(examples[example_choice]):
            # Create a dummy sketch
            img = np.ones((256, 256, 3), dtype=np.uint8) * 255
            cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 0), 3)
            cv2.circle(img, (128, 128), 50, (0, 0, 0), 3)
            input_image = img
        else:
            input_image = cv2.imread(examples[example_choice])
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        st.image(input_image, caption=f"Example: {example_choice}", use_container_width=True)
    
    # Option to convert photo to sketch
    if input_method == "Upload Image" and input_image is not None:
        if st.button("🖌️ Convert to Sketch"):
            with st.spinner("Converting to sketch..."):
                input_image = create_sketch_from_image(input_image)
                st.image(input_image, caption="Converted Sketch", use_container_width=True)
                st.success("✓ Image converted to sketch!")
    
    # Generate button
    st.markdown("---")
    if st.button("🎨 Generate Colored Image", use_container_width=True, type="primary"):
        if not st.session_state['model_loaded']:
            st.error("❌ Please load a model first!")
        elif input_image is None:
            st.error("❌ Please provide an input image!")
        else:
            with st.spinner("Generating colored image... This may take a few seconds"):
                try:
                    model = st.session_state['model']
                    start_time = time.time()
                    
                    # Generate image
                    output_image = generate_image(model, input_image, device)
                    
                    st.session_state['generated_image'] = output_image
                    st.session_state['generation_time'] = time.time() - start_time
                    
                    st.success(f"✅ Generation completed in {st.session_state['generation_time']:.2f} seconds!")
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")

with col2:
    st.markdown("## 🎨 Output")
    
    if st.session_state['generated_image'] is not None:
        # Display generated image
        st.image(
            st.session_state['generated_image'],
            caption="Generated Colored Image",
            use_container_width=True
        )
        
        # Display metrics
        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.metric("Generation Time", f"{st.session_state['generation_time']:.2f}s")
        with col_metrics2:
            img_shape = st.session_state['generated_image'].shape
            st.metric("Output Size", f"{img_shape[0]}x{img_shape[1]}")
        
        # Download button
        from io import BytesIO
        import base64
        
        def get_image_download_link(img_array, filename="colored_output.png"):
            img = Image.fromarray(img_array)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 Download Image</a>'
            return href
        
        st.markdown(
            get_image_download_link(st.session_state['generated_image']),
            unsafe_allow_html=True
        )
        
        # Comparison slider
        if input_method != "Draw Sketch" and input_image is not None:
            st.markdown("---")
            st.markdown("### 🔍 Compare Original vs Generated")
            
            # Resize original to match output if needed
            original_display = cv2.resize(input_image, (256, 256))
            comparison = np.hstack([original_display, st.session_state['generated_image']])
            st.image(comparison, caption="Original Sketch | Generated Color", use_container_width=True)
    else:
        # Placeholder
        st.info("👈 Upload a sketch and click 'Generate Colored Image' to see results here")
        
        # Show example of what to expect
        st.markdown("### 📸 Example Results")
        st.markdown("""
        *Upload a clear sketch to get the best results!*
        
        **Tips for best results:**
        - Use high contrast sketches (black on white)
        - Keep the drawing clear and not too messy
        - Face profiles work best for anime style
        - Try different input methods
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by pix2pix GAN | Made with Streamlit</div>",
    unsafe_allow_html=True
)

# Auto-load model if path exists
if not st.session_state['model_loaded'] and os.path.exists("models/best_generator.pth"):
    if st.sidebar.button("🔌 Auto-load default model"):
        with st.spinner("Loading default model..."):
            try:
                model = load_model("models/best_generator.pth", device)
                st.session_state['model'] = model
                st.session_state['model_loaded'] = True
                st.sidebar.success("✅ Model loaded!")
            except Exception as e:
                st.sidebar.error(f"Failed to load: {str(e)}")