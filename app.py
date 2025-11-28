"""
Hall of Fame Face Predictor - Streamlit App

An interactive app that predicts whether a baseball player "looks like" 
a Hall of Famer based on their Baseball Reference photo.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.scraper import scrape_bbref_images, get_player_name
from utils.preprocessor import preprocess_image

# Configure page
st.set_page_config(
    page_title="Hall of Fame Face Predictor",
    page_icon="⚾",
    layout="wide"
)

# Load model (cached)
@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    import tensorflow as tf
    model_path = Path(__file__).parent / "model" / "hof_model.keras"
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Please run `python model/train_model.py` first.")
        st.stop()
    
    return tf.keras.models.load_model(model_path)

# Header
st.title("⚾ Hall of Fame Face Predictor")
st.markdown("""
Does this player *look* like a Hall of Famer? Let's find out!

Enter a Baseball Reference player ID and we'll predict their Hall of Fame worthiness 
based solely on their face. Because apparently, we can judge a book by its cover.
""")

st.divider()

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    player_id = st.text_input(
        "Baseball Reference Player ID",
        placeholder="e.g., bondsba01, aaronha01, rosepe01",
        help="Find player IDs at baseball-reference.com (it's in the URL)"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

# Example IDs
with st.expander("Need help? Try these player IDs"):
    st.markdown("""
    - **bondsba01** - Barry Bonds (not in HOF)
    - **aaronha01** - Hank Aaron (HOF)
    - **rosepe01** - Pete Rose (banned)
    - **jeterde01** - Derek Jeter (HOF)
    - **pujolal01** - Albert Pujols (not yet eligible)
    - **troutmi01** - Mike Trout (active player)
    """)

# Prediction logic
if predict_button and player_id:
    try:
        # Progress indicators
        with st.spinner("Fetching player image from Baseball Reference..."):
            # Get player name and image
            player_name = get_player_name(player_id)
            image_path = scrape_bbref_images(player_id)
            original_img = Image.open(image_path)
        
        st.success(f"Found: **{player_name}**")
        
        # Preprocess image
        with st.spinner("Processing image..."):
            img_flat, img_gray, img_resized = preprocess_image(original_img)
        
        # Make prediction
        with st.spinner("Consulting the neural network oracle..."):
            model = load_model()
            prediction = model.predict(img_flat.reshape(1, -1), verbose=0)
            prob_not_hof = prediction[0][0]
            prob_hof = prediction[0][1]
        
        st.divider()
        
        # Display images
        st.subheader("📸 Image Transformation Pipeline")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(original_img, use_container_width=True)
            st.caption(f"Source: Baseball Reference")
        
        with col2:
            st.markdown("**Grayscale**")
            st.image(img_gray, use_container_width=True)
            st.caption("Converted to grayscale")
        
        with col3:
            st.markdown("**32x32 (Model Input)**")
            # Resize for display (make it bigger so it's visible)
            display_resized = img_resized.resize((128, 128), Image.NEAREST)
            st.image(display_resized, use_container_width=True)
            st.caption("Resized to 32x32 pixels")
        
        st.divider()
        
        # Display prediction
        st.subheader("🎯 Prediction Results")
        
        # Determine prediction class
        is_hof = prob_hof > 0.5
        
        # Create two columns for results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Probability bars
            st.markdown("**Hall of Fame Probability**")
            st.progress(float(prob_hof))
            st.metric("HOF Likelihood", f"{prob_hof*100:.1f}%")
            
            st.markdown("**Not Hall of Fame Probability**")
            st.progress(float(prob_not_hof))
            st.metric("Non-HOF Likelihood", f"{prob_not_hof*100:.1f}%")
        
        with col2:
            # Verdict
            if is_hof:
                st.success("### 🏆 Looks like a Hall of Famer!")
                if prob_hof > 0.8:
                    st.markdown("*The model is very confident!*")
                elif prob_hof > 0.6:
                    st.markdown("*Pretty good odds!*")
                else:
                    st.markdown("*It's close, but the face says yes!*")
            else:
                st.error("### 🚫 Doesn't look like a Hall of Famer")
                if prob_not_hof > 0.8:
                    st.markdown("*The model is very confident!*")
                elif prob_not_hof > 0.6:
                    st.markdown("*Probably not HOF material...*")
                else:
                    st.markdown("*It's close, but the face says no.*")
        
        # Disclaimer
        st.divider()
        st.info("""
        **⚠️ Disclaimer**: This is a fun project and should NOT be taken seriously! 
        A player's appearance has nothing to do with their baseball ability or Hall of Fame worthiness. 
        This model finds spurious correlations in image data and is meant for entertainment purposes only.
        """)
        
    except ValueError as e:
        st.error(f"❌ Error: {e}")
        st.info("Make sure you're using a valid Baseball Reference player ID. You can find it in the URL of the player's page.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.info("Something went wrong. Please try again or try a different player ID.")

elif predict_button:
    st.warning("Please enter a player ID first!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
Made with ❤️ and a healthy dose of skepticism | 
Data from <a href='https://www.baseball-reference.com'>Baseball Reference</a> | 
Model trained on 100+ player images
</div>
""", unsafe_allow_html=True)
