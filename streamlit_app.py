import streamlit as st
from PIL import Image
import time

from ultralytics import YOLO
from pathlib import Path

# Models
object_classifer = YOLO('object_classifer.pt')
#scene_classifer = YOLO('scene_classifer.pt')

# Creating output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

st.set_page_config(layout="wide") 
st.title("Object Detection Model")
st.divider()

# Create two columns for the "Drag and Drop" vs "Output" look
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        input_path = output_dir / uploaded_file.name
        # Display the uploaded image
        st.image(img, caption="Original Image", use_container_width=True)
    else:
        # Placeholder box if no image is uploaded
        st.info("Please drag and drop or upload an image to start.")

with col2:
    st.subheader("Model Output")
    if uploaded_file is not None:
        """
        text = st.empty()
        text.text_input("Processing", key=f"{i}0{i}")
        text.empty()
        time.sleep(.5)
        text.empty()
        text.text_input("Processing .", key=f"{i}1{i}")
        time.sleep(.5)
        text.empty()
        text.text_input("Processing . .", key=f"{i}2{i}")
        time.sleep(.5)
        text.empty()
        text.text_input("Processing . . .", key=f"{i}3{i}")
        time.sleep(.5)
        text.empty()
        text.text_input("Processing .. .", key=f"{i}4{i}")
        time.sleep(.5)
        text.empty()
        text.text_input("Processing ...", key=f"{i}5{i}")
        time.sleep(.5)
        text.text_input("Processing .. .", key=f"{i}6{i}")
        time.sleep(.5)
        text.empty()
        text.text_input("Processing . . .", key=f"{i}7{i}")
        time.sleep(.5)
        """
        objects = object_classifer(input_path)
        for result in objects:
            save_path = output_dir / f"{input_path.stem}_detected.jpg"
            result.save(filename=str(save_path))

        st.image(str(save_path), caption="Processed Image with Bounding Boxes", use_container_width=True)
    else:
        st.write("Results will appear here.")
