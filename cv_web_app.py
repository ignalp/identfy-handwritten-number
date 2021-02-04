# Importing the libraries.
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import time
from utils import *


def prediction(net):   

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ","#FFFFFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000")
    bg_image = None
    drawing_mode = "freedraw"
    #realtime_update = st.sidebar.checkbox("Update in realtime", True)
    realtime_update =  True

# Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="" if bg_image else bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=280, width=280,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    digit = 0 
    confidence = 0 

    # Predicting the image
    if canvas_result.image_data is not None:
        if st.button('Predict'):
            # Model inference
            digit, confidence = predictDigit(canvas_result.image_data,net)
    

    return digit, confidence


def main():
    # Load Digit Recognition model
    net = cv2.dnn.readNetFromONNX('model.onnx')
    
    st.title("Digit Recognizer")
    st.write("\n\n")
    st.write("Draw a digit below and click on Predict button")
    st.write("\n")
    st.write("To clear the digit, click the trash bin icon below the image")
    
    digit, confidence = prediction(net)

    st.write('Recognized Digit: {}'.format(digit))
    st.write('Confidence: {:.2f}'.format(confidence))

    


if __name__ == '__main__':
    main()
