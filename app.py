import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

model = keras.models.load_model(r'E:/AppCode/model_hand2.h5')
stroke_width = st.sidebar.slider("Stroke width: ", 1, 32,32)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

word_dict = {0:'a', 1:'aa', 2:'aw', 3:'b', 4:'c', 5:'d', 6:'dd', 7:'e', 8:'ee', 9:'g', 10:'h', 11:'i', 12:'k', 13:'l', 14:'m', 15:'n', 16:'o', 17:'oo', 18:'ow', 19:'p', 20:'q', 21:'r', 22:'s', 23:'t', 24:'u', 25:'uw', 26:'v', 27:'x', 28:'y'}
# word_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}

realtime_update = st.sidebar.checkbox("Update in realtime", True)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", 
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=300,
    drawing_mode="freedraw",
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)


if canvas_result.image_data is not None:
    image = canvas_result.image_data
    img_copy = image.copy()

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400,440))

    img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28,28))
    img_final =np.reshape(img_final, (1,28,28,1))

    st.image(img_final)

    st.title(word_dict[np.argmax(model.predict(img_final))])
