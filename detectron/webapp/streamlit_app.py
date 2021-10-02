"""Streamlit web app"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch
from detectron.predictor import NctModel
from argparse import ArgumentParser
st.set_option("deprecation.showfileUploaderEncoding", False)

arg_parser = ArgumentParser()
arg_parser.add_argument("--op_name", "-op", type=str, default="20190917")
arg_parser.add_argument("--path_location", "-p", type=str, default="crossval_full")
arg_parser.add_argument("--op_phase", "-ph", type=str, default="FIRST_INCISION")
args = arg_parser.parse_args()


@st.cache
def cached_model():
    model = NctModel()


    return model


model = cached_model()

st.title("Detecting tissues for first incision phase")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:

    pil_image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    # st.image(image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting tissues...")
    # with torch.no_grad():
    model.model_predict(image)
    predicted_image = model.model_predict(image)

    # if not annotations[0]["bbox"]:
    #     st.write("No faces detected")

        # visualized_image = vis_annotations(image, annotations)

    st.image(predicted_image, caption="After", use_column_width=True)
