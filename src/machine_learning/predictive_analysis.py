import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results.

    Parameters:
        pred_proba (float): The predicted probability.
        pred_class (str): The predicted class.
    """

    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index={'Healthy': 0, 'Infected': 1}.keys(),
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    # Create a bar chart using Plotly Express
    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        color='Diagnostic',
        title='Probability Distribution',
        template='seaborn'
    )

    # Adjust the layout if needed
    fig.update_layout(width=600, height=400)

    # Show the pie chart using Streamlit
    st.plotly_chart(fig)


def resize_input_image(img, version):
    """
    Reshape image to average image size.

    Parameters:
        img (PIL.Image.Image): The input image.
        version (str): The version identifier.

    Returns:
        np.ndarray: The resized image as a NumPy array.
    """
    # Load the image shape from a saved pickle file
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")

    # Resize the image using Lanczos interpolation
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)

    # Expand dimensions and normalize the image
    my_image = np.expand_dims(img_resized, axis=0) / 255

    return my_image


def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images.

    Parameters:
        my_image (np.ndarray): The input image as a NumPy array.
        version (str): The version identifier.

    Returns:
        tuple: A tuple containing the predicted probability and class.
    """
    # Load the pre-trained model
    model = load_model(f"outputs/{version}/mildew_detection_model.keras")

    # Make a prediction on the input image
    pred_proba = model.predict(my_image)[0, 0]

    # Map numerical class to actual class names
    target_map = {v: k for k, v in {'Healthy': 0, 'Infected': 1}.items()}
    pred_class = target_map[pred_proba > 0.5]

    # Adjust probability and display the predicted class
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates the sample is a "
        f"**{pred_class.lower()}** leaf")

    return pred_proba, pred_class
