# Importing necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time

# Importing custom functions
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)


def page_mildew_detector_body():
    """
    Generates the content for the 'Powdery Mildew Detector' page.

    This function provides a tool for identifying whether a cherry leaf is
    affected by powdery mildew or not. It allows users to upload images of
    cherry leaves for live predictions.
    """
    st.write("---")
    st.info(
        f"**Powdery Mildew Detector** \n\n This tool is designed to help"
        f" you identify whether a cherry leaf is affected"
        f" by powdery mildew or not."
    )

    st.write(
        f"To perform a live prediction, you can upload images of cherry"
        f" leaves. For your convenience, you can download a sample dataset"
        f" containing images of both healthy and infected leaves from "
        f"[here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    # Upload images
    images_buffer = st.file_uploader(
        "Upload images of cherry leaves. You may select more than one.",
        type=["PNG", "JPG"],
        accept_multiple_files=True,
    )

    if images_buffer is not None:
        # Initialize an empty DataFrame for the analysis report
        df_report = pd.DataFrame([])
        start_time = time.time()  # Record the start time of the analysis
        for image in images_buffer:
            # Open the uploaded image using PIL
            img_pil = Image.open(image)
            st.info(f"Cherry Leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            # Display the uploaded image
            st.image(
                img_pil,
                caption=(
                    f"Image Size: {img_array.shape[1]}px width x "
                    f"{img_array.shape[0]}px height"),
            )

            version = "v1"
            # Resize the input image for model compatibility
            resized_img = resize_input_image(img=img_pil, version=version)
            # Load the model and make predictions
            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version)
            # Plot the prediction probabilities
            plot_predictions_probabilities(pred_proba, pred_class)

            # Append the results to the analysis report DataFrame
            df_report = df_report.append(
                {"Name": image.name, "Result": pred_class}, ignore_index=True
            )

        end_time = time.time()  # Record the end time of the analysis
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        if not df_report.empty:
            # Display the analysis report table
            st.success("Analysis Report")
            st.table(df_report)
            # Provide a download link for the analysis report as a CSV file
            st.markdown(download_dataframe_as_csv(
                df_report), unsafe_allow_html=True)
            # Display the elapsed time for the analysis
            st.warning(
                f"Elapsed Time for Analysis: {elapsed_time:.2f} seconds")
