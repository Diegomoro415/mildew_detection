import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)


def page_mildew_detector_body():
    st.write("---")
    st.info(
        "Welcome to the Powdery Mildew Detector! This tool is designed to help"
        " you identify whether a cherry leaf is affected"
        " by powdery mildew or not."
    )

    st.write(
        "To perform a live prediction, you can upload images of cherry leaves."
        " For your convenience, you can download a sample dataset containing"
        " images of both healthy and infected leaves from"
        " [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader(
        "Upload images of cherry leaves. You may select more than one.",
        type=["PNG", "JPG"],
        accept_multiple_files=True,
    )

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        start_time = time.time()  # Registre o tempo inicial
        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"Cherry Leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(
                img_pil,
                caption=(
                    f"Image Size: {img_array.shape[1]}px width x "
                    f"{img_array.shape[0]}px height"),
            )

            version = "v1"
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append(
                {"Name": image.name, "Result": pred_class}, ignore_index=True
            )

        end_time = time.time()  # Registre o tempo final
        elapsed_time = end_time - start_time  # Calcule o tempo decorrido

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(
                df_report), unsafe_allow_html=True)

            st.info(f"Elapsed Time for Analysis: {elapsed_time:.2f} seconds")
