import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("## Project Hypothesis and Validation")
    st.write("---")

    st.write(
        "### Hypothesis and Validation 1")
    st.info(
        f"**Visual Differentiation Hypothesis:**\n\n"
        f" We hypothesize that cherry leaves infected with powdery mildew"
        f" exhibit distinctive visual patterns or marks that differentiate"
        f" them from healthy leaves. This can include specific"
        f" discolorations, lesions, or other identifiable features."
    )

    st.success(
        f"**Validation:**\n\n To validate this hypothesis, we will conduct an"
        f" in-depth analysis of the image data, exploring average images,"
        f" variability images, and differences between averages. If clear"
        f" patterns emerge, it supports the hypothesis."
    )

    st.write("---")

    st.write("### Hypothesis and Validation 2")

    st.info(
        f"**Model Prediction Hypothesis:**\n\n We hypothesize that a"
        f" machine learning model, trained on a dataset of cherry leaf images,"
        f" can accurately predict whether a leaf is healthy or infected"
        f" with powdery mildew based on visual features"
        f" extracted from the images."
    )

    st.success(
        f"**Validation:**\n\n To validate this hypothesis, we will evaluate"
        f" the model's performance metrics, including accuracy, precision,"
        f" recall, and F1 score. A high level of accuracy and reliable"
        f" predictions will support the effectiveness of the model."
    )

    st.write("---")

    st.write("### Hypothesis and Validation 3")

    st.info(
        f"**Time Efficiency Hypothesis:**\n\n We hypothesize that implementing"
        f" the ML system for instant cherry leaf analysis will significantly"
        f" reduce the time spent on manual inspections. The time saved"
        f" will make the detection process more scalable"
        f" across thousands of cherry trees."
    )

    st.success(
        f"**Validation:**\n\n To validate this hypothesis, we will compare the"
        f" time required for manual inspections with the time taken"
        f" by the ML model for the same number of cherry trees."
        f" If the ML system demonstrates a substantial time reduction,"
        f" it supports the hypothesis."
    )

    st.write("---")

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file]"
        f"(https://github.com/yourusername/yourproject#readme)."
    )
