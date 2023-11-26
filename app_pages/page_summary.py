import streamlit as st


def page_summary_body():

    st.write("## Project Summary")
    st.write("---")

    st.info(
        f"**General Information**\n"
        f"* Cherry powdery mildew is a surface fungus that manifests itself "
        f"on both the developing leaves and fruit of cherry trees. "
        f"Cherry powdery mildew appears as areas of white fungal growth, "
        f"often in circular shapes. Presence of powdery mildew on cherries "
        f"not only damages the fruit on which it grows, but can quickly lead "
        f"to contamination of entire boxes of packaged cherries at harvest, "
        f"where fruit infected with powdery mildew begin to rot and quickly "
        f"spread to neighbouring fruit in the box or bin. \n\n")

    st.write(
        f"**source: **"
        f"[WSU-Decision Aid System](https://ca.decisionaid.systems/"
        f"articles/cherry_powdery_mildew)")

    st.warning(
        f"**Project Dataset**\n"
        f"* The available dataset contains 4208 image, The images show "
        f"2104 healthy cherry leaves images and 2104 cherry leaves that have "
        f"powdery mildew, a fungal disease that affects many plant species. "
        f"The cherry plantation crop is one of the finest products in their "
        f"portfolio, and the company is concerned about supplying the market "
        f"with a compromised quality product.")

    st.write(
        f"**Dataset: **"
        f"[Kaggle]"
        f"(https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)")

    st.success(
        f"**The project has 2 business requirements:**\n"
        f"* 1 - The client is interested in conducting a study to visually "
        f"differentiate a healthy cherry leaf from one with powdery mildew.\n"
        f"* 2 - The client is interested in predicting if a cherry leaf is "
        f"healthy or contains powdery mildew. ")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file]"
        f"(https://github.com/Diegomoro415/leaves_analysis/blob/main/"
        f"README.md)"
    )
