# Importing necessary libraries
import streamlit as st
import os
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random


def page_leaves_view_body():
    st.write("# Data Visualizer")
    st.info(
        "Data Visualizer, where we delve into the intricate details"
        " of cherry leaves."
        " Our study focuses on visually differentiating leaves affected"
        " by powdery mildew from healthy ones."
    )

    st.write("## Study Overview")
    st.success(
        "In this research, our primary objective is to pinpoint visual"
        " indicators of powdery mildew infection on cherry leaves. Early signs"
        " often manifest as circular lesions with a subtle green tint,"
        " eventually giving rise to a delicate, cottony appearance"
        " manifests in the infected regions.\n\n"

        "Transforming these visual traits into the language of machine"
        " learning necessitates meticulous image preparation before initiating"
        " model training. This preparatory step is crucial for optimal"
        " feature extraction and effective training.\n\n"

        "When handling an image dataset, the imperative lies in the"
        " normalization of images before the Neural Network training process."
        " Normalization entails determining the mean and standard deviation"
        " across the entire dataset, employing a mathematical algorithm"
        " tailored to the inherent qualities of each image.\n\n"

        "This normalization process significantly enhances the model's"
        " capacity to comprehend and generalize from the dataset,"
        " thereby elevating overall performance."
    )

    st.write("For additional details and guidelines, please refer to the Project "
             "[Project README](https://github.com/Diegomoro415/mildew_detection/tree/main)")

    # Using expanders for each section
    with st.beta_expander("Difference between average and variability image"):
        version = 'v1'
        avg_var_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")
        avg_var_powdery_mildew = plt.imread(
            f"outputs/{version}/avg_var_powdery_mildew.png")

        st.warning(
            "Note that the average and variability images did not show distinct"
            " patterns. "
            "However, a slight difference in color pigment is observed in the"
            " average images for both classes."
        )

        st.image(avg_var_healthy,
                 caption='Healthy Leaf - Average and Variability')
        st.image(avg_var_powdery_mildew,
                 caption='Powdery Mildew Leaf - Average and Variability')
        st.write("---")

    with st.beta_expander(
            "Differences between average healthy and average infected leaves"):
        avg_diff = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "Note that this study didn't show distinct patterns where we"
            " could intuitively differentiate one from another."
        )
        st.image(avg_diff, caption='Difference between average images')

    with st.beta_expander("Image Montage"):
        st.write(
            "* To refresh the montage, click on the 'Create Montage' button"
        )
        my_data_dir = 'inputs/cherryleaves_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(
            label="Select class", options=labels, index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=3, ncols=3, figsize=(10, 10))
        st.write("---")


# Function to create the image montage
def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    # Check if the selected class exists
    if label_to_display in labels:
        images_list = os.listdir(os.path.join(dir_path, label_to_display))
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(
                f"Decrease the number of rows (nrows) or columns (ncols) to create your montage. \n"
                f"There are {len(images_list)} images in the class. "
                f"You requested a montage with {nrows * ncols} spaces"
            )
            return

        list_rows = range(0, nrows)
        list_cols = range(0, ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(0, nrows * ncols):
            img = imread(os.path.join(dir_path, label_to_display, img_idx[x]))
            img_shape = img.shape
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(
                f"Width {img_shape[1]}px x Height {img_shape[0]}px"
            )
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])

        st.pyplot(fig=fig)

    else:
        st.warning("The selected class does not exist.")
        st.warning(f"The existing options are: {labels}")


# Run the Streamlit code
if __name__ == '__main__':
    page_leaves_visualizer_body()
