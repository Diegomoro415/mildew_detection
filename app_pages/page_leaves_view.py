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

    st.write("### Leaves View")

# Checkbox for the user to visualize the difference between the average and variability of the images
    if st.checkbox("Difference between average and variability of the image"):
        version = 'v1'
        avg_var_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")
        avg_var_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")

        st.warning(
            "Note that the average and variability images did not show distinct patterns. "
            "However, a slight difference in color pigment is observed in the average images for both classes."
        )

        st.image(avg_var_healthy, caption='Healthy Leaf - Average and Variability')
        st.image(avg_var_powdery_mildew, caption='Powdery Mildew Leaf - Average and Variability')
        st.write("---")

    # Checkbox for the user to visualize the difference between the average images of healthy and powdery mildew leaves
    if st.checkbox("Differences between average healthy and average powdery mildew leaves"):
        avg_diff = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "Note that this study didn't show distinct patterns where we could intuitively differentiate one from another."
        )
        st.image(avg_diff, caption='Difference between average images')

    # Checkbox for the user to create an image montage
    if st.checkbox("Image Montage"):
        st.write("* To refresh the montage, click on the 'Create Montage' button")
        my_data_dir = 'inputs/cherryleaves_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label="Select class", options=labels, index=0)
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