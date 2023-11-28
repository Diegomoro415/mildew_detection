<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD051 -->
# 🌱 Cherry Leaf Mildew Detection

## Table of Contents

- [🌱 Cherry Leaf Mildew Detection](#-cherry-leaf-mildew-detection)
  - [Table of Contents](#table-of-contents)
    - [Project Dashboard](#project-dashboard)
  - [Brief Introduction](#brief-introduction)
  - [Dataset Content](#dataset-content)
  - [Business Requeriment](#business-requeriment)
  - [Hypothesis and Validations](#hypothesis-and-validations)
    - [Hypotheis 1](#hypothesis-1)
  - [Hypothesis 2](#hypothesis-2)
    - [Hypothesis 3](#hypothesis-3)

---

### Project Dashboard

- ### [Mildew Detection](https://mildew-detector-77f8e7e2c6fd.herokuapp.com/)

---

## Brief Introduction

Cherry Leaf Mildew Detection project, uses the power of machine learning to
transform the way we inspect cherry trees. Our goal is to visually
differentiate healthy cherry leaves from those affected by powdery mildew
and predict the health of cherry leaves. This project is driven by the
critical needs of the cherry plantation industry to ensure the supply of
high-quality cherries to the market efficiently and cost-effectively.

---

## Dataset Content

- The dataset is sourced from
[Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then
created a fictitious user story where predictive analytics can be applied
in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields.
The images show healthy cherry leaves and cherry leaves that have powdery
mildew, a fungal disease that affects many plant species. The cherry
plantation crop is one of the finest products in their portfolio, and
the company is concerned about supplying the market with a compromised
quality product.

---

## Business Requeriment

The cherry plantation crop from Farmy & Foods is facing a challenge where
their cherry plantations have been presenting powdery mildew. Currently,
the process is manual verification if a given cherry tree contains powdery
mildew. An employee spends around 30 minutes in each tree, taking a few
samples of tree leaves and verifying visually if the leaf tree is healthy
or has powdery mildew. If there is powdery mildew, the employee applies a
specific compound to kill the fungus. The time spent applying this compound
is 1 minute.  The company has thousands of cherry trees, located on multiple
farms across the country. As a result, this manual process is not scalable
due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects
instantly, using a leaf tree image, if it is healthy or has powdery mildew.
A similar manual process is in place for other crops for detecting pests,
and if this initiative is successful, there is a realistic chance to
replicate this project for all other crops. The dataset is a collection of
cherry leaf images provided by Farmy & Foods, taken from their crops.

1. The client is interested in conducting a study to visually differentiate a
healthy cherry leaf from one with powdery mildew.
2. The client is interested in predicting if a cherry leaf is healthy or
contains powdery mildew.

---

## Hypothesis and Validations

Hypothesis and Validation 1

- **Visual Differentiation:** We hypothesize that cherry leaves infected with
   powdery mildew exhibit distinctive visual patterns or marks that
   differentiate them from healthy leaves. This can include specific
    discolorations, lesions, or other identifiable features.

- **Validation:** To validate this hypothesis, we will conduct an
    in-depth analysis of the image data, exploring average images,
    variability images, and differences between averages. If clear patterns
    emerge, it supports the hypothesis.

Hypothesis and Validation 2

- **Model Prediction:** We hypothesize that a machine learning model,
  trained on a dataset of cherry leaf images, can accurately predict whether
  a leaf is healthy or infected with powdery mildew based on visual
  features extracted from the images.

- **Validation:** To validate this hypothesis, we will evaluate the model's
  performance metrics, including accuracy, precision, recall, and F1 score.
  A high level of accuracy and reliable predictions will support
  the effectiveness of the model.

Hypothesis and Validation 3

- **Time Efficiency:** We hypothesize that implementing the ML system for
  instant cherry leaf analysis will significantly reduce the time spent on
  manual inspections. The time saved will make the detection process more
  scalable across thousands of cherry trees.

- **Validation:** To validate this hypothesis, we will compare the time
  required for manual inspections with the time taken by the ML model for
  the same number of cherry trees. If the ML system demonstrates a substantial
  time reduction, it supports the hypothesis.

---

### **Hypothesis 1**
>
> **Visual Differentiation:** We hypothesize that cherry leaves infected with
powdery mildew exhibit distinctive visual patterns or marks that differentiate
them from healthy leaves. This can include specific discolorations, lesions,
or other identifiable features.

When cherry leaves are infected with Powdery Mildew, characteristic marks
become apparent. These may include specific discolorations, lesions, or other
identifiable features, evolving into distinctive white powdery spots. It's
crucial to convey this visual understanding to our system.

The initial step involves the transformation, splitting, and preparation of
our dataset for optimal learning outcomes. One key preprocessing step is
the normalization of our images.

Before we train our model, we must calculate the mean and standard deviation
for our images. The mean, obtained by dividing the sum of pixel values by the
total number of pixels in the dataset, helps understand the average brightness.
Meanwhile, standard deviation indicates the variation in brightness across
the images, aiding in distinguishing features.

- **Visual Representation**
  
  To visually demonstrate the distinction between healthy and infected leaves,
  we can create an image montage.

   ><details><summary>Healthy Leaves</summary><img src="/readme_images/healthy_leaves.png">
    </details>

   > <details><summary>Infected Leaves</summary><img src="/readme_images/infected_leaves.png">
   </details>

    Next, examining average and variability images allows us to identify
    patterns more clearly. In the case of infected leaves, we might observe
    more white spots and lines.
  
    > <details><summary>Difference between average and variability image</summary><img src="/readme_images/avarege_image.png"></details>

   However, it's crucial to note that there might be no visual differences
   in the average images of infected and healthy leaves, as illustrated below.

   > <details><summary>Differences between average healthy and average infected leaves</summary><img src="/readme_images/avarege_healthy_infected.png"></details>

- **Effective Learning**

    Despite potential challenges, the system demonstrates its capability to
    detect differences in our dataset. This step is vital as it ensures that
    our model can comprehend patterns and features, enabling accurate
    predictions for new data with similar challenges.

    This approach, grounded in visual markers and supported by robust
    preprocessing techniques, forms the foundation for our hypothesis
    validation. As we progress, it will be essential to assess how well
    our model learns and generalizes from these distinctive visual cues.

### **Hypothesis 2**

>**Model Prediction:** We hypothesize that a machine learning model,
  trained on a dataset of cherry leaf images, can accurately predict whether
  a leaf is healthy or infected with powdery mildew based on visual
  features extracted from the images.

- **Training for Predictive Accuracy**
  
  Our hypothesis revolves around the model's predictive prowess in discerning the health status of cherry leaves. To achieve this, our machine learning model must be trained on a dataset comprising images of both healthy and infected leaves. The primary focus is on extracting visual features from these images to enable accurate predictions.

    Prior to training, it's imperative to preprocess the dataset, employing techniques such as normalization for optimal learning conditions. Once equipped with the necessary features, the model can delve into the intricacies of distinguishing visual patterns associated with healthy and infected cherry leaves.
