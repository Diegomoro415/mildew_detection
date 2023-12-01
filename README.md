<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD051 -->
# 🌱 Cherry Leaf Mildew Detection

## Table of Contents

- [Cherry Leaf Mildew Detection](#-cherry-leaf-mildew-detection)
  - [Table of Contents](#table-of-contents)
    - [Project Dashboard](#project-dashboard)
  - [Brief Introduction](#brief-introduction)
  - [Dataset Content](#dataset-content)
  - [Business Requeriment](#business-requeriment)
  - [Hypothesis and Validations](#hypothesis-and-validations)
    - [**Hypothesis 1**](#hypothesis-1)
    - [**Hypothesis 2**](#hypothesis-2)
    - [**Hypothesis 3**](#hypothesis-3)
    - [**Rationale to Map business Requirements to Data Visualizations and ML Tasks**](#rationale-to-map-business-requirements-to-data-visualizations-and-ml-tasks)
      - [**Mapping Business Requirements to Data Visualizations and ML Tasks**](#mapping-business-requirements-to-data-visualizations-and-ml-tasks)
  - [ML Business Case](#ml-business-case)
  - [Dashboard Design (Streamlit App User Interface)](#dashboard-design-streamlit-app-user-interface)

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
3. The client wants to obtain a report from ML predictions on new leaves.

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
  
  Our hypothesis revolves around the model's predictive prowess in discerning
  the health status of cherry leaves. To achieve this, our machine learning
  model must be trained on a dataset comprising images of both healthy and
  infected leaves. The primary focus is on extracting visual features from
  these images to enable accurate predictions.
  
  Prior to training, it's imperative to preprocess the dataset, employing
  techniques such as normalization for optimal learning conditions.
  Once equipped with the necessary features, the model can delve into the
  intricacies of distinguishing visual patterns associated with
  healthy and infected cherry leaves.

- **Visual Representation**
  In addition to numerical metrics, visualizing the model's output is essential.
  This can be achieved by generating visualizations that showcase the model's
  predictions on sample images. These visual outputs provide a tangible
  representation of the model's understanding and offer insights into how well
  it can distinguish between healthy and infected leaves.

- ### **Distribution of Imagens per Set and Labels**

><details><summary>Labels Distribution</summary><img src="/outputs/v1/labels_distribution.png">
</details>

><details><summary>Sets Distribution</summary><img src="/outputs/v1/sets_distribution.png">
</details>

The set distribution shows how the dataset was divided among training,
validation, and test sets, while the labels distribution highlights the
proportion of healthy and infected leaves in each set.

- ### **Classification report**

><details><summary>Classification Report</summary><img src="/outputs/v1/c_report.png">
</details>

The classification report provides detailed metrics, including *Precision,
Recall, F1 Score*, and *Support* for each class, offering a comprehensive
understanding of the model's performance on specific categories.

- ### **ROC Curve**

><details><summary>ROC Curve</summary><img src="/outputs/v1/roc_curve.png">
</details>

The ROC curve illustrates the model's ability to distinguish between classes,
while the Area Under the Curve (AUC) measures the overall performance of
the model.

- ### **Confusion Matrix**

><details><summary>Confusion Matrix</summary><img src="/outputs/v1/confusion_matrix.png">
</details>

The confusion matrix helps evaluate classifier performance, showing true
positives/negatives and false positives/negatives.

- ### **Model Performance Plots**

><details><summary>Model Training Loss</summary><img src="/outputs/v1/model_training_loss.png">
</details>

><details><summary>Model Training Accurancy</summary><img src="/outputs/v1/model_training_losses.png">
</details>

The overall performance of the model on the test set is presented below:

><summary>Generalized Performance on Test Set</summary><img src="/readme_images/g_performance.png">

- **Conclusion:**

    Empowering Precision Agriculture
    The successful validation of this hypothesis not only substantiates the
    model's predictive accuracy but also positions it as a valuable tool in the
    realm of precision agriculture. By leveraging machine learning to swiftly
    assess the health of cherry leaves, we contribute to the efficiency and
    scalability of the inspection process.

### **Hypothesis 3**

> **Time Efficiency:** We hypothesize that implementing the ML system for
  instant cherry leaf analysis will significantly reduce the time spent on
  manual inspections. The time saved will make the detection process more
  scalable across thousands of cherry trees.

Implementing the ML system for instant cherry leaf analysis will significantly
reduce the time spent on manual inspections, making the detection process more
scalable across thousands of cherry trees.

- **Introduction:**
  
    Addressing Time Constraints
    Our hypothesis centers on the notion that integrating a machine learning system
    into the cherry leaf analysis workflow will lead to substantial time savings
    compared to manual inspections. This is particularly crucial for scalability,
    given the vast number of cherry trees under consideration.

- **Time Comparison:**

    Manual vs. ML Inspection
    To validate this hypothesis, a comparative analysis of the time required for
    manual inspections versus the ML system's instant analysis is essential.
    By measuring and contrasting these timeframes, we can gauge the efficiency
    gains brought about by the integration of our machine learning solution.

- **Scalability Impact:**

    Enabling Large-Scale Deployment
    Beyond time savings, the scalability impact of the ML system is a key
    aspect of this hypothesis. Successful validation will pave the way for
    the widespread deployment of the system across numerous cherry trees,
    addressing the scalability challenges inherent in manual inspection
    processes.

- **Visual Representation:**
  
    Our system includes a time tracking feature that records the duration of each analysis. This information provides a practical demonstration of the time saved compared to manual inspections.

    ><summary>Elapsed Time For Analysis</summary><img src="/readme_images/elapsed_time.png">

- **Conclusion:**

    A Paradigm Shift in Inspection Dynamics
    The validation of this hypothesis heralds a paradigm shift in cherry leaf
    inspection dynamics. The infusion of machine learning not only optimizes
    time utilization but also positions the agricultural process for seamless
    scalability, aligning with the evolving landscape of modern farming
    practices.

---

### **Rationale to Map business Requirements to Data Visualizations and ML Tasks**

The client's interest in conducting a study to visually differentiate healthy cherry leaves from those affected by powdery mildew steam from a critical need within the cherry plantation industry. The ability to rapidy identify leaves with powdery mildew is essential to ensure the supply of high-quality cherries to the market.

The process of manually inspecting each cherry tree, collecting leaf samples, and visually detecting powdery mildew is time-consuming and not scalable due to the large number of cherry tree across multiple farms. To address this challenge, the IT team proposed the implementation of a machine learning (ML) system, which aligns with the client's second requirement, oo predict the health of cherry leaves.

In alignment with the CRISP-DM methodology, our approach encompasses six key phases to ensure a systematic and effective development process:

1. **Business Understanding:**
   - Define objectives and requirements to conduct a visual differentiation study between healthy and powdery mildew-affected cherry leaves.
   - Set clear goals to achieve a nuanced understanding of the visual characteristics associated with both healthy and infected leaves.

2. **Data Understanding:**
   - Collect and explore the dataset to gain insights into the visual features that distinguish healthy cherry leaves from those with powdery mildew.

3. **Data Preparation:**
   - Clean, transform, and prepare the data, focusing on visual attributes crucial for conducting the differentiation study.

4. **Modeling:**
   - Apply various techniques to the data and evaluate results for effectiveness in meeting project goals.

5. **Evaluation:**
   - Assess model results, concentrating on the model's ability to visually differentiate between healthy and powdery mildew-affected cherry leaves.

6. **Deploymet:**
   - Deploy the model and continuously monitor results to ensure ongoing alignment with project objectives.

#### **Mapping Business Requirements to Data Visualizations and ML Tasks**

**Business Requirements 1:**
> The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.

*User Story:*

- As a user, I want an interactive, navigable dashboard for a clear understanding of visual features distinguishing healthy and infected cherry leaves.
- As a user, I want to visualize average and variability images to facilitate the visual differentiation study.
- As a user, I want to display the difference between average healthy and infected leaves for a nuanced visual understanding.
- As a user, I want an image montage of healthy and infected leaves for a comprehensive visual comparison.

 *Implementation:*

- Developed a Streamlit-based dashboard with an easy navigation sidebar.
- Calculated the difference between average infected and healthy leaves.
- Presented "mean" and "standard deviation" images for healthy and infected leaves.
- Created an image montage for both infected and healthy leaves.

**Business Requirement 2:**

>The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

*User Story:*

- As a user, I want a ML model to predict with high accuracy whether a cherry leaf is healthy or contains powdery mildew.

*Implementation:*

- Deployed an ML model with optimal hyperparameters, achieving a prediction accuracy of 97%.
- Enabled users to upload cherry leaf images for instant evaluation through an uploader widget.
- Displayed uploaded images with prediction statements, indicating the presence of powdery mildew and associated probabilities.

**Business Requirement 3:**
> The client wants to obtain a report from ML predictions on new leaves.

*User Story:*

- As a user, I want to obtain a report from ML predictions on new leaves.

*Implementation:*

- Generated a downloadable .csv report after each batch of uploaded images with predicted status.

---

## ML Business Case

In the context of our project, the Machine Learning (ML) business case revolves around developing a predictive model capable of discerning whether a cherry leaf is healthy or infected with powdery mildew. This problem falls under the category of supervised learning, specifically a two-class, single-label classification model.

**Objectives:**

1. **Prediction on Cherry Leaf Dataset:**
   - The primary objective is to create an ML model that predicts, based on a given dataset of cherry leaves, whether they are healthy or infected.
   - This is a supervised learning task with a binary outcome, making it a classification problem.

2. **Outcome Improvement:**
   - The model's purpose is to offer a more efficient and accurate means of detecting powdery mildew in cherry trees.
   - The current heuristic involves labor-intensive manual inspection, which is time-consuming and prone to human error. The goal is to significantly enhance this process.
  
3. **Model Success Metrics:**
   - Aim for a model accuracy of 97% or higher on the test set.
   - The model output will be a binary flag indicating whether the leaf is healthy or infected.

4. **User Interaction:**
   - Users, in this case, the owners or caretakers of cherry tree plantations, will capture images of leaves and upload them to the application.
   - The prediction will be provided swiftly, offering real-time insights into the health status of the leaves.

**Heuristics:**

- The existing method relies on manual inspections, consuming approximately 30 minutes for each tree. This process is laborious, prone to errors, and inefficient.
- The training dataset, consisting of 4208 images of cherry leaves, is provided by Farmy & Foody and available on Kaggle.

**Business Impact:**

- The successful implementation of the ML model is anticipated to revolutionize the inspection process, saving time, reducing labor costs, and improving the overall accuracy of powdery mildew detection.

---

## Dashboard Design (Streamlit App User Interface)

1. ### **Quick Project Summary**

   - General Information
     - Cherry powdery mildew is a surface fungus that manifests itself on both the developing leaves and fruit of cherry trees. Cherry powdery mildew appears as areas of white fungal growth, often in circular shapes. Presence of powdery mildew on cherries not only damages the fruit on which it grows, but can quickly lead to contamination of entire boxes of packaged cherries at harvest, where fruit infected with powdery mildew begin to rot and quickly spread to neighbouring fruit in the box or bin.

   - Project Dataset
     - The available dataset contains 4208 image, The images show 2104 healthy cherry leaves images and 2104 cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.
   - Business Requirements
     - The client is interested in conducting a study to visually differentiate a
     healthy cherry leaf from one with powdery mildew.
     - The client is interested in predicting if a cherry leaf is healthy or
     contains powdery mildew.
     - The client wants to obtain a report from ML predictions on new leaves.

2. ### **Leaves View**

   *It will answer business requirement 1*
   - **Study Overview**
      - In this research, our primary objective is to pinpoint visual indicators of powdery mildew infection on cherry leaves. Early signs often manifest as circular lesions with a subtle green tint, eventually giving rise to a delicate, cottony appearance manifests in the infected regions.Transforming these visual traits into the language of machine learning necessitates meticulous image preparation before initiating model training. This preparatory step is crucial for optimal feature extraction and effective training. When handling an image dataset, the imperative lies in the normalization of images before the Neural Network training process. Normalization entails determining the mean and standard deviation across the entire dataset, employing a mathematical algorithm tailored to the inherent qualities of each image. This normalization process significantly enhances the model's capacity to comprehend and generalize from the dataset, thereby elevating overall performance.
  
   >- Difference between Average and Variability Image
   >- Differences between average healthy and average infected leaves
   >- Image Montage

3. ### **Powdery Mildew Detector**

   *It will answer business requirement 2 and 3*
   - **Live Prediction**
     - Users can upload images of cherry leaves for live predictions. The tool provides detailed analysis, prediction statements, and associated probabilities.

   - **Sample Dataset**
     - To perform a live prediction, you can upload images of cherry leaves. For your convenience, you can download a sample dataset containing images of both healthy and infected leaves from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)

   - **Analysis Report**
     - After analysis, the tool generates a comprehensive report, including image names and prediction results, probability distribution and time elapsed.

    Example 1 - Healthy Sample

    - > <details><summary>Image Sample</summary><img src="/readme_images/healthy_sample_img.png">
    </details>

    - > <details><summary>Predict Analysis</summary><img src="/readme_images/healthy_predict.png">
    </details>

    - > <details><summary>Probability Distribution</summary><img src="/readme_images/healthy_probability.png">
    </details>

    - > <details><summary>Time elapsed</summary><img src="/readme_images/healthy_time.png">
    </details>

    Example 2 - Infected Sample

   - ><details><summary>Image Sample</summary><img src="/readme_images/infected_sample_img.png">
   </details>

   - ><details><summary>Predict Analysis</summary><img src="/readme_images/infected_predict.png">
   </details>

   - ><details><summary>Probability Distribution</summary><img src="/readme_images/infected_probability.png">
   </details>

   - ><details><summary>Time elapsed</summary><img src="/readme_images/infected_time.png">
   </details>

4. ### **Powdery Mildew Detector**

    - Block for each project hypothesis, describe the conclusion and how
     it is validated.

5. ### **ML Performance Metrics**

    - Images Distribution per Sets and Label
    - Model Performance
        - Classification Report
        - ROC Curve
        - Confusion Matrix
        - Model Performance - Loss and Accuracy
    - Generalized Performance on Test Set

---

## Technologies used

- Codeanywhere
- Jupyter Notebook
- Python
- Git
- Github
- Kaggle
- Numpy
- Pandas
- Seaborn
- Matplotlib
- TensorFlow
- Heroku

---

## Deployment

### **Heroku**

- The App live link is: <https://mildew-detector-77f8e7e2c6fd.herokuapp.com/>
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to **Heroku** using the following steps.

1. Log in to **Heroku** and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.
