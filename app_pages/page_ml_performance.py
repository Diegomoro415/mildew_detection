# Importing necessary libraries
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_dashboard():
    """
    Generates the content for the 'ML Performance Metrics' dashboard page.

    This function creates a user-friendly presentation of how the dataset
    was divided and how the model performed on that data. It includes sections
    on Images Distribution, Model Performance, and Generalized Performance
    on the Test Set.
    """
    version = 'v1'
    st.write("## ML Performance Metrics")
    st.write("---")
    st.info(
        f"This dashboard provides a user-friendly presentation of how the"
        f" dataset was divided and how the model performed on that data."
    )

    # Section: Images Distribution
    st.write("### Images Distribution per Set and Label")

    st.warning(
        f"The leaves dataset was divided into three subsets:\n\n"
        f"- **Train set (70%):** Initial data used to train the model.\n"
        f"- **Validation set (10%):** Used to fine-tune the model.\n"
        f"- **Test set (20%):** Unseen data for final model evaluation."
    )
    # Display images showing the distribution of labels and sets
    labels_distribution_img = plt.imread(
        f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution_img,
             caption='Labels Distribution on Train, Validation, and Test Sets',
             width=500)

    sets_distribution_img = plt.imread(
        f"outputs/{version}/sets_distribution.png")
    st.image(sets_distribution_img, caption='Sets Distribution', width=500)

    st.write("---")

    # Section: Model Performance
    st.write("### Model Performance")

    # Classification Report
    st.warning(
        f"**Classification Report:**\n\n"
        f"Precision, Recall, F1 Score, and Support provide insights into"
        f" model performance for each class."
    )

    clf_report_img = plt.imread(f"outputs/{version}/c_report.png")
    st.image(clf_report_img, caption='Classification Report', width=500)

    # ROC Curve
    st.warning(
        f"**ROC Curve:**\n\n"
        f"ROC curve illustrates the model's ability to distinguish between"
        f" classes."
        f" AUC (Area Under the Curve) measures overall model performance."
    )

    roc_curve_img = plt.imread(f"outputs/{version}/roc_curve.png")
    st.image(roc_curve_img, caption='ROC Curve', width=500)

    # Confusion Matrix
    st.warning(
        f"**Confusion Matrix:**\n\n"
        f"Confusion Matrix helps evaluate classifier performance, showing"
        f" true positive/negative and false positive/negative."
    )

    confusion_matrix_img = plt.imread(
        f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix_img, caption='Confusion Matrix', width=500)

    # Model Performance Plots
    st.warning(
        f"**Model Performance:**\n\n"
        f"Loss and Accuracy plots depict the model's learning progress"
        f" during training."
    )

    model_losses_img = plt.imread(
        f"outputs/{version}/model_training_losses.png")
    st.image(model_losses_img, caption='Model Training Losses', width=500)

    model_acc_img = plt.imread(f"outputs/{version}/model_training_acc.png")
    st.image(model_acc_img, caption='Model Training Accuracy', width=500)

    st.write("---")

    # Section: Generalized Performance on Test Set
    st.write("### Generalized Performance on Test Set")

    test_evaluation_df = pd.DataFrame(
        load_test_evaluation(version), index=['Loss', 'Accuracy'])
    st.dataframe(test_evaluation_df)

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file]"
        f"(https://github.com/yourusername/yourproject#readme)."
    )
