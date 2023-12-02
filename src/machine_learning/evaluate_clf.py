# Importing necessary libraries
import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):
    """
    Load the evaluation results for the test set from a saved pickle file.

    Parameters:
        version (str): The version identifier for the evaluation results.

    Returns:
        dict: A dictionary containing evaluation metrics for the test set.
              The structure of the dictionary may include keys such as 'loss'
              and 'accuracy'.
    """
    return load_pkl_file(f'outputs/{version}/evaluation.pkl')
