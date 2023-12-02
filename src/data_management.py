# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime
import joblib


def download_dataframe_as_csv(df):
    """
    Generate a download link for a DataFrame in CSV format.

    Parameters:
        df (pd.DataFrame): The DataFrame to be downloaded.

    Returns:
        str: An HTML link for downloading the DataFrame as a CSV file.
    """
    # Get the current date and time for creating a unique filename
    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")

    # Convert DataFrame to CSV, encode as base64
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()

    # Generate a download link with a unique filename
    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="Report {datetime_now}.csv" '
        f'target="_blank">Download Report</a>'
    )

    return href


def load_pkl_file(file_path):
    """
    Load a file in Pickle format.

    Parameters:
        file_path (str): The path to the Pickle file.

    Returns:
        object: The loaded object from the Pickle file.
    """
    return joblib.load(filename=file_path)
