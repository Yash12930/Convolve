# Convolve - Credit Card Behavior Scoring

This project analyzes credit card data to predict customer behavior scores using a Random Forest model. It includes exploratory data analysis, model training, and a web application built with Streamlit to demonstrate the model's predictions.

## About The Project

The core of this project is a machine learning model trained to classify credit card holder behavior based on their transaction and profile data. The repository contains the datasets, Jupyter notebooks detailing the analysis and model creation process, and a simple interactive web app to use the trained model.

**Key Features:**
*   **Exploratory Data Analysis (EDA):** In-depth analysis of the customer dataset to find patterns and insights.
*   **Machine Learning Model:** A Random Forest classifier is trained and evaluated for predicting behavior scores.
*   **Saved Model:** The trained model and data scaler are saved as `.pkl` files for easy deployment.
*   **Interactive Web App:** A Streamlit application allows users to interact with the model and get predictions.

### Built With
This project utilizes the following main libraries and frameworks:
*   Python
*   Pandas
*   Scikit-learn
*   Jupyter Notebook
*   Streamlit

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need Python and pip installed. It is recommended to use a virtual environment to manage dependencies.
```
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate
pip install -r requirements.txt
```


### Installation

1.  Clone the repository:
    ```
    git clone https://github.com/Yash12930/Convolve.git
    ```
2.  Navigate to the project directory:
    ```
    cd Convolve
    ```
3.  Install the required packages as described in the Prerequisites section.

## Usage

There are two primary ways to use this repository:

1.  **Run the Web Application**
    To start the interactive Streamlit application, run the following command from the project's root directory:
    ```
    streamlit run streamlit.py
    ```

2.  **Explore the Notebooks**
    To understand the data analysis and model training process, you can explore the Jupyter notebooks:
    *   `eda.ipynb`: Contains the exploratory data analysis.
    *   `finalcode.ipynb`: Shows the final model building, training, and evaluation steps.

## Repository Structure

Here is an overview of the key files in this repository:

*   `streamlit.py`: The script for the Streamlit web application.
*   `finalcode.ipynb`: The main Jupyter Notebook for model training.
*   `eda.ipynb`: Jupyter Notebook for exploratory data analysis.
*   `random_forest_model.pkl`: The serialized, pre-trained Random Forest model.
*   `scaler.pkl`: The saved scaler object for preprocessing data.
*   `Dev_data_to_be_shared.csv`: The dataset used for training the model.
*   `validation_data_to_be_shared.csv`: The dataset used for validating the model.
*   `Credit_Card_Behaviour_Scores.docx`: A document providing details about the data features.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.[1]

## Author

**Yash12930** - [GitHub Profile](https://github.com/Yash12930)
