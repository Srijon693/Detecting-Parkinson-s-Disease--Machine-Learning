# Parkinson's Disease Detection with Machine Learning

This repository contains a project focused on detecting Parkinson's disease using a dataset of various voice features. The project implements a machine learning model, utilizing the XGBoost algorithm to classify whether a subject has Parkinson's disease based on their voice features.

## Dataset

The dataset used in this project is sourced from the UCI Machine Learning Repository and contains various voice measurements collected from people, some with Parkinson's disease and some healthy.

### Dataset Columns:
- `name`: Patient ID
- `MDVP:Fo(Hz)`: Average vocal fundamental frequency
- `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
- `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency
- `MDVP:Jitter(%)`: Several measures of variation in fundamental frequency
- `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`: Different jitter measures
- `MDVP:Shimmer`, `MDVP:Shimmer(dB)`: Measures of variation in amplitude
- `status`: Health status (1 = Parkinson's, 0 = healthy) **(Target Variable)**
- And other features capturing different aspects of phonation.

## Project Overview

This project consists of two main parts:
1. **Data Visualization**: We used Plotly to create various visualizations to explore the relationships between different voice features.
2. **Machine Learning Model**: We built a classification model using the XGBoost algorithm to predict Parkinson's disease status.

### Visualizations:
- **Line Plot**: Shows the relationship between two features.
- **Scatter Plot**: Displays the correlation between two voice features.
- **Histogram**: Visualizes the distribution of a specific feature.
- **Box Plot**: Summarizes the spread and distribution of feature values.
- **3D Scatter Plot**: Illustrates the relationship between three features in 3D space.

### Model Development:
- **Scaling Features**: The features were scaled using the `MinMaxScaler` to bring them into a range of -1 to 1.
- **XGBoost Classifier**: The classifier was trained using a train-test split (85% training data, 15% testing data).
- **Evaluation**: The model achieved an accuracy score which is printed after the prediction step.

## Installation & Setup

### Prerequisites:
- Python 3.x
- Libraries: `pandas`, `numpy`, `plotly`, `xgboost`, `scikit-learn`

### Step-by-Step Instructions:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/parkinsons-disease-detection.git
    cd parkinsons-disease-detection
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons) or use the provided dataset in the repository.

4. Open and run the Jupyter Notebook or Google Colab with the code provided:
    - Explore data visualizations.
    - Train the XGBoost model on the dataset.
    - Evaluate the accuracy of the model.

## Code Structure

- `parkinsons.data`: The dataset used for training the model.
- `parkinsons_detection.ipynb`: Jupyter Notebook containing the code for visualization and model training.
- `README.md`: This file, detailing project setup and information.

## Results

The XGBoost model was able to predict the presence of Parkinson's disease with a high accuracy of **X%** (actual result will vary based on the train-test split). 

## Future Work

- Hyperparameter tuning can be applied to further improve model performance.
- Investigating other classification models (e.g., SVM, Random Forest) to compare performance.
- Adding more detailed feature analysis and insight generation.

## References

- Dataset: [Parkinson's Disease Dataset](https://www.kaggle.com/code/vikasukani/detecting-parkinson-s-disease-machine-learning/input) from UCI Machine Learning Repository.
- XGBoost Documentation: [XGBoost](https://xgboost.readthedocs.io/en/latest/)


