# Credit-Score-Classification

This Credit Score Classification project utilizes machine learning techniques to predict the creditworthiness of individuals based on various features such as income, credit history, debt-to-income ratio, and more. By training on historical data with labeled credit scores (Poor, Standard, Good), the model can classify new applicants into these categories.

## Project Structure

- **datasets/**
  - [Train.csv](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) (from Kaggle Credit Score Classification Dataset) - Dataset containing historical data used for training and testing the machine learning model.

- **models/**
  - [best_model.pkl.gz](models/best_model.pkl.gz) - The trained machine learning model serialized and compressed using gzip format.
  - [scaler.pkl](models/scaler.pkl) - The serialized MinMaxScaler object used to scale features during training.

- **notebooks/**
  - [Credit_Score_Classification.ipynb](notebooks/Credit_Score_Classification.ipynb) - Jupyter notebook containing code for data exploration, preprocessing, model training, and evaluation.

## Installation Requirements

To run the code in this project, you will need the following libraries:

- pandas
- matplotlib
- seaborn
- wordcloud
- scikit-learn
- streamlit
- scipy
- joblib

You can install these dependencies using pip:

```bash
pip install pandas matplotlib seaborn wordcloud scikit-learn streamlit scipy joblib
