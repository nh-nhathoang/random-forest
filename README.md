# Predicting Customer Buying Behavior

## Project Overview
This project aims to predict customer buying behavior using machine learning techniques. The dataset consists of various customer attributes, and the goal is to build a model that accurately classifies whether a customer will make a booking.

## Dataset
- **File**: `customer_booking.csv`
- **Features**: Various customer-related attributes.
- **Target Variable**: Whether the customer makes a booking (binary classification problem).

## Approach
1. **Data Exploration & Preprocessing**
   - Performed exploratory data analysis (EDA) to understand feature distributions and correlations.
   - Handled missing values and outliers.
   - Created new features where applicable to improve model performance.
   
2. **Model Training**
   - Implemented a **Random Forest Classifier** as the baseline model.
   - Used feature importance to analyze which variables contribute the most.
   - Handled class imbalance using techniques like resampling or adjusting class weights.

3. **Evaluation & Interpretation**
   - Evaluated model performance using **accuracy, precision, recall, and F1-score**.
   - Used cross-validation to ensure generalization.
   - Created visualizations to interpret feature importance and model predictions.

## Results
- **Model Accuracy**: 85%
- **Class Imbalance Issue**: While accuracy was high, recall for the minority class (customers who booked) was low, indicating the need for further model tuning.

## Future Improvements
- Experiment with other models like **XGBoost, LightGBM, or Logistic Regression**.
- Try oversampling/undersampling techniques or SMOTE to handle class imbalance.
- Tune hyperparameters to improve recall and overall performance.

## Files
- `Getting Started.ipynb`: Initial exploration and data preprocessing.
- `random_forest.ipynb`: Implementation of the Random Forest model and evaluation.

## Requirements
- Python 3.8+
- Required Libraries: `pandas, numpy, scikit-learn, matplotlib, seaborn`

## Usage
1. Load the dataset and preprocess the data.
2. Train the model using `random_forest.ipynb`.
3. Evaluate the model performance and analyze feature importance.

