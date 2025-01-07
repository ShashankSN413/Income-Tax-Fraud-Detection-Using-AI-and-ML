1. Fraudulent Synthetic Data Generator
fraudulent_synthetic_data_generator.py is a Python script designed to generate synthetic transaction data with realistic features for fraud detection tasks. It allows you to control the volume of data and the fraud rate to simulate real-world scenarios. The generated dataset includes features such as transaction type, amounts, origins, destinations, timestamps, locations, and even noise for added complexity.
Key Features:
- Customizable Fraud Rate: Adjust the fraud occurrence probability to suit your needs.
- Realistic Features: Includes categorical features (e.g., transaction type, time, location) and numerical features (e.g., balances, amounts).
- Noise Injection: Adds noise features to simulate unpredictable real-world data.
- CSV Export: Saves the generated dataset as a CSV file for further analysis.
Usage:
Run the script and specify:
- Number of rows to generate.
- Desired fraud rate (e.g., 0.9 for 90% fraud cases).
- The dataset is saved as synthetic_fraud_dataset_with_noise.csv in the current directory.

2. Fraud Detector and Models Evaluator
fraud_detector_and_models_evaluator.py is a machine learning pipeline for detecting fraud in transaction data. It evaluates several models and provides insights into their performance.
Key Features:
- Preprocessing Pipeline: Handles categorical encoding, scaling, and preparation of features for machine learning models.
- Multiple Model Implementations:
- Random Forest Classifier: High-performance ensemble method.
- Decision Tree Classifier: Lightweight and interpretable.
- XGBoost: Gradient boosting for enhanced accuracy.
- Convolutional Neural Network (CNN): Leverages deep learning to capture patterns.
- LSTM (Long Short-Term Memory): For time-dependent patterns in sequential data.
- Evaluation Metrics: Reports accuracy, precision, recall, F1-score, and the number of fraud cases detected for each model.
- Customizable Training: Easily modify hyperparameters and data splits.
Usage:
Ensure the synthetic dataset (synthetic_fraud_dataset_with_noise.csv) is available.
Run the script to:
- Preprocess the dataset.
- Train and evaluate multiple models.
- Generate detailed classification reports and metrics.
- Compare model performance and select the best one for your application.
Applications
- Fraud Detection Systems: Ideal for testing and training fraud detection algorithms.
- Machine Learning Research: Explore the performance of various models on synthetic yet realistic datasets.
- Data Science Practice: Hands-on experience in data generation, preprocessing, and model evaluation.
Requirements
- Python 3.x
- Libraries: pandas, numpy, sklearn, xgboost, keras, and tensorflow.

Feel free to contribute by enhancing the data generator or adding new models for evaluation! 😊
