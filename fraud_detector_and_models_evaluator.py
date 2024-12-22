import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
import xgboost as xgb

data = pd.read_csv("/content/synthetic_fraud_dataset_with_noise.csv")

data.head(), data.info()
X = data.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
y = data['isFraud']

categorical_columns = ['type', 'transaction_time', 'transaction_location']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_columns)],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================
# Random Forest Model
# ============================
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=4, random_state=42))
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", rf_model.score(X_test, y_test))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

rf_fraud_count = np.sum(rf_pred)
print(f"Random Forest detected {rf_fraud_count} fraud cases")
print("-" * 50)

# ============================
# Decision Tree Model
# ============================
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=10, min_samples_split=4, random_state=42))
])

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree Accuracy:", dt_model.score(X_test, y_test))
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

dt_fraud_count = np.sum(dt_pred)
print(f"Decision Tree detected {dt_fraud_count} fraud cases")
print("-" * 50)

# ============================
# XGBoost Model
# ============================
X_train_xgb = preprocessor.fit_transform(X_train)
X_test_xgb = preprocessor.transform(X_test)

xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.01, random_state=42)

xgb_model.fit(X_train_xgb, y_train)

xgb_pred = xgb_model.predict(X_test_xgb)

print("XGBoost Accuracy:", xgb_model.score(X_test_xgb, y_test))
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

xgb_fraud_count = np.sum(xgb_pred)
print(f"XGBoost detected {xgb_fraud_count} fraud cases")
print("-" * 50)

# ============================
# Convolutional Neural Network (CNN)
# ============================
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

X_train_cnn = X_train_preprocessed.reshape((X_train_preprocessed.shape[0], X_train_preprocessed.shape[1], 1))
X_test_cnn = X_test_preprocessed.reshape((X_test_preprocessed.shape[0], X_test_preprocessed.shape[1], 1))

cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Dropout(0.3))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=1)

cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print(f"CNN Model Accuracy: {cnn_accuracy[1]:.2f}, Loss: {cnn_accuracy[0]:.2f}")

cnn_pred = (cnn_model.predict(X_test_cnn) > 0.5).astype("int32")
print("CNN Model Classification Report:")
print(classification_report(y_test, cnn_pred))

cnn_fraud_count = np.sum(cnn_pred)
print(f"CNN Model detected {cnn_fraud_count} fraud cases")
print("-" * 50)

# ============================
# LSTM Model
# ============================
X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential()
lstm_model.add(LSTM(100, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)

lstm_accuracy = lstm_model.evaluate(X_test_lstm, y_test)
print(f"LSTM Model Accuracy: {lstm_accuracy[1]:.2f}, Loss: {lstm_accuracy[0]:.2f}")

lstm_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype("int32")
print("LSTM Model Classification Report:")
print(classification_report(y_test, lstm_pred))

lstm_fraud_count = np.sum(lstm_pred)
print(f"LSTM Model detected {lstm_fraud_count} fraud cases")
print("-" * 50)