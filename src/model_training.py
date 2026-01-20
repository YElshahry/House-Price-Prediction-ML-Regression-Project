import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.utils import load_data
from src.data_preprocessing import preprocess_data, split_data

def train_model():
    print ("Loading data...")
    df = load_data('data/housing.csv')

    print ("Preprocessing data...")
    X, y, pipeline = preprocess_data(df)

    print ("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print ("Training Random Forest Regressor...")
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_train, y_train)

    print ("Evaluating model...")
    predictions = forest_reg.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: ${rmse:.2f}")

    # Save the model and pipeline
    print("Saving model artifacts...")
    joblib.dump(forest_reg, 'models/housing_model.pkl')

    return rmse

if __name__ == "__main__":
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    train_model()