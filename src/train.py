import os
import argparse
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_model(input_path: str, model_dir: str):
    """
    Loads raw data, preprocesses it, trains a Random Forest model, 
    and saves the model and scaler artifacts.
    """
    logging.info(f"Loading data from {input_path}...")
    
    try:
        # 1. Load Data
        df = pd.read_csv(input_path)
        
        # California Housing target column is 'MedHouseVal' when fetched via sklearn
        # If your Day 1 script saved it exactly as fetched, 'MedHouseVal' is the target.
        X = df.drop(columns=['MedHouseVal'])
        y = df['MedHouseVal']
        
        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logging.info(f"Data split complete. Training on {len(X_train)} samples.")
        
        # 3. Feature Scaling
        logging.info("Fitting StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Train Model
        logging.info("Training Random Forest Regressor (Baseline)...")
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1 # Use all available CPU cores
        )
        model.fit(X_train_scaled, y_train)
        
        # 5. Evaluate Model
        logging.info("Evaluating model on test set...")
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logging.info(f"Model Performance -> MSE: {mse:.4f} | R2 Score: {r2:.4f}")
        
        # 6. Save Artifacts
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logging.info(f"Artifacts saved successfully in {model_dir}/")
        
    except FileNotFoundError:
        logging.error(f"Data not found at {input_path}. Please run data ingestion first.")
        raise
    except KeyError as e:
        logging.error(f"Target column missing. Expected 'MedHouseVal', got columns: {df.columns}. Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Model Training Pipeline")
    parser.add_argument(
        '--input_path', 
        type=str, 
        default='data/raw/california_housing.csv',
        help='Path to the raw dataset'
    )
    parser.add_argument(
        '--model_dir', 
        type=str, 
        default='models',
        help='Directory to save model artifacts'
    )
    
    args = parser.parse_args()
    train_model(args.input_path, args.model_dir)
