import os
import argparse
import logging
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Configure logging so we can track this in our CI/CD pipeline later
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_save_data(output_path: str):
    """
    Fetches the California Housing dataset and saves it to the specified path.
    """
    logging.info("Starting data ingestion process...")
    
    try:
        # Fetch the dataset from Scikit-Learn
        logging.info("Fetching California Housing dataset from scikit-learn...")
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        
        logging.info(f"Dataset successfully fetched. Shape: {df.shape}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logging.info(f"Raw data successfully saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise

if __name__ == "__main__":
    # Use argparse so we can pass different paths via command line
    parser = argparse.ArgumentParser(description="Data Ingestion Pipeline")
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='data/raw/california_housing.csv',
        help='Path to save the raw dataset'
    )
    
    args = parser.parse_args()
    
    load_and_save_data(args.output_path)
