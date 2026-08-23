import os
import argparse
import logging
import pandas as pd
from ydata_profiling import ProfileReport

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_profile_report(input_path: str, output_path: str):
    """
    Loads raw data, generates an automated EDA report, and saves it as HTML.
    """
    logging.info(f"Loading data from {input_path}...")
    
    try:
        # Load the dataset you downloaded on Day 1
        df = pd.read_csv(input_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Generate the profile report
        logging.info("Generating ydata-profiling report. This may take a minute...")
        profile = ProfileReport(
            df, 
            title="California Housing Data Profiling Report",
            explorative=True
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the report
        profile.to_file(output_path)
        logging.info(f"EDA report successfully saved to: {output_path}")
        
    except FileNotFoundError:
        logging.error(f"File not found at {input_path}. Did you run Day 1's ingestion script?")
        raise
    except Exception as e:
        logging.error(f"Error during EDA generation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated EDA Pipeline")
    parser.add_argument(
        '--input_path', 
        type=str, 
        default='data/raw/california_housing.csv',
        help='Path to the raw dataset'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='reports/data_profile.html',
        help='Path to save the HTML report'
    )
    
    args = parser.parse_args()
    generate_profile_report(args.input_path, args.output_path)
