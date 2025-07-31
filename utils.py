import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Define constants and configuration
CONFIG = {
    'DATA_DIR': 'data',
    'MODEL_DIR': 'models',
    'TOKENIZER_NAME': 't5-base',
    'MAX_SEQ_LENGTH': 512,
    'BATCH_SIZE': 32,
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-5,
    'VALIDATION_SPLIT': 0.2
}

# Define exception classes
class DataLoadingError(Exception):
    """Error loading data"""
    pass

class DataPreprocessingError(Exception):
    """Error preprocessing data"""
    pass

class VisualizationError(Exception):
    """Error visualizing results"""
    pass

# Define utility functions
class DataProcessor:
    def __init__(self, data_dir: str, model_dir: str):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['TOKENIZER_NAME'])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG['TOKENIZER_NAME'])

    def load_data(self, file_name: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(os.path.join(self.data_dir, file_name))
            logging.info(f"Loaded data from {file_name}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise DataLoadingError(f"Error loading data: {str(e)}")

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # Split data into training and validation sets
            train_data, val_data = train_test_split(data, test_size=CONFIG['VALIDATION_SPLIT'], random_state=42)

            # Preprocess text data
            train_text = train_data['text']
            val_text = val_data['text']
            train_text = self.tokenizer.encode_plus(train_text, 
                                                     max_length=CONFIG['MAX_SEQ_LENGTH'], 
                                                     padding='max_length', 
                                                     truncation=True, 
                                                     return_attention_mask=True, 
                                                     return_tensors='pt')
            val_text = self.tokenizer.encode_plus(val_text, 
                                                   max_length=CONFIG['MAX_SEQ_LENGTH'], 
                                                   padding='max_length', 
                                                   truncation=True, 
                                                   return_attention_mask=True, 
                                                   return_tensors='pt')

            # Preprocess label data
            train_labels = train_data['label']
            val_labels = val_data['label']

            # Create data loaders
            train_dataset = Dataset(train_text, train_labels)
            val_dataset = Dataset(val_text, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

            logging.info("Preprocessed data")
            return train_loader, val_loader
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise DataPreprocessingError(f"Error preprocessing data: {str(e)}")

    def visualize_results(self, data: pd.DataFrame, labels: pd.Series) -> None:
        try:
            # Plot accuracy
            accuracy = accuracy_score(labels, data['predicted_label'])
            plt.bar(['Accuracy'], [accuracy])
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Accuracy')
            plt.show()

            # Plot classification report
            report = classification_report(labels, data['predicted_label'])
            print(report)

            # Plot confusion matrix
            matrix = confusion_matrix(labels, data['predicted_label'])
            plt.imshow(matrix, interpolation='nearest')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()

            logging.info("Visualized results")
        except Exception as e:
            logging.error(f"Error visualizing results: {str(e)}")
            raise VisualizationError(f"Error visualizing results: {str(e)}")

# Define main class
class Utils:
    def __init__(self, data_dir: str, model_dir: str):
        self.data_processor = DataProcessor(data_dir, model_dir)

    def load_data(self, file_name: str) -> pd.DataFrame:
        return self.data_processor.load_data(file_name)

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.data_processor.preprocess_data(data)

    def visualize_results(self, data: pd.DataFrame, labels: pd.Series) -> None:
        self.data_processor.visualize_results(data, labels)

# Create instance of Utils class
utils = Utils(CONFIG['DATA_DIR'], CONFIG['MODEL_DIR'])

# Example usage
if __name__ == "__main__":
    data = utils.load_data('train.csv')
    train_loader, val_loader = utils.preprocess_data(data)
    utils.visualize_results(data, data['label'])