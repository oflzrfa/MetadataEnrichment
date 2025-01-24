import os
import duckdb
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
from concurrent.futures import ProcessPoolExecutor

class SubjectivityDetection:
    def __init__(self, database_file='my_database.duckdb', output_folder='subjectivity_metadata'):
        """
        Initialize the subjectivity detector with the database file and output folder.
        """
        self.database_file = database_file
        self.output_folder = output_folder
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def load_updated_metadata(self, table_name):
        """Load data from the DuckDB table into a DataFrame."""
        conn = duckdb.connect(self.database_file)
        query = f"SELECT * FROM {table_name}"
        df = conn.execute(query).fetchdf()
        conn.close()
        return df

    def detect_subjectivity(self, text):
        """Detect subjectivity score and label using DistilBERT."""
        if not text:
            return None, None
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)
        
        subjectivity_score = probs[0][1].item()
        label = 'Subjective' if subjectivity_score > 0.5 else 'Objective'
        
        return subjectivity_score, label

    def add_subjectivity_columns_parallel(self, df):
        """Add subjectivity score and label columns to the DataFrame in parallel."""
        if 'cleaned_text' not in df.columns:
            print("No 'cleaned_text' column found in DataFrame.")
            return df

        num_workers = os.cpu_count()
        chunk_size = max(1, len(df) // num_workers)
        
        chunks = [df['cleaned_text'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.process_chunk, chunks))
        
        subjectivity_data = pd.concat(results, ignore_index=True)
        df[['score', 'label']] = subjectivity_data
        return df

    def process_chunk(self, chunk):
        """Process a chunk of data to detect subjectivity."""
        return pd.DataFrame([self.detect_subjectivity(text) for text in chunk], columns=['subjectivity_score', 'subjectivity_label'])

    def get_table_names(self):
        """Get a list of all table names in the DuckDB database."""
        conn = duckdb.connect(self.database_file)
        query = "SHOW TABLES"
        tables = conn.execute(query).fetchdf()
        conn.close()
        return tables['name'].tolist()

    def process_and_save_metadata(self):
        """Process all tables in the DuckDB database and update with subjectivity score and label."""
        table_names = self.get_table_names()
        
        for table_name in table_names:
            print(f"Processing table: {table_name}")
            
            df = self.load_updated_metadata(table_name)
            
            if df.empty:
                print(f"The '{table_name}' table is empty or could not be loaded.")
                continue
            
            df = self.add_subjectivity_columns_parallel(df)
            
            conn = duckdb.connect(self.database_file)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")  
            conn.close()

            output_file = os.path.join(self.output_folder, f"{table_name}_subjectivity.parquet")
            df.to_parquet(output_file)
            print(f"Exported table '{table_name}' to Parquet: {output_file}")

if __name__ == "__main__":
    subjectivity_detector = SubjectivityDetection()
    subjectivity_detector.process_and_save_metadata()
