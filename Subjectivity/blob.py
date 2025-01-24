import os
import duckdb
import pandas as pd
import nltk
from textblob import TextBlob
from concurrent.futures import ProcessPoolExecutor

# Download required NLTK resources
nltk.download('punkt')

class SubjectivitySentimentDetectionTextBlob:
    def __init__(self, database_file='my_database.duckdb', output_folder='subjectivity_sentiment_lexicon_metadata'):
        """
        Initialize the subjectivity and sentiment detector with the database file and output folder.
        """
        self.database_file = database_file
        self.output_folder = output_folder
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def load_updated_metadata(self, table_name):
        """Load data from the DuckDB table into a DataFrame."""
        conn = duckdb.connect(self.database_file)
        query = f"SELECT * FROM {table_name}"
        df = conn.execute(query).fetchdf()
        conn.close()
        return df

    def detect_subjectivity_sentiment(self, text):
        """Detect subjectivity and sentiment using TextBlob."""
        if not text:
            return None, None, None
        
        blob = TextBlob(text)
        
        subjectivity_score = blob.sentiment.subjectivity
        
        polarity_score = blob.sentiment.polarity
        if polarity_score > 0:
            sentiment_label = 'Positive'
        elif polarity_score < 0:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        subjectivity_label = 'Subjective' if subjectivity_score > 0.5 else 'Objective'
        
        return subjectivity_score, subjectivity_label, polarity_score, sentiment_label

    def add_subjectivity_sentiment_columns_parallel(self, df):
        """Add subjectivity score, subjectivity label, sentiment score, and sentiment label columns to the DataFrame in parallel."""
        if 'cleaned_text' not in df.columns:
            print("No 'cleaned_text' column found in DataFrame.")
            return df

        num_workers = os.cpu_count()
        chunk_size = max(1, len(df) // num_workers)
        
        chunks = [df['cleaned_text'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.process_chunk, chunks))
        
        subjectivity_sentiment_data = pd.concat(results, ignore_index=True)
        df[['score_subjectivity_textblob', 'label_subjectivity_textblob',
            'score_sentiment_textblob', 'label_sentiment_textblob']] = subjectivity_sentiment_data
        return df

    def process_chunk(self, chunk):
        """Process a chunk of data to detect subjectivity and sentiment."""
        return pd.DataFrame([self.detect_subjectivity_sentiment(text) for text in chunk], 
                            columns=['score_subjectivity_textblob', 'label_subjectivity_textblob',
                                     'score_sentiment_textblob', 'label_sentiment_textblob'])

    def get_table_names(self):
        """Get a list of all table names in the DuckDB database."""
        conn = duckdb.connect(self.database_file)
        query = "SHOW TABLES"
        tables = conn.execute(query).fetchdf()
        conn.close()
        return tables['name'].tolist()

    def process_and_save_metadata(self):
        """Process all tables in the DuckDB database and update with subjectivity and sentiment score and label."""
        table_names = self.get_table_names()
        
        for table_name in table_names:
            print(f"Processing table: {table_name}")
            
            df = self.load_updated_metadata(table_name)
            
            if df.empty:
                print(f"The '{table_name}' table is empty or could not be loaded.")
                continue
            
            df = self.add_subjectivity_sentiment_columns_parallel(df)
            conn = duckdb.connect(self.database_file)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")  # Overwrite table in DuckDB
            conn.close()

            output_file = os.path.join(self.output_folder, f"{table_name}_subjectivity_sentiment_textblob.parquet")
            df.to_parquet(output_file)
            print(f"Exported table '{table_name}' to Parquet: {output_file}")

if __name__ == "__main__":
    subjectivity_sentiment_detector = SubjectivitySentimentDetectionTextBlob()
    subjectivity_sentiment_detector.process_and_save_metadata()
