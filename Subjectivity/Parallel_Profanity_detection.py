import os
import duckdb
import pandas as pd
from better_profanity import profanity
from concurrent.futures import ProcessPoolExecutor

# Initialize the profanity filter
profanity.load_censor_words()

class ProfanityDetection:
    def __init__(self, output_folder='Profanity_metadata/'):
        """
        Initialize the detector with the output folder for results.
        """
        self.output_folder = output_folder
        self.ensure_output_folder_exists()

    def ensure_output_folder_exists(self):
        """
        Ensure the output folder exists. If not, create it.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def detect_profanity_batch(self, texts):
        """
        Detect profanity in a batch of text data.
        Returns a DataFrame with 'contains_profanity' and 'profanity_probability' columns.
        """
        results = []
        for text in texts:
            if not text: 
                results.append((0, 0.0))  # No profanity detected
                continue
            is_profane = profanity.contains_profanity(text)
            results.append((int(is_profane), 1.0 if is_profane else 0.0))
        return pd.DataFrame(results, columns=['contains_profanity', 'profanity_probability'])

    def add_profanity_columns_parallel(self, df):
        """
        Add two columns in parallel:
        - 'contains_profanity' to indicate profanity presence in 'cleaned_text'.
        - 'profanity_probability' to indicate the likelihood of profanity in 'cleaned_text'.
        """
        if 'cleaned_text' not in df.columns:
            print("No 'cleaned_text' column found in DataFrame.")
            return df

        num_workers = os.cpu_count()
        chunk_size = max(1, len(df) // num_workers)

        chunks = [df['cleaned_text'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(self.detect_profanity_batch, chunks)

        profanity_data = pd.concat(results, ignore_index=True)
        df[['contains_profanity', 'profanity_probability']] = profanity_data
        return df

    def export_to_parquet(self, df, table_name):
        """
        Export the processed DataFrame to a Parquet file in the output folder.
        """
        output_file = os.path.join(self.output_folder, f"{table_name}_profanity_metadata.parquet")
        df.to_parquet(output_file)
        print(f"Exported processed table '{table_name}' to Parquet: {output_file}")

    def process_metadata_in_duckdb(self, database_file='my_database.duckdb'):
        """
        Process all tables in the DuckDB database, add 'contains_profanity' and 'profanity_probability' columns,
        and save the result as Parquet files locally.
        """

        conn = duckdb.connect(database_file)
        tables = conn.execute("SHOW TABLES").fetchdf()['name'].tolist()

        if not tables:
            print(f"No tables found in DuckDB database {database_file}")
            return

        for table_name in tables:
            print(f"Processing table: {table_name}")

            query = f"SELECT * FROM {table_name}"
            df = conn.execute(query).fetchdf()

            if df.empty:
                print(f"The table '{table_name}' is empty or could not be loaded.")
                continue

            df = self.add_profanity_columns_parallel(df)

            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

            self.export_to_parquet(df, table_name)

        conn.close()

if __name__ == "__main__":
    profanity_detector = ProfanityDetection()
    profanity_detector.process_metadata_in_duckdb(database_file='my_database.duckdb')
