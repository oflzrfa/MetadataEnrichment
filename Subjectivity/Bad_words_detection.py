import os
import duckdb
import pandas as pd
import string

class BadWordsDetection:
    def __init__(self, lexicon_path='BadWords/profanity_en.csv', output_folder='BadWords_metadata/'):
        """
        Initialize the detector with a path to the lexicon of bad words and the output folder.
        """
        self.bad_words_set = self.load_bad_words(lexicon_path)
        self.output_folder = output_folder
        self.ensure_output_folder_exists()

    def load_bad_words(self, lexicon_path):
        """
        Load bad words from a CSV file into a set. Assumes bad words are in a column named 'text'.
        """
        try:
            bad_words_df = pd.read_csv(lexicon_path, usecols=['text'])  # Read only the 'text' column
            return set(bad_words_df['text'].str.lower().str.strip())
        except FileNotFoundError:
            print(f"Error: Bad words lexicon not found at {lexicon_path}")
            return set()
        except KeyError:
            print(f"Error: 'text' column not found in the CSV file at {lexicon_path}")
            return set()

    def ensure_output_folder_exists(self):
        """
        Ensure the output folder exists. If not, create it.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def detect_bad_words(self, text):
        """
        Detect bad words in a given text. Returns a binary flag indicating presence.
        """
        if not text: 
            return 0  
        

        text = text.translate(str.maketrans('', '', string.punctuation))

        words = text.lower().split() 
        return int(any(word in self.bad_words_set for word in words))  # Binary flag (1 if bad words found)

    def add_bad_words_column(self, df):
        """
        Add a binary column 'contains_bad_word' to indicate bad words presence in 'cleaned_text'.
        """
        if 'cleaned_text' not in df.columns:
            print("No 'cleaned_text' column found in DataFrame.")
            return df
        df['contains_bad_word'] = df['cleaned_text'].apply(self.detect_bad_words)
        return df

    def export_to_parquet(self, df, table_name):
        """
        Export the processed DataFrame to a Parquet file in the output folder.
        """
        output_file = os.path.join(self.output_folder, f"{table_name}_bad_words_metadata.parquet")
        df.to_parquet(output_file)
        print(f"Exported processed table '{table_name}' to Parquet: {output_file}")

    def update_duckdb_table(self, df, table_name, database_file='my_database.duckdb'):
        """
        Update the DuckDB table with the new column 'contains_bad_word'.
        """
        conn = duckdb.connect(database_file)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")  
        conn.close()
        print(f"Updated table '{table_name}' in DuckDB.")

    def process_metadata_in_duckdb(self, database_file='my_database.duckdb'):
        """
        Process all tables in the DuckDB database, add a 'contains_bad_word' column,
        and save the result as Parquet files locally or update DuckDB.
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


            df = self.add_bad_words_column(df)
            self.export_to_parquet(df, table_name)
            self.update_duckdb_table(df, table_name, database_file)

        conn.close()

if __name__ == "__main__":
    bad_words_detector = BadWordsDetection()
    bad_words_detector.process_metadata_in_duckdb(database_file='my_database.duckdb')





