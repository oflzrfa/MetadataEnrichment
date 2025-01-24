import os
import re
import pandas as pd
import nltk
import spacy
import multiprocessing as mp
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import duckdb
from Parallel_Subjectivity_detection import SubjectivityDetection
from Parallel_Profanity_detection import ProfanityDetection
# Load spaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

class TextCleaner:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.stop_words = set(stopwords.words('english'))  

    def remove_irrelevant_sections(self, text):
        """
        Remove irrelevant sections such as navigation, headers, lists, logos, etc.
        """
        irrelevant_patterns = [
            r"(skip to main content|skip to content|Skip links|navigation|play|live|end of list|list \d+ of \d+|•|jump to content|aj-logo|Follow Al Jazeera|©|Al Jazeera Media Network|aj-logo|Features|Live|Copyright|\bLogo\b|social media|advertisement|Media|Events|News|Contact|FAQ|Cookies|Connect|with us|support|section|available|usernam|required|password)",  # Updated pattern with "Connect with us"
            r"\[.*\]",  # Remove content within square brackets (e.g., citation links)
            r"^\s*$",  # Remove empty lines
            r"list \d+ of \d+",  # Skip list headers
            r"^(?:[a-zA-Z]{1,2})$",  # Remove very short words (noise)
            r"^\d{4,}",  # Remove numbers that aren't part of meaningful text
            r"^[\d\w]+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$",  # Remove emails if any
            r"\d{1,2} \w+ \d{4}",  # Remove timestamps like "15 Nov 2023"
            r"^\s*—\s*$",  # Remove em dashes (if present in metadata)
        ]
        
        for pattern in irrelevant_patterns:
            text = re.sub(rf"{pattern}.*\n", "", text, flags=re.IGNORECASE)
        
        return text

    def clean_sentence(self, sentence):
        """
        Clean individual sentences by trimming excessive spaces and handling special characters.
        """
        sentence = sentence.strip()
        return sentence

    def clean_text(self, text):
        """
        Clean input text by:
        1. Removing irrelevant sections (e.g., navigation, footers).
        2. Removing stopwords, very short words, extra spaces.
        3. Normalizing the text into `cleaned_text`.
        """
        if not isinstance(text, str):
            return ""
        
        text = self.remove_irrelevant_sections(text)

        words_list = word_tokenize(text)

        filtered_words = [
            word.lower() for word in words_list
            if word.lower() not in self.stop_words and len(word) > 2 and word.isalpha() 
        ]
        

        cleaned_text = " ".join(filtered_words)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    def clean_web_document(self, text):
        """
        Clean input text by:
        1. Removing irrelevant sections (e.g., navigation, footers).
        2. Tokenizing the text into meaningful sentences.
        3. Cleaning sentences and preserving context.
        """
        if not isinstance(text, str):
            return ""
        text = self.remove_irrelevant_sections(text)

        sentences = sent_tokenize(text)

        meaningful_sentences = []
        for sentence in sentences:
            sentence_cleaned = self.clean_sentence(sentence)
            if sentence_cleaned:
                meaningful_sentences.append(sentence_cleaned)

        web_document = " ".join(meaningful_sentences)

        return web_document

    def extract_named_entities(self, text):
        """
        Extract named entities (persons, locations, organizations) from the cleaned text.
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "ORG"]:  
                entities.append((ent.text, ent.label_))
        return entities

    def process_text(self, text):
        """
        Complete processing pipeline:
        1. Clean the text by removing irrelevant sections and formatting.
        2. Extract named entities.
        3. Further clean the text by removing unwanted characters (special chars, numbers, emoticons).
        """
        web_document = self.clean_web_document(text)
        cleaned_text = self.clean_text(web_document)
        named_entities = self.extract_named_entities(cleaned_text)
        
        return web_document, cleaned_text, named_entities


class ParquetProcessor:
    def __init__(self, folder_path='Resources/'):
        self.folder_path = folder_path
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.english_words = set(words.words()) 
        self.text_cleaner = TextCleaner(nlp)

        self.counter = 1 

    def read_parquet_file(self, file_path):
        """Reads a single Parquet file from the given file path using DuckDB."""
        conn = duckdb.connect(database='my_database.duckdb', read_only=False)
        query = f"SELECT * FROM read_parquet('{file_path}')"
        df = conn.execute(query).fetchdf()
        conn.close()
        return df

    def parallel_clean_text(self, texts):
        """Clean text using multiprocessing."""
        with mp.Pool(mp.cpu_count()) as pool:
            cleaned_texts = pool.map(self.process_single_text, texts)
        return cleaned_texts

    def process_single_text(self, text):
        """Process text through TextCleaner and extract entities."""
        web_document, cleaned_text, named_entities = self.text_cleaner.process_text(text)
        return web_document, cleaned_text, named_entities

    def process_and_save(self):
        """Processes all Parquet files and saves the cleaned data to DuckDB."""
        for subdir, _, files in os.walk(self.folder_path):
            parquet_files = [f for f in files if f.endswith('.parquet')]

            for parquet_file in parquet_files:
                file_path = os.path.join(subdir, parquet_file)
                df = self.read_parquet_file(file_path)

                if df.empty:
                    print(f"No data found in {parquet_file}")
                    continue

                print(f"Processing {parquet_file}...")

                cleaned_data = self.parallel_clean_text(df['plain_text'].tolist())

                web_documents, cleaned_texts, named_entities_list = zip(*cleaned_data)

                df['web_document'] = web_documents
                df['cleaned_text'] = cleaned_texts
                df['named_entities'] = named_entities_list

                print(df[['id', 'plain_text', 'web_document', 'cleaned_text', 'named_entities']])


                table_name = f"updated_metadata_{self.counter}"
                conn = duckdb.connect(database='my_database.duckdb', read_only=False)

                try:
                    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
                    conn.commit()
                except Exception as e:
                    print(f"Error creating table {table_name}: {e}")
                    conn.close()
                    continue
                
                conn.close()

                self.counter += 1


if __name__ == "__main__":
    processor = ParquetProcessor(folder_path='Resources/')
    processor.process_and_save()


