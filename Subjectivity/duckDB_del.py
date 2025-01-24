import duckdb

# Initialize DuckDB connection (in-memory or persistent database)
conn = duckdb.connect(":memory:")  # Use ':memory:' for an in-memory database or a file path for persistence.

# Get the list of all tables in the current database
tables_in_db = conn.execute("PRAGMA show_tables;").fetchall()

# Drop each table
for table in tables_in_db:
    table_name = table[0]
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")

# Optionally, print confirmation message
print("All tables have been dropped.")
# Connect to the DuckDB database
conn = duckdb.connect('my_database.duckdb')

# Query to drop a specific table
table_name = 'updated_metadata_2_BW_detection_4'  # Replace with the table you want to drop
conn.execute(f"DROP TABLE IF EXISTS {table_name}")

# Close the connection
conn.close()

#print(f"Table {table_name} has been dropped.")
import duckdb

def drop_all_tables(database_file='my_database.duckdb'):
    # Connect to the DuckDB database
    conn = duckdb.connect(database_file)
    
    # Retrieve all table names from the database
    tables = conn.execute("SHOW TABLES").fetchdf()['name'].tolist()

    if not tables:
        print("No tables found in the DuckDB database.")
    else:
        # Loop through the tables and drop each one
        for table in tables:
            print(f"Dropping table: {table}")
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    
    # Close the connection
    conn.close()

# Run the function to drop all tables
if __name__ == "__main__":
    drop_all_tables(database_file='my_database.duckdb')
