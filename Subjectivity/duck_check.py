import duckdb

# Connect to the DuckDB database
conn = duckdb.connect('my_database.duckdb')

# Query to view all rows of the updated_metadata table
data = conn.execute("SELECT * FROM updated_metadata_1").fetchdf()

# Display the data
print(data)

# Query to view the schema of the table
schema = conn.execute("DESCRIBE updated_metadata_1").fetchdf()

# Display the schema
print(schema)


# Close the connection
conn.close()
