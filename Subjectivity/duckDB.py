import duckdb

# Connect to the DuckDB database
conn = duckdb.connect('my_database.duckdb')

# Query to list all tables
tables = conn.execute("SHOW TABLES").fetchdf()

# Display the list of tables
print(tables)

# Close the connection
conn.close()




