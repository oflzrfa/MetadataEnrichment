import duckdb
conn = duckdb.connect('my_database.duckdb')
data = conn.execute("SELECT * FROM updated_metadata_1").fetchdf()

print(data)

schema = conn.execute("DESCRIBE updated_metadata_1").fetchdf()
print(schema)

conn.close()
