import duckdb

conn = duckdb.connect('my_database.duckdb')

tables = conn.execute("SHOW TABLES").fetchdf()

print(tables)
conn.close()




