import streamlit as st
import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_table_names(database_file='my_database.duckdb'):
    conn = duckdb.connect(database_file)
    query = "SHOW TABLES"
    tables = conn.execute(query).fetchdf()['name'].tolist()
    conn.close()
    return tables

def load_table_data(table_name, database_file='my_database.duckdb', max_rows=500):
    conn = duckdb.connect(database_file)
    query = f"SELECT * FROM {table_name} LIMIT {max_rows}"  # Limit to max_rows
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def plot_heatmap(df):
    """Generate a heatmap for the correlation matrix of numeric columns."""
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

def plot_count_chart(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column)
    st.pyplot(plt)

def plot_scatter_chart(df, column_x, column_y):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=column_x, y=column_y)
    st.pyplot(plt)

def main():
    st.title('Metadata Insights')
    
    tables = get_table_names()
    selected_table = st.selectbox('Select a table', tables)
    df = load_table_data(selected_table)

    if df.empty:
        st.write("No data found.")
    else:
        st.write(f"Showing data for table: {selected_table}")
        st.dataframe(df)
        
        st.subheader("Visualize Data")
        
        tab1, tab2, tab3 = st.tabs(["Count Plot", "Scatter Plot", "Heatmap"])

        with tab1:
            column = st.selectbox('Select the column for Count Plot', df.columns)
            if column:
                plot_count_chart(df, column)

        with tab2:
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            column_x = st.selectbox('Select the X-axis column for Scatter Plot', numeric_columns)
            column_y = st.selectbox('Select the Y-axis column for Scatter Plot', numeric_columns)

            if column_x and column_y:
                plot_scatter_chart(df, column_x, column_y)

        with tab3:
            st.subheader("Correlation Heatmap")
            plot_heatmap(df)

if __name__ == "__main__":
    main()

