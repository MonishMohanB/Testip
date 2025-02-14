from pyhive import hive
import pandas as pd
import uuid
import time

# Hive connection parameters
hive_host = 'your_hive_host'
hive_port = 10000
hive_username = 'your_username'
hive_password = 'your_password'
hive_database = 'your_database'
hive_table = 'your_table'

# Function to connect to Hive
def connect_to_hive():
    conn = hive.Connection(
        host=hive_host,
        port=hive_port,
        username=hive_username,
        password=hive_password,
        database=hive_database,
        auth='CUSTOM'
    )
    return conn

# Function to insert DataFrame rows incrementally with a random ID
def insert_dataframe_incrementally(conn, df):
    cursor = conn.cursor()
    
    # Prepare the insert query
    insert_query = f"INSERT INTO TABLE {hive_table} VALUES (%s, %s, %s)"  # Adjust the query based on your table schema
    
    # Insert rows one by one
    for index, row in df.iterrows():
        # Generate a random UUID for the ID column
        random_id = str(uuid.uuid4())
        
        # Prepare the row data with the random ID
        row_data = (random_id, row['name'], row['age'])  # Adjust columns based on your DataFrame and table schema
        
        # Execute the insert query
        cursor.execute(insert_query, row_data)
        print(f"Inserted row: {row_data}")
        time.sleep(1)  # Optional: Add delay between inserts

    cursor.close()

# Example DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
}
df = pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    try:
        # Connect to Hive
        connection = connect_to_hive()
        print("Connected to Hive successfully!")
        
        # Insert DataFrame rows incrementally
        insert_dataframe_incrementally(connection, df)
        
        # Close the connection
        connection.close()
        print("Connection closed.")
    except Exception as e:
        print(f"An error occurred: {e}")
