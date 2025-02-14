from pyhive import hive
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

# Function to insert rows incrementally
def insert_rows_incrementally(conn, data):
    cursor = conn.cursor()
    
    # Prepare the insert query
    insert_query = f"INSERT INTO TABLE {hive_table} VALUES (%s, %s, %s)"  # Adjust the query based on your table schema
    
    # Insert rows one by one
    for row in data:
        cursor.execute(insert_query, row)
        print(f"Inserted row: {row}")
        time.sleep(1)  # Optional: Add delay between inserts

    cursor.close()

# Example data to insert
data_to_insert = [
    (1, 'Alice', 25),
    (2, 'Bob', 30),
    (3, 'Charlie', 35),
    # Add more rows as needed
]

# Main execution
if __name__ == "__main__":
    try:
        # Connect to Hive
        connection = connect_to_hive()
        print("Connected to Hive successfully!")
        
        # Insert rows incrementally
        insert_rows_incrementally(connection, data_to_insert)
        
        # Close the connection
        connection.close()
        print("Connection closed.")
    except Exception as e:
        print(f"An error occurred: {e}")
