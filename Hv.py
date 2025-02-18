import pandas as pd
import yaml
from impala.dbapi import connect
from impala.util import as_pandas

class HiveConnection:
    def __init__(self, host='localhost', port=21050, username=None, password=None, 
                 database=None, use_ssl=True):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.use_ssl = use_ssl
        self.conn = None
        self.cursor = None
        self.tables = {}  # This will hold table schemas

    def create_connection(self):
        """Creates connection to Hive using Impala with LDAP authentication and auto SSL"""
        try:
            # Connect using SSL and LDAP Authentication
            self.conn = connect(
                host=self.host, 
                port=self.port, 
                user=self.username, 
                password=self.password,
                use_ssl=self.use_ssl,  # Enable SSL
                database=self.database,  # Optionally set the database
                auth_mechanism='LDAP'  # Set the authentication mechanism to LDAP
            )

            self.cursor = self.conn.cursor()
            print("Connection established successfully!")

        except Exception as e:
            print(f"Error establishing connection: {e}")

    def close_connection(self):
        """Closes the connection to Hive"""
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("Connection closed.")

    def execute_query(self, query):
        """Executes a given SQL query"""
        try:
            self.cursor.execute(query)
            print("Query executed successfully!")
        except Exception as e:
            print(f"Error executing query: {e}")

    def fetch_result(self):
        """Fetches query results"""
        return self.cursor.fetchall()

    def check_table_exists(self, table_name):
        """Checks if a table exists"""
        query = f"SHOW TABLES LIKE '{table_name}'"
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        return len(result) > 0

    def load_yaml_schema(self, yaml_file):
        """Loads table schemas from a YAML file"""
        try:
            # Load schema from YAML file
            with open(yaml_file, 'r') as file:
                schema = yaml.safe_load(file)
                self.tables = {table['table_name']: table for table in schema['tables']}
            print("YAML schema loaded successfully!")
        except Exception as e:
            print(f"Error loading YAML schema: {e}")

    def create_table_from_yaml(self, table_name):
        """Creates a table based on schema defined in YAML file by table name"""
        if table_name in self.tables:
            schema = self.tables[table_name]
            columns = schema['columns']
            
            # Build the CREATE TABLE query
            columns_definitions = ", ".join([f"{col['name']} {col['type']}" for col in columns])
            create_query = f"CREATE TABLE {table_name} ({columns_definitions})"
            
            # Execute the query to create the table
            self.execute_query(create_query)
            print(f"Table {table_name} created successfully from YAML schema!")
        else:
            print(f"Table {table_name} not found in YAML schema!")

    def add_column_to_table(self, table_name, column_name, column_type):
        """Adds a new column to the table if it doesn't exist in the schema"""
        alter_query = f"ALTER TABLE {table_name} ADD COLUMNS ({column_name} {column_type})"
        self.execute_query(alter_query)
        print(f"Added new column '{column_name}' of type '{column_type}' to table '{table_name}'.")

    def insert_dataframe(self, table_name, dataframe):
        """Inserts multiple records from a DataFrame into a table based on YAML schema"""
        if self.check_table_exists(table_name):
            # Check and modify the table schema if necessary
            if table_name in self.tables:
                columns = self.tables[table_name]['columns']
                column_names = [col['name'] for col in columns]
                
                # Check for extra columns in the DataFrame that are not in the YAML schema
                dataframe_columns = dataframe.columns.tolist()
                
                # Loop through DataFrame columns and add new ones if necessary
                for col in dataframe_columns:
                    if col not in column_names:
                        column_type = 'STRING'  # Default type for new columns
                        self.add_column_to_table(table_name, col, column_type)
                        columns.append({'name': col, 'type': column_type})  # Update local schema
                
                # Create the insert query
                insert_query = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(['%s'] * len(column_names))})"
                
                # Iterate over the DataFrame rows and insert the data
                for index, row in dataframe.iterrows():
                    values = tuple(row[col] for col in column_names)  # Match values to columns
                    self.cursor.execute(insert_query, values)
                print(f"Inserted {len(dataframe)} records into {table_name}")
            else:
                print(f"Table {table_name} schema not found in YAML!")
        else:
            print(f"Table {table_name} does not exist!")

# Example usage
if __name__ == "__main__":
    # Set your credentials here
    username = 'your_username'
    password = 'your_password'
    database = 'your_database'  # Specify the database to use

    # Create a connection with SSL and LDAP authentication
    hive = HiveConnection(host='your_host', port=21050, username=username, password=password, 
                          database=database, use_ssl=True)  # Set SSL to True
    hive.create_connection()

    # Load table schemas from the YAML file
    yaml_file = 'schema.yaml'  # Path to the YAML file
    hive.load_yaml_schema(yaml_file)

    # Allow user to choose the table they want to create
    selected_table = input("Enter the table name to create (e.g., test_table_1, test_table_2): ")
    hive.create_table_from_yaml(selected_table)

    # Create a sample DataFrame for inserting data
    data = {
        'id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Alice Johnson'],
        'age': [28, 35, 22],
        'extra_column': ['extra_1', 'extra_2', 'extra_3']  # New column not in YAML
    }
    df = pd.DataFrame(data)

    # Insert DataFrame data into the table
    hive.insert_dataframe(selected_table, df)

    # Fetch and print results
    fetch_query = f"SELECT * FROM {selected_table}"
    hive.execute_query(fetch_query)
    results = hive.fetch_result()
    print("Fetched Results:", results)

    # Close the connection
    hive.close_connection()




import pytest
from unittest.mock import MagicMock
from hive_connection import HiveConnection  # Assuming the class is in `hive_connection.py`
import pandas as pd


@pytest.fixture
def hive_connection(mocker):
    """Fixture to set up a mocked HiveConnection object."""
    # Mock the connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Mock the connection behavior
    mock_conn.cursor.return_value = mock_cursor
    mocker.patch('impala.dbapi.connect', return_value=mock_conn)
    
    # Create the HiveConnection object
    hive = HiveConnection(host="localhost", port=21050, username="test_user", password="test_password", 
                          database="default", use_ssl=True)
    hive.conn = mock_conn
    hive.cursor = mock_cursor
    
    # Mock load_yaml_schema to return a predefined schema
    mocker.patch.object(hive, 'load_yaml_schema', return_value=None)
    hive.tables = {
        'test_table_1': {
            'table_name': 'test_table_1',
            'columns': [
                {'name': 'id', 'type': 'INT'},
                {'name': 'name', 'type': 'STRING'},
                {'name': 'age', 'type': 'INT'}
            ]
        }
    }
    
    return hive


def test_check_table_exists(hive_connection):
    """Test the check_table_exists method."""
    hive_connection.cursor.fetchall.return_value = [("test_table_1",)]
    assert hive_connection.check_table_exists("test_table_1") is True
    
    hive_connection.cursor.fetchall.return_value = []
    assert hive_connection.check_table_exists("test_table_2") is False


def test_create_table_from_yaml(hive_connection, mocker):
    """Test creating a table from the YAML schema."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query

    # Test if the table is created successfully
    hive_connection.create_table_from_yaml('test_table_1')
    hive_connection.execute_query.assert_called_once_with(
        "CREATE TABLE test_table_1 (id INT, name STRING, age INT)"
    )


def test_add_column_to_table(hive_connection, mocker):
    """Test adding a column to an existing table."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query
    
    # Simulate adding a new column to the table
    hive_connection.add_column_to_table('test_table_1', 'email', 'STRING')
    
    hive_connection.execute_query.assert_called_once_with(
        "ALTER TABLE test_table_1 ADD COLUMNS (email STRING)"
    )


def test_insert_dataframe_with_new_columns(hive_connection, mocker):
    """Test inserting a DataFrame with new columns that don't exist in the table schema."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query
    mocker.patch.object(hive_connection, 'check_table_exists', return_value=True)  # Mock table check

    # Sample DataFrame with new columns not in YAML
    data = {
        'id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Alice Johnson'],
        'age': [28, 35, 22],
        'email': ['john@example.com', 'jane@example.com', 'alice@example.com']  # New column
    }
    df = pd.DataFrame(data)
    
    # Insert DataFrame into the table
    hive_connection.insert_dataframe('test_table_1', df)
    
    # Assert that the new column 'email' was added
    hive_connection.execute_query.assert_any_call("ALTER TABLE test_table_1 ADD COLUMNS (email STRING)")
    hive_connection.execute_query.assert_any_call(
        "INSERT INTO test_table_1 (id, name, age, email) VALUES (%s, %s, %s, %s)"
    )


def test_insert_dataframe_without_new_columns(hive_connection, mocker):
    """Test inserting a DataFrame where all columns already exist in the table schema."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query
    mocker.patch.object(hive_connection, 'check_table_exists', return_value=True)  # Mock table check

    # Sample DataFrame where all columns already exist in the YAML schema
    data = {
        'id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Alice Johnson'],
        'age': [28, 35, 22]
    }
    df = pd.DataFrame(data)
    
    # Insert DataFrame into the table
    hive_connection.insert_dataframe('test_table_1', df)
    
    # Assert no ALTER TABLE was called since no new columns were added
    hive_connection.execute_query.assert_not_called_with("ALTER TABLE test_table_1 ADD COLUMNS")
    hive_connection.execute_query.assert_any_call(
        "INSERT INTO test_table_1 (id, name, age) VALUES (%s, %s, %s)"
    )


def test_create_table_with_invalid_schema(hive_connection, mocker):
    """Test creating a table with an invalid or missing schema."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query
    
    # Pass a table name that doesn't exist in the YAML schema
    hive_connection.create_table_from_yaml('non_existent_table')
    
    # Ensure execute_query was not called
    hive_connection.execute_query.assert_not_called()


def test_fetch_result(hive_connection, mocker):
    """Test the fetch_result method."""
    hive_connection.cursor.fetchall.return_value = [("John Doe", 28), ("Jane Smith", 35)]
    
    result = hive_connection.fetch_result()
    assert result == [("John Doe", 28), ("Jane Smith", 35)]
    hive_connection.cursor.fetchall.assert_called_once()


if __name__ == "__main__":
    pytest.main()
    
