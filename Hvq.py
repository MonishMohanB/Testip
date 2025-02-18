import pandas as pd
import yaml
import logging
from impala.dbapi import connect
from impala.util import as_pandas


# Set up logging
logger = logging.getLogger('HiveConnection')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Custom exceptions
class HiveConnectionError(Exception):
    """Exception raised for errors in the Hive connection."""
    pass


class TableNotFoundError(Exception):
    """Exception raised when the table is not found in the database."""
    pass


class ColumnNotFoundError(Exception):
    """Exception raised when a column is not found in the table schema."""
    pass


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
            logger.info(f"Connecting to Hive at {self.host}:{self.port} with SSL: {self.use_ssl}")
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
            logger.info("Connection established successfully!")

        except Exception as e:
            logger.error(f"Error establishing connection: {e}")
            raise HiveConnectionError(f"Failed to connect to Hive: {e}")

    def close_connection(self):
        """Closes the connection to Hive"""
        try:
            if self.conn:
                self.cursor.close()
                self.conn.close()
                logger.info("Connection closed.")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
            raise HiveConnectionError(f"Failed to close connection: {e}")

    def execute_query(self, query):
        """Executes a given SQL query"""
        try:
            logger.debug(f"Executing query: {query}")
            self.cursor.execute(query)
            logger.info("Query executed successfully!")
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise Exception(f"Failed to execute query: {e}")

    def fetch_result(self):
        """Fetches query results"""
        try:
            result = self.cursor.fetchall()
            logger.debug(f"Fetched result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error fetching results: {e}")
            raise Exception(f"Failed to fetch results: {e}")

    def check_table_exists(self, table_name):
        """Checks if a table exists"""
        query = f"SHOW TABLES LIKE '{table_name}'"
        try:
            logger.debug(f"Checking if table '{table_name}' exists.")
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            if len(result) > 0:
                logger.info(f"Table '{table_name}' exists.")
                return True
            else:
                logger.warning(f"Table '{table_name}' does not exist.")
                raise TableNotFoundError(f"Table '{table_name}' not found in the database.")
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            raise TableNotFoundError(f"Failed to check if table exists: {e}")

    def load_yaml_schema(self, yaml_file):
        """Loads table schemas from a YAML file"""
        try:
            # Load schema from YAML file
            with open(yaml_file, 'r') as file:
                schema = yaml.safe_load(file)
                self.tables = {table['table_name']: table for table in schema['tables']}
            logger.info("YAML schema loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading YAML schema: {e}")
            raise Exception(f"Failed to load YAML schema: {e}")

    def create_table_from_yaml(self, table_name):
        """Creates a table based on schema defined in YAML file by table name"""
        try:
            if table_name in self.tables:
                schema = self.tables[table_name]
                columns = schema['columns']
                
                # Build the CREATE TABLE query
                columns_definitions = ", ".join([f"{col['name']} {col['type']}" for col in columns])
                create_query = f"CREATE TABLE {table_name} ({columns_definitions})"
                
                # Execute the query to create the table
                self.execute_query(create_query)
                logger.info(f"Table {table_name} created successfully from YAML schema!")
            else:
                logger.warning(f"Table '{table_name}' not found in YAML schema!")
                raise TableNotFoundError(f"Table '{table_name}' not found in YAML schema!")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise Exception(f"Failed to create table: {e}")

    def add_column_to_table(self, table_name, column_name, column_type):
        """Adds a new column to the table if it doesn't exist in the schema"""
        try:
            alter_query = f"ALTER TABLE {table_name} ADD COLUMNS ({column_name} {column_type})"
            self.execute_query(alter_query)
            logger.info(f"Added new column '{column_name}' of type '{column_type}' to table '{table_name}'.")
        except Exception as e:
            logger.error(f"Error adding column to table: {e}")
            raise Exception(f"Failed to add column '{column_name}' to table: {e}")

    def insert_dataframe(self, table_name, dataframe):
        """Inserts multiple records from a DataFrame into a table based on YAML schema"""
        try:
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
                    logger.info(f"Inserted {len(dataframe)} records into {table_name}")
                else:
                    logger.warning(f"Table '{table_name}' schema not found in YAML!")
                    raise TableNotFoundError(f"Table '{table_name}' schema not found in YAML!")
            else:
                raise TableNotFoundError(f"Table '{table_name}' does not exist!")
        except Exception as e:
            logger.error(f"Error inserting DataFrame: {e}")
            raise Exception(f"Failed to insert DataFrame into table: {e}")


# Example usage
if __name__ == "__main__":
    # Set your credentials here
    username = 'your_username'
    password = 'your_password'
    database = 'your_database'  # Specify the database to use

    # Create a connection with SSL and LDAP authentication
    hive = HiveConnection(host='your_host', port=21050, username=username, password=password, 
                          database=database, use_ssl=True)  # Set SSL to True
    try:
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

    except Exception as e:
        logger.error(f"Error during Hive operations: {e}")
    finally:
        hive.close_connection()






import pytest
from unittest.mock import MagicMock
from hive_connection import HiveConnection, HiveConnectionError, TableNotFoundError
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


def test_create_connection_success(hive_connection, mocker):
    """Test successful Hive connection creation."""
    hive_connection.create_connection()
    hive_connection.conn.cursor.assert_called_once()
    assert hive_connection.cursor is not None


def test_create_connection_failure(hive_connection, mocker):
    """Test failure in Hive connection creation."""
    mocker.patch('impala.dbapi.connect', side_effect=Exception("Connection failed"))
    
    with pytest.raises(HiveConnectionError):
        hive_connection.create_connection()


def test_check_table_exists_success(hive_connection):
    """Test table existence check when table exists."""
    hive_connection.cursor.fetchall.return_value = [("test_table_1",)]
    assert hive_connection.check_table_exists("test_table_1") is True


def test_check_table_exists_failure(hive_connection):
    """Test table existence check when table does not exist."""
    hive_connection.cursor.fetchall.return_value = []
    with pytest.raises(TableNotFoundError):
        hive_connection.check_table_exists("non_existent_table")


def test_create_table_from_yaml_success(hive_connection, mocker):
    """Test creating a table from the YAML schema."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query
    hive_connection.create_table_from_yaml('test_table_1')
    hive_connection.execute_query.assert_called_once_with(
        "CREATE TABLE test_table_1 (id INT, name STRING, age INT)"
    )


def test_create_table_from_yaml_failure(hive_connection, mocker):
    """Test creating a table that doesn't exist in YAML schema."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query
    with pytest.raises(TableNotFoundError):
        hive_connection.create_table_from_yaml('non_existent_table')


def test_add_column_to_table_success(hive_connection, mocker):
    """Test adding a column to an existing table."""
    mocker.patch.object(hive_connection, 'execute_query')  # Mock execute_query
    hive_connection.add_column_to_table('test_table_1', 'email', 'STRING')
    hive_connection.execute_query.assert_called_once_with(
        "ALTER TABLE test_table_1 ADD COLUMNS (email STRING)"
    )


def test_add_column_to_table_failure(hive_connection, mocker):
    """Test failure when trying to add a column."""
    mocker.patch.object(hive_connection, 'execute_query', side_effect=Exception("Failed to add column"))
    with pytest.raises(Exception):
        hive_connection.add_column_to_table('test_table_1', 'email', 'STRING')


def test_insert_dataframe_with_new_columns(hive_connection, mocker):
    """Test inserting a DataFrame with new columns that are not in the table schema."""
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


def test_insert_dataframe_table_not_found(hive_connection, mocker):
    """Test inserting data into a non-existent table."""
    mocker.patch.object(hive_connection, 'check_table_exists', return_value=False)  # Mock table check failure
    
    # Sample DataFrame
    data = {
        'id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Alice Johnson'],
        'age': [28, 35, 22]
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(TableNotFoundError):
        hive_connection.insert_dataframe('non_existent_table', df)


def test_fetch_result(hive_connection, mocker):
    """Test the fetch_result method."""
    hive_connection.cursor.fetchall.return_value = [("John Doe", 28), ("Jane Smith", 35)]
    
    result = hive_connection.fetch_result()
    assert result == [("John Doe", 28), ("Jane Smith", 35)]
    hive_connection.cursor.fetchall.assert_called_once()


def test_connection_closing(hive_connection, mocker):
    """Test closing the connection."""
    hive_connection.close_connection()
    hive_connection.conn.close.assert_called_once()
    hive_connection.cursor.close.assert_called_once()


def test_create_connection_exception_logging(mocker):
    """Test logging and exception handling when connection fails."""
    mocker.patch('impala.dbapi.connect', side_effect=Exception("Connection failed"))
    with pytest.raises(HiveConnectionError):
        hive_connection.create_connection()


if __name__ == "__main__":
    pytest.main()
