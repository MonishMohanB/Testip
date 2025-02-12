from pyhive import hive
from TCLIService.ttypes import TOperationState
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hive connection parameters
HIVE_HOST = 'localhost'
HIVE_PORT = 10000
HIVE_DATABASE = 'default'
HIVE_USER = 'hiveuser'
HIVE_TABLE = 'sample_table'

def create_hive_connection():
    """
    Create a connection to the Hive database.

    Returns:
        connection (pyhive.hive.Connection): A connection object to the Hive database.

    Raises:
        Exception: If the connection to the Hive database fails.
    """
    try:
        conn = hive.Connection(
            host=HIVE_HOST,
            port=HIVE_PORT,
            username=HIVE_USER,
            database=HIVE_DATABASE
        )
        logger.info("Successfully connected to Hive.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Hive: {e}")
        raise

def close_hive_connection(conn):
    """
    Close the Hive database connection.

    Args:
        conn (pyhive.hive.Connection): The connection object to be closed.

    Raises:
        Exception: If an error occurs while closing the connection.
    """
    try:
        if conn:
            conn.close()
            logger.info("Hive connection closed.")
    except Exception as e:
        logger.error(f"Error while closing Hive connection: {e}")

def execute_hive_query(conn, query):
    """
    Execute a Hive query.

    Args:
        conn (pyhive.hive.Connection): The connection object to the Hive database.
        query (str): The SQL query to be executed.

    Returns:
        cursor (pyhive.hive.Cursor): A cursor object containing the query results.

    Raises:
        Exception: If the query execution fails.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        logger.info(f"Query executed successfully: {query}")
        return cursor
    except Exception as e:
        logger.error(f"Failed to execute query: {query}. Error: {e}")
        raise

def check_table_exists(conn, table_name):
    """
    Check if a table exists in the Hive database.

    Args:
        conn (pyhive.hive.Connection): The connection object to the Hive database.
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.

    Raises:
        Exception: If an error occurs while checking the table existence.
    """
    try:
        query = f"SHOW TABLES LIKE '{table_name}'"
        cursor = execute_hive_query(conn, query)
        result = cursor.fetchall()
        return len(result) > 0
    except Exception as e:
        logger.error(f"Error checking if table exists: {e}")
        raise

def create_table(conn, table_name):
    """
    Create a table in Hive if it does not exist.

    Args:
        conn (pyhive.hive.Connection): The connection object to the Hive database.
        table_name (str): The name of the table to create.

    Raises:
        Exception: If the table creation fails.
    """
    try:
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT,
            name STRING,
            age INT
        )
        """
        execute_hive_query(conn, query)
        logger.info(f"Table '{table_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        raise

def update_table(conn, table_name):
    """
    Update the table if it exists.

    Args:
        conn (pyhive.hive.Connection): The connection object to the Hive database.
        table_name (str): The name of the table to update.

    Raises:
        Exception: If the table update fails.
    """
    try:
        query = f"""
        ALTER TABLE {table_name} ADD COLUMNS (new_column STRING COMMENT 'New column added')
        """
        execute_hive_query(conn, query)
        logger.info(f"Table '{table_name}' updated successfully.")
    except Exception as e:
        logger.error(f"Failed to update table: {e}")
        raise

def main():
    """
    Main function to handle table creation or update.

    Steps:
        1. Create a connection to the Hive database.
        2. Check if the table exists.
        3. If the table exists, update it. Otherwise, create it.
        4. Close the connection to the Hive database.

    Raises:
        Exception: If an error occurs during the process.
    """
    conn = None
    try:
        # Create a connection to Hive
        conn = create_hive_connection()

        # Check if the table exists
        if check_table_exists(conn, HIVE_TABLE):
            logger.info(f"Table '{HIVE_TABLE}' already exists. Updating table...")
            update_table(conn, HIVE_TABLE)
        else:
            logger.info(f"Table '{HIVE_TABLE}' does not exist. Creating table...")
            create_table(conn, HIVE_TABLE)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Close the connection
        if conn:
            close_hive_connection(conn)

if __name__ == "__main__":
    main()
