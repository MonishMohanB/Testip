import pytest
from unittest.mock import MagicMock
from hive_operations import (
    create_hive_connection,
    close_hive_connection,
    execute_hive_query,
    check_table_exists,
    create_table,
    update_table,
)

# Mock Hive connection and cursor
@pytest.fixture
def mock_hive_connection(mocker):
    mock_conn = mocker.patch('pyhive.hive.Connection', autospec=True)
    mock_cursor = MagicMock()
    mock_conn.return_value.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor

def test_create_hive_connection(mock_hive_connection):
    """
    Test the `create_hive_connection` function.
    """
    mock_conn, _ = mock_hive_connection

    # Call the function
    conn = create_hive_connection()

    # Assertions
    mock_conn.assert_called_once_with(
        host='localhost',
        port=10000,
        username='hiveuser',
        database='default'
    )
    assert conn == mock_conn.return_value

def test_close_hive_connection(mock_hive_connection):
    """
    Test the `close_hive_connection` function.
    """
    mock_conn, _ = mock_hive_connection

    # Call the function
    close_hive_connection(mock_conn.return_value)

    # Assertions
    mock_conn.return_value.close.assert_called_once()

def test_execute_hive_query(mock_hive_connection):
    """
    Test the `execute_hive_query` function.
    """
    mock_conn, mock_cursor = mock_hive_connection
    test_query = "SELECT * FROM sample_table"

    # Call the function
    cursor = execute_hive_query(mock_conn.return_value, test_query)

    # Assertions
    mock_cursor.execute.assert_called_once_with(test_query)
    assert cursor == mock_cursor

def test_check_table_exists(mock_hive_connection):
    """
    Test the `check_table_exists` function.
    """
    mock_conn, mock_cursor = mock_hive_connection
    table_name = "sample_table"

    # Mock the fetchall result
    mock_cursor.fetchall.return_value = [(table_name,)]

    # Call the function
    result = check_table_exists(mock_conn.return_value, table_name)

    # Assertions
    mock_cursor.execute.assert_called_once_with(f"SHOW TABLES LIKE '{table_name}'")
    assert result is True

def test_create_table(mock_hive_connection):
    """
    Test the `create_table` function.
    """
    mock_conn, mock_cursor = mock_hive_connection
    table_name = "sample_table"

    # Call the function
    create_table(mock_conn.return_value, table_name)

    # Assertions
    expected_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT,
        name STRING,
        age INT
    )
    """
    mock_cursor.execute.assert_called_once_with(expected_query.strip())

def test_update_table(mock_hive_connection):
    """
    Test the `update_table` function.
    """
    mock_conn, mock_cursor = mock_hive_connection
    table_name = "sample_table"

    # Call the function
    update_table(mock_conn.return_value, table_name)

    # Assertions
    expected_query = f"""
    ALTER TABLE {table_name} ADD COLUMNS (new_column STRING COMMENT 'New column added')
    """
    mock_cursor.execute.assert_called_once_with(expected_query.strip())

def test_main_function(mocker, mock_hive_connection):
    """
    Test the `main` function.
    """
    mock_conn, mock_cursor = mock_hive_connection
    mocker.patch('hive_operations.check_table_exists', return_value=True)
    mocker.patch('hive_operations.update_table')

    # Call the main function
    from hive_operations import main
    main()

    # Assertions
    mock_conn.assert_called_once()
    mock_cursor.execute.assert_called()
    mock_conn.return_value.close.assert_called_once()
