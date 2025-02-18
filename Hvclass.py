import logging
import yaml
import pandas as pd
from impala.dbapi import connect
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HiveOperations:
    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)
        self.connection = None

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load nested YAML configuration file"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            raise

    def _create_connection(self):
        """Create Impala connection with SSL/LDAP"""
        try:
            self.connection = connect(
                host=self.config['impala']['host'],
                port=self.config['impala']['port'],
                user=self.config['impala']['user'],
                password=self.config['impala']['password'],
                use_ssl=True,
                auth_mechanism='LDAP'
            )
            logger.info("Connected to Impala")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def _close_connection(self):
        """Close Impala connection"""
        if self.connection:
            self.connection.close()
            logger.info("Connection closed")

    def execute_query(self, query: str):
        """Execute generic query"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            logger.debug(f"Executed: {query}")
            return cursor
        except Exception as e:
            logger.error(f"Query failed: {e}\nQuery: {query}")
            raise
        finally:
            if cursor:
                cursor.close()

    def check_table_exists(self, table: str) -> bool:
        """Check table existence"""
        try:
            query = f"SHOW TABLES LIKE '{table}'"
            result = self.execute_query(query).fetchall()
            return len(result) > 0
        except Exception as e:
            logger.error(f"Table check failed: {e}")
            raise

    def _build_schema(self, table_config: Dict[str, Any]) -> str:
        """Build SQL schema from nested YAML config"""
        columns = [f"{name} {dtype}" for name, dtype in table_config['columns'].items()]
        if 'partitioned_by' in table_config:
            partitions = [f"{name} {dtype}" for name, dtype in table_config['partitioned_by'].items()]
            return f"({', '.join(columns)}) PARTITIONED BY ({', '.join(partitions)})"
        return f"({', '.join(columns)})"

    def create_table(self, table: str, table_config: Dict[str, Any]):
        """Create table from YAML config"""
        try:
            if self.check_table_exists(table):
                logger.info(f"Table {table} already exists")
                return

            schema = self._build_schema(table_config)
            query = f"CREATE TABLE {table} {schema}"
            if 'file_format' in table_config:
                query += f" STORED AS {table_config['file_format']}"
            self.execute_query(query)
            logger.info(f"Created table {table}")
        except Exception as e:
            logger.error(f"Create table failed: {e}")
            raise

    def create_tables_from_yaml(self):
        """Create all tables from YAML configuration"""
        try:
            for table, config in self.config['tables'].items():
                self.create_table(table, config)
        except Exception as e:
            logger.error(f"Table creation from YAML failed: {e}")
            raise

    def incremental_update_from_dataframe(self, table: str, df: pd.DataFrame):
        """Dynamically adapt and update table from DataFrame"""
        try:
            # Get current table schema from YAML
            columns = list(self.config['tables'][table]['columns'].keys())
            
            # Filter and order dataframe columns to match table schema
            df = df[columns]
            
            # Generate INSERT statements
            for _, row in df.iterrows():
                values = ", ".join([f"'{value}'" if isinstance(value, str) else str(value) 
                                   for value in row.values])
                query = f"INSERT INTO TABLE {table} VALUES ({values})"
                self.execute_query(query)
            
            logger.info(f"Incrementally updated {table} with {len(df)} records")
        except Exception as e:
            logger.error(f"DataFrame update failed: {e}")
            raise

    def run_operations(self):
        """Main execution flow"""
        try:
            self._create_connection()
            self.create_tables_from_yaml()
            
            # Example DataFrame update
            sample_data = pd.DataFrame({
                'id': [1, 2],
                'name': ['Alice', 'Bob'],
                'created_at': ['2023-01-01', '2023-01-02']
            })
            self.incremental_update_from_dataframe('users', sample_data)
            
        finally:
            self._close_connection()

# Example config.yaml structure
"""
impala:
  host: impala.example.com
  port: 21050
  user: admin
  password: securepassword

tables:
  users:
    columns:
      id: INT
      name: STRING
      created_at: TIMESTAMP
    partitioned_by:
      event_date: STRING
    file_format: PARQUET
  
  transactions:
    columns:
      tx_id: BIGINT
      user_id: INT
      amount: DOUBLE
      tx_time: TIMESTAMP
    partitioned_by:
      tx_date: STRING
    file_format: AVRO
"""

if __name__ == "__main__":
    try:
        hive_ops = HiveOperations('config.yaml')
        hive_ops.run_operations()
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


import logging
import getpass
from typing import Dict, Any, Optional
from impala.dbapi import connect
from impala.error import Error as ImpalaError
import yaml
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('impala_operations.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class YamlHandler:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.config = self._load_yaml()
        self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        try:
            with open(self.yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading YAML file: {str(e)}")
            raise

    def _validate_config(self):
        required_keys = ['impala_connection', 'tables']
        if not all(key in self.config for key in required_keys):
            raise ValueError("Missing required sections in YAML config")

class ImpalaClient:
    def __init__(self, yaml_handler: YamlHandler):
        self.config = yaml_handler.config
        self.conn = None
        self.cursor = None

    def _get_connection(self, password: str):
        conn_config = self.config['impala_connection'].copy()
        conn_config['password'] = password
        conn_config['auth_mechanism'] = 'LDAP'
        
        try:
            self.conn = connect(**conn_config)
            self.cursor = self.conn.cursor()
            logger.info("Successfully connected to Impala")
        except ImpalaError as e:
            logger.error(f"Connection failed: {str(e)}")
            raise

    def connect(self):
        password = getpass.getpass("Enter LDAP password: ")
        self._get_connection(password)

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Connection closed")

    def execute_query(self, query: str, parameters: Optional[dict] = None):
        try:
            logger.debug(f"Executing query: {query}")
            self.cursor.execute(query, parameters)
            self.conn.commit()
        except ImpalaError as e:
            logger.error(f"Query failed: {str(e)}")
            self.conn.rollback()
            raise

    def table_exists(self, table_name: str) -> bool:
        try:
            self.cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            return bool(self.cursor.fetchone())
        except ImpalaError as e:
            logger.error(f"Table check failed: {str(e)}")
            return False

    def create_table(self, table_name: str, columns: list, partitions: list = None):
        base_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
        columns_str = ", ".join([f"{col['name']} {col['type']}" for col in columns])
        partition_str = ""
        
        if partitions:
            partition_str = " PARTITIONED BY (" + ", ".join(
                [f"{part['name']} {part['type']}" for part in partitions]
            ) + ")"
            
        query = base_query + columns_str + ")" + partition_str
        self.execute_query(query)
        logger.info(f"Table {table_name} created/validated")

    def create_tables_from_config(self):
        for table_name, table_config in self.config['tables'].items():
            if not self.table_exists(table_name):
                self.create_table(
                    table_name,
                    table_config['columns'],
                    table_config.get('partitions')
                )

    def _convert_pandas_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        return df

    def incremental_update(self, table_name: str, df: pd.DataFrame):
        if table_name not in self.config['tables']:
            raise ValueError(f"Table {table_name} not in config")
            
        if not self.config['tables'][table_name].get('incremental', False):
            self._full_refresh_update(table_name, df)
            return

        try:
            df = self._convert_pandas_dtypes(df)
            columns = [col['name'] for col in self.config['tables'][table_name]['columns']]
            
            # Dynamic upsert logic
            temp_table = f"temp_{table_name}"
            self.create_table(temp_table, self.config['tables'][table_name]['columns'])
            
            # Insert data into temp table
            placeholders = ", ".join(["%s"] * len(columns))
            insert_query = f"INSERT INTO {temp_table} VALUES ({placeholders})"
            
            with self.conn.cursor() as temp_cursor:
                for _, row in df.iterrows():
                    temp_cursor.execute(insert_query, tuple(row))
            
            # Merge data using dynamic column matching
            update_clause = ", ".join([f"t1.{col} = t2.{col}" for col in columns])
            on_clause = " AND ".join([f"t1.{col} = t2.{col}" for col in columns])
            
            merge_query = f"""
                MERGE INTO {table_name} t1
                USING {temp_table} t2
                ON {on_clause}
                WHEN MATCHED THEN UPDATE SET {update_clause}
                WHEN NOT MATCHED THEN INSERT VALUES ({", ".join([f"t2.{col}" for col in columns])})
            """
            
            self.execute_query(merge_query)
            self.execute_query(f"DROP TABLE {temp_table}")
            logger.info(f"Incrementally updated table {table_name}")
            
        except ImpalaError as e:
            logger.error(f"Incremental update failed: {str(e)}")
            raise

    def _full_refresh_update(self, table_name: str, df: pd.DataFrame):
        try:
            self.execute_query(f"TRUNCATE TABLE {table_name}")
            df = self._convert_pandas_dtypes(df)
            placeholders = ", ".join(["%s"] * len(df.columns))
            insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
            
            with self.conn.cursor() as temp_cursor:
                for _, row in df.iterrows():
                    temp_cursor.execute(insert_query, tuple(row))
            
            logger.info(f"Full refresh completed for {table_name}")
        except ImpalaError as e:
            logger.error(f"Full refresh failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        yaml_handler = YamlHandler("config.yaml")
        client = ImpalaClient(yaml_handler)
        client.connect()
        
        # Create tables from YAML config
        client.create_tables_from_config()
        
        # Example incremental update
        sample_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'event_date': pd.to_datetime(['2023-01-01', '2023-01-02'])
        })
        
        client.incremental_update('table1', sample_df)
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
    finally:
        client.close()



import pytest
from unittest.mock import Mock, patch, MagicMock
from impala_client import YamlHandler, ImpalaClient
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import yaml
import logging
from impala.error import Error as ImpalaError

@pytest.fixture
def temp_config_file(tmp_path):
    config = {
        'impala_connection': {
            'host': 'test-host',
            'port': 21050,
            'use_ssl': True,
            'ldap_username': 'test-user',
            'ssl_ca_cert': '/fake/path',
            'timeout': 3600
        },
        'tables': {
            'test_table': {
                'columns': [
                    {'name': 'id', 'type': 'INT'},
                    {'name': 'ts', 'type': 'TIMESTAMP'}
                ],
                'partitions': [
                    {'name': 'event_date', 'type': 'TIMESTAMP'}
                ],
                'incremental': True
            }
        }
    }
    file_path = tmp_path / "test_config.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(config, f)
    return file_path

@pytest.fixture
def yaml_handler(temp_config_file):
    return YamlHandler(temp_config_file)

@pytest.fixture
def impala_client(yaml_handler):
    client = ImpalaClient(yaml_handler)
    client.conn = MagicMock()
    client.cursor = MagicMock()
    return client

def test_yaml_handler_loading(temp_config_file):
    handler = YamlHandler(temp_config_file)
    assert 'impala_connection' in handler.config
    assert 'tables' in handler.config
    assert handler.config['tables']['test_table']['columns'][0]['name'] == 'id'

def test_yaml_handler_missing_sections(tmp_path):
    bad_config = {'wrong_section': {}}
    file_path = tmp_path / "bad_config.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(bad_config, f)
    
    with pytest.raises(ValueError):
        YamlHandler(file_path)

@patch('getpass.getpass')
def test_connection_success(mock_getpass, yaml_handler):
    mock_getpass.return_value = 'test-pass'
    client = ImpalaClient(yaml_handler)
    
    with patch('impala.dbapi.connect') as mock_connect:
        mock_connect.return_value = MagicMock()
        client.connect()
        
    mock_connect.assert_called_once_with(
        host='test-host',
        port=21050,
        use_ssl=True,
        ldap_username='test-user',
        password='test-pass',
        auth_mechanism='LDAP',
        ssl_ca_cert='/fake/path',
        timeout=3600
    )
    assert client.conn is not None

def test_table_exists_true(impala_client):
    impala_client.cursor.fetchone.return_value = ['test_table']
    assert impala_client.table_exists('test_table') is True
    impala_client.cursor.execute.assert_called_with("SHOW TABLES LIKE 'test_table'")

def test_table_exists_false(impala_client):
    impala_client.cursor.fetchone.return_value = None
    assert impala_client.table_exists('non_existent') is False

def test_create_table_with_partitions(impala_client):
    table_config = {
        'columns': [{'name': 'id', 'type': 'INT'}],
        'partitions': [{'name': 'dt', 'type': 'STRING'}]
    }
    
    impala_client.create_table('test_table', **table_config)
    
    expected_query = ("CREATE TABLE IF NOT EXISTS test_table (id INT) "
                      "PARTITIONED BY (dt STRING)")
    impala_client.cursor.execute.assert_called_with(expected_query, None)

def test_incremental_update_happy_path(impala_client):
    test_df = pd.DataFrame({
        'id': [1, 2],
        'ts': pd.to_datetime(['2023-01-01', '2023-01-02'])
    })
    
    impala_client.incremental_update('test_table', test_df)
    
    # Verify temp table creation
    create_call = impala_client.cursor.execute.call_args_list[0]
    assert "CREATE TABLE IF NOT EXISTS temp_test_table" in create_call[0][0]
    
    # Verify merge query
    merge_call = impala_client.cursor.execute.call_args_list[-2]
    assert "MERGE INTO test_table" in merge_call[0][0]
    assert "DROP TABLE temp_test_table" in impala_client.cursor.execute.call_args_list[-1][0][0]

def test_incremental_update_error_handling(impala_client, caplog):
    impala_client.cursor.execute.side_effect = ImpalaError("Test error")
    test_df = pd.DataFrame({'id': [1], 'ts': [pd.Timestamp.now()]})
    
    with pytest.raises(ImpalaError):
        impala_client.incremental_update('test_table', test_df)
    
    assert "Incremental update failed" in caplog.text
    impala_client.conn.rollback.assert_called()

def test_pandas_datetime_conversion():
    client = ImpalaClient(MagicMock())
    test_df = pd.DataFrame({
        'dt': pd.to_datetime(['2023-01-01', '2023-01-02'])
    })
    converted_df = client._convert_pandas_dtypes(test_df.copy())
    
    assert not is_datetime64_any_dtype(converted_df['dt'])
    assert converted_df['dt'].iloc[0] == '2023-01-01 00:00:00'

def test_full_refresh_update(impala_client):
    impala_client.config['tables']['test_table']['incremental'] = False
    test_df = pd.DataFrame({'id': [1], 'ts': [pd.Timestamp.now()]})
    
    impala_client.incremental_update('test_table', test_df)
    
    impala_client.cursor.execute.assert_any_call("TRUNCATE TABLE test_table")
    assert "INSERT INTO test_table" in impala_client.cursor.execute.call_args_list[-1][0][0]

def test_close_connection(impala_client):
    impala_client.close()
    impala_client.cursor.close.assert_called_once()
    impala_client.conn.close.assert_called_once()

@patch('impala_client.logging.getLogger')
def test_main_execution(mock_logger, yaml_handler, impala_client):
    with patch('impala_client.YamlHandler') as mock_yaml, \
         patch('impala_client.ImpalaClient') as mock_client, \
         patch('pandas.DataFrame') as mock_df:
        
        mock_yaml.return_value = yaml_handler
        mock_client.return_value = impala_client
        mock_df.return_value = pd.DataFrame()
        
        # Run the __main__ block
        with patch('impala_client.__name__', '__main__'):
            import impala_client
            impala_client
        
        mock_client.return_value.create_tables_from_config.assert_called_once()
        mock_client.return_value.incremental_update.assert_called_once()

def test_config_validation_error(tmp_path):
    bad_config = {'wrong_section': {}}
    file_path = tmp_path / "bad_config.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(bad_config, f)
    
    with pytest.raises(ValueError) as excinfo:
        YamlHandler(file_path)
    
    assert "Missing required sections" in str(excinfo.value)
