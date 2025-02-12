import pyodbc
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    try:
        # Step 1: Train and serialize the model
        print("Training the model...")
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        print("Serializing the model...")
        model_bytes = pickle.dumps(model)

        # Step 2: Connect to the Oracle database using pyodbc
        print("Connecting to the Oracle database...")
        connection_string = (
            "DRIVER={Oracle ODBC Driver};"
            "DBQ=hostname:port/service_name;"
            "UID=username;"
            "PWD=password;"
        )
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()

        # Step 3: Check if the table exists
        table_name = 'SAVED_MODELS'
        try:
            print(f"Checking if table '{table_name}' exists...")
            cursor.execute(f"""
                SELECT table_name
                FROM user_tables
                WHERE table_name = ?
            """, table_name.upper())
        except pyodbc.Error as e:
            print(f"Error checking table existence: {e}")
            raise

        # If the table does not exist, create it
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist. Creating it...")
            try:
                create_table_sql = f"""
                CREATE TABLE {table_name} (
                    model_id NUMBER GENERATED BY DEFAULT AS IDENTITY,
                    model_name VARCHAR2(100),
                    model_data BLOB,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_id)
                """
                cursor.execute(create_table_sql)
                connection.commit()
                print(f"Table '{table_name}' created successfully.")
            except pyodbc.Error as e:
                print(f"Error creating table: {e}")
                connection.rollback()
                raise
        else:
            print(f"Table '{table_name}' already exists.")

        # Step 4: Insert the serialized model into the database
        try:
            print("Inserting the model into the database...")
            insert_sql = f"""
            INSERT INTO {table_name} (model_name, model_data)
            VALUES (?, ?)
            """
            cursor.execute(insert_sql, 'RandomForestClassifier', model_bytes)
            connection.commit()
            print("Model saved to the database.")
        except pyodbc.Error as e:
            print(f"Error inserting model into the database: {e}")
            connection.rollback()
            raise

        # Step 5: Retrieve and deserialize the model
        try:
            print("Retrieving the model from the database...")
            cursor.execute(f"SELECT model_data FROM {table_name} WHERE model_name = ?", 'RandomForestClassifier')
            row = cursor.fetchone()

            if row:
                model_bytes = row[0]  # pyodbc returns the BLOB directly
                loaded_model = pickle.loads(model_bytes)

                # Test the loaded model
                predictions = loaded_model.predict(X_test)
                print("Predictions from the loaded model:", predictions)
            else:
                print("No model found in the database.")
        except pyodbc.Error as e:
            print(f"Error retrieving model from the database: {e}")
            raise
        except pickle.PickleError as e:
            print(f"Error deserializing the model: {e}")
            raise

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Step 6: Close the database connection
        print("Closing the database connection...")
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    main()
