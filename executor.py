import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Executor:
    """
    Executor class to convert structured plans into executable SQL queries, execute them, and retrieve results.

    ...

    Attributes
    ----------
    connection_uri : str
        URI for database connection.
    connection : sqlalchemy.engine.base.Connection
        Database connection object.
    use_numpy : bool
        Flag to use numpy for data processing.
    use_pandas : bool
        Flag to use pandas for data processing.

    Methods
    -------
    connect(uri)
        Establish a database connection.
    disconnect()
        Close the database connection.
    generate_sql_query(plan)
        Convert a structured plan into an executable SQL query.
    execute_query(query, params)
        Execute an SQL query with parameters.
    get_results(query, params)
        Execute a query and return the results.
    """

    def __init__(self, uri: str, use_numpy: bool = True, use_pandas: bool = True):
        """
        Initialize the Executor with database connection details and data processing options.

        Parameters
        ----------
        uri : str
            URI for database connection.
        use_numpy : bool, optional
            Flag to use numpy for data processing, by default True.
        use_pandas : bool, optional
            Flag to use pandas for data processing, by default True.

        Raises
        ------
        ValueError
            If both use_numpy and use_pandas are set to False.
        """
        self.connection_uri = uri
        self.connection = None
        self.use_numpy = use_numpy
        self.use_pandas = use_pandas

        if not self.use_numpy and not self.use_pandas:
            raise ValueError("At least one of 'use_numpy' or 'use_pandas' must be True.")

    def connect(self):
        """
        Establish a database connection using the provided URI.

        Raises
        ------
        RuntimeError
            If unable to establish a database connection.
        """
        try:
            self.connection = create_engine(self.connection_uri).connect()
            logger.info("Database connection established.")
        except SQLAlchemyError as e:
            logger.error(f"Error connecting to database: {e}")
            raise RuntimeError("Failed to connect to database.")

    def disconnect(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")

    def generate_sql_query(self, plan: dict) -> str:
        """
        Convert a structured plan into an executable SQL query.

        Parameters
        ----------
        plan : dict
            Structured plan containing query information.

        Returns
        -------
        str
            Executable SQL query.

        Raises
        ------
        ValueError
            If the plan is missing required fields.
        """
        if not isinstance(plan, dict) or "columns" not in plan or "table" not in plan:
            raise ValueError("Invalid plan format. Plan must be a dict with 'columns' and 'table' fields.")

        columns = plan["columns"]
        table = plan["table"]

        if not isinstance(columns, list) or not isinstance(table, str):
            raise ValueError("Invalid plan format. 'columns' must be a list and 'table' must be a string.")

        select_clause = ", ".join(columns)
        sql_query = f"SELECT {select_clause} FROM {table}"

        logger.debug(f"Generated SQL query: {sql_query}")
        return sql_query

    def execute_query(self, query: str, params: dict = None) -> None:
        """
        Execute an SQL query with optional parameters.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : dict, optional
            Parameters for the query, by default None.

        Raises
        ------
        ValueError
            If the query is not a string.
        SQLAlchemyError
            If an error occurs during query execution.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        try:
            with self.connection.begin() as conn:
                conn.execute(text(query), params or {})
            logger.info("Query executed successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_results(self, query: str, params: dict = None) -> np.array or pd.DataFrame:
        """
        Execute a query and return the results as a NumPy array or Pandas DataFrame.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : dict, optional
            Parameters for the query, by default None.

        Returns
        -------
        np.array or pd.DataFrame
            Query results.

        Raises
        ------
        ValueError
            If the query is not a string.
        SQLAlchemyError
            If an error occurs during query execution.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        try:
            with self.connection.begin() as conn:
                result = conn.execute(text(query), params or {})

                if self.use_numpy:
                    return np.array(result.fetchall())
                elif self.use_pandas:
                    return pd.DataFrame(result.fetchall(), columns=result.keys())
                else:
                    return result.fetchall()

        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise

# Example usage
if __name__ == "__main__":
    executor = Executor("sqlite:///example.db")
    executor.connect()

    # Example structured plan
    plan = {
        "table": "my_table",
        "columns": ["column1", "column2", "column3"]
    }

    sql_query = executor.generate_sql_query(plan)
    executor.execute_query(sql_query)

    results = executor.get_results(sql_query)
    print(results)

    executor.disconnect()