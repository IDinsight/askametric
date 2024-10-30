from functools import wraps
from typing import Any, Callable, Dict, List

from aiocache import cached
from cachetools import TTLCache
from sqlalchemy import MetaData, inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.schema import CreateTable

from ..utils import track_time

_tools_instance = None
_tools_instance_multiturn = None


class SQLTools:
    """Tools to query the SQL database."""

    def __init__(self) -> None:
        """Initialize the SQLTools class."""
        self._max_sql_response_length = 20000
        self._response_too_long_message = "Sorry, SQL response was too long"
        self._schema_cache: TTLCache = TTLCache(maxsize=100, ttl=60 * 60 * 24)

    @staticmethod
    def handle_sql_response_length(func: Callable) -> Callable:
        """Decorator to handle the length of the SQL response."""

        @wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Wrapper"""
            response = await func(self, *args, **kwargs)
            if len(str(response)) > self._max_sql_response_length:
                return self._response_too_long_message
            return response

        return wrapper

    @track_time(create_class_attr="timings")
    @cached(ttl=60 * 60 * 24)
    @handle_sql_response_length
    async def _get_table_schema(
        self,
        table_list: List[str],
        asession: AsyncSession,
    ) -> dict[str, str]:
        """
        Queries the target SQL database and returns the schema of the tables.

        Args:
        - table_list (list[str]): The list of table names for which you
            want to get the schema.
        - asession (AsyncSession): The SQLAlchemy AsyncSession object.

        Returns:
        - dict[str, str]: The schema of all the relevant tables in the database.
        """

        metadata = MetaData()
        return_schema = {table: "" for table in table_list}

        def _do_reflect(_: Any) -> List[str]:
            """Reflect the tables."""
            engine = asession.get_bind()
            inspector = inspect(engine)
            existing_tables = [
                table for table in table_list if inspector.has_table(table)
            ]
            metadata.reflect(bind=engine, only=existing_tables, views=True)
            return existing_tables

        # Execute the reflection
        existing_tables = await asession.run_sync(_do_reflect)

        for table_name in existing_tables:
            table = metadata.tables.get(table_name)
            if table is not None:
                ddl_statement = str(
                    CreateTable(table).compile(bind=asession.get_bind())
                )
                return_schema[table.name] += f"\nTable: {table.name}\n{ddl_statement}\n"

                # Fetching the first three rows from the table
                first_n_rows_result = await asession.execute(select(table).limit(3))
                first_n_rows = first_n_rows_result.mappings().all()
                first_n_rows_str = "\n".join(
                    ["\t".join(map(str, row.values())) for row in first_n_rows]
                )
                return_schema[table.name] += f"Sample rows:\n{first_n_rows_str}\n"

        return return_schema

    @track_time(create_class_attr="timings")
    @cached(ttl=60 * 60 * 24)
    @handle_sql_response_length
    async def get_tables_schema(
        self,
        table_list: List[str],
        asession: AsyncSession,
        metric_db_id: str,
    ) -> str:
        """
        Queries the target SQL database and returns the schema of the tables.

        Args:
        - table_list (list[str]): The list of table names for which you
            want to get the schema.
        - asession (AsyncSession): The SQLAlchemy AsyncSession object.

        Returns:
        - str: The schema of all the relevant tables in the database.
        """
        if metric_db_id not in self._schema_cache:
            add_to_cache = await self._get_table_schema(table_list, asession)
            self._schema_cache[metric_db_id] = add_to_cache
        else:
            # Update cache with tables if not in cache
            tables_not_in_cache = [
                table
                for table in table_list
                if table not in self._schema_cache[metric_db_id]
            ]
            if tables_not_in_cache:
                add_to_cache = await self._get_table_schema(
                    tables_not_in_cache, asession
                )
                self._schema_cache[metric_db_id].update(add_to_cache)
        # Add the value for each table in table_list to return schema as a string append
        return_schema = "\n".join(
            [self._schema_cache[metric_db_id][table] for table in table_list]
        )
        return return_schema

    @track_time(create_class_attr="timings")
    @cached(ttl=60 * 60 * 24)
    @handle_sql_response_length
    async def get_common_column_values(
        self,
        table_column_dict: Dict[str, List[str]],
        asession: AsyncSession,
        num_common_values: int,
        indicator_vars: list,
    ) -> Dict[str, Dict]:
        """
        Queries the target SQL database and returns the top k (=num_common_values)
        most common values of the columns from respective tables.

        Args:
        - table_column_dict: A dictionary with table names as keys and a list of
            column names as values.
        - asession (AsyncSession): The SQLAlchemy AsyncSession object.
        - num_common_values (int): The number of common values to return.
        - indicator_vars (list): The list of indicator variables

        Returns:
        - dict[str, dict]: A Dictionary with the top common values for each table
            column combination which was asked for.
        """
        # Create dictionary of tables, their columns, and the columns' types
        result: Dict[str, Dict] = {}
        for table, columns in table_column_dict.items():
            result[table] = {}
            for column in columns:
                query = f"""
                SELECT {column}, COUNT(*)
                FROM {table}
                GROUP BY {column}
                ORDER BY COUNT(*) DESC
                """

                if column.lower() not in [var.lower() for var in indicator_vars]:
                    query += f" LIMIT {num_common_values}"

                sql_response = await asession.execute(text(query + ";"))
                result[table][column] = sql_response.fetchall()

        return result

    @track_time(create_class_attr="timings")
    @cached(ttl=60 * 60 * 24)
    @handle_sql_response_length
    async def run_sql(
        self, sql_query: str, asession: AsyncSession
    ) -> List[Dict[str, Any]]:
        """
        Executes the SQL query on the target SQL database.

        Args:
        - sql_query (str): The SQL query to execute.
        - asession (AsyncSession): The SQLAlchemy AsyncSession object.

        Returns:
        - list[dict[str, Any]]: The result of the SQL query.
        """
        sql_response = await asession.execute(text(sql_query))

        return sql_response.fetchall()


def get_tools() -> SQLTools:
    """Return the SQLTools instance."""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = SQLTools()
    return _tools_instance


def get_tools_multiturn() -> SQLTools:
    """Return the SQLTools instance."""
    global _tools_instance_multiturn
    if _tools_instance_multiturn is None:
        _tools_instance_multiturn = SQLTools()
    return _tools_instance_multiturn
