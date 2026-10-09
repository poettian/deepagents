"""SQLite tools for exploring and querying the Chinook database."""

import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool


def _authorize_read(
    action: int,
    arg1: str | None,
    arg2: str | None,
    database: str | None,
    source: str | None,
) -> int:
    # Let SQLite enforce read-only statements, including queries using CTEs.
    allowed = (
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    )
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.set_authorizer(_authorize_read)
    return connection


def _table_schema(connection: sqlite3.Connection, name: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    if row is None:
        msg = f"Unknown table: {name}"
        raise ValueError(msg)
    # Identifiers cannot be bound as parameters; quote a verified table name.
    identifier = '"' + name.replace('"', '""') + '"'
    cursor = connection.execute(f"SELECT * FROM {identifier} LIMIT 3")
    columns = ", ".join(column[0] for column in cursor.description)
    return f"{row[0]}\nSample rows ({columns}):\n{cursor.fetchall()}"


def create_sql_tools(path: Path, model: BaseChatModel) -> list[BaseTool]:
    """Create read-only SQLite tools and a model-backed SQL checker.

    Args:
        path: Existing SQLite database file.
        model: Chat model used to review SQL before execution.

    Returns:
        Tools for listing tables, reading schemas, querying, and checking SQL.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    path = path.resolve()
    if not path.is_file():
        msg = f"Database not found: {path}. Download chinook.db as described in README.md."
        raise FileNotFoundError(msg)

    @tool
    def sql_db_list_tables(tool_input: str = "") -> str:
        """List database tables. Pass an empty string as tool_input."""
        with closing(_connect(path)) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return ", ".join(row[0] for row in rows)

    @tool
    def sql_db_schema(table_names: str) -> str:
        """Get CREATE TABLE definitions and 3 sample rows for comma-separated table_names."""
        try:
            with closing(_connect(path)) as connection:
                return "\n\n".join(
                    _table_schema(connection, name.strip())
                    for name in table_names.split(",")
                )
        except (sqlite3.Error, ValueError) as error:
            return f"Error: {error}"

    @tool
    def sql_db_query(query: str) -> str:
        """Execute a read-only SQLite query. On error, revise the SQL and retry.

        Use sql_db_schema to check columns and sql_db_query_checker before executing.
        """
        try:
            with closing(_connect(path)) as connection:
                return str(connection.execute(query).fetchall())
        except sqlite3.Error as error:
            return f"Error: {error}"

    @tool
    def sql_db_query_checker(query: str) -> str:
        """Check a SQLite query for mistakes before calling sql_db_query."""
        response = model.invoke(
            [
                (
                    "system",
                    "Review this read-only SQLite query. Check NULL handling, "
                    "joins, types, quoting, function arguments, UNION versus UNION ALL, "
                    "and range boundaries. Return only the corrected SQL, or the "
                    "original SQL if it is correct.",
                ),
                ("human", query),
            ]
        )
        return response.text

    return [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]
