"""Behavior checks for the SQLite tools without network access."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sql_tools import create_sql_tools


class SQLToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.db"
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
                CREATE TABLE Artist (id INTEGER PRIMARY KEY, name TEXT);
                CREATE TABLE Album (id INTEGER, artist_id INTEGER REFERENCES Artist(id));
                INSERT INTO Artist VALUES (1, 'Alice'), (2, 'Bob');
                INSERT INTO Album VALUES (10, 1);
            """)
        model = FakeListChatModel(responses=["SELECT COUNT(*) FROM Artist"])
        self.tools = {tool.name: tool for tool in create_sql_tools(self.path, model)}

    def test_list_and_schema(self) -> None:
        self.assertEqual(self.tools["sql_db_list_tables"].invoke({}), "Album, Artist")
        schema = self.tools["sql_db_schema"].invoke({"table_names": "Artist, Album"})
        for content in ("PRIMARY KEY", "REFERENCES Artist", "Alice", "artist_id"):
            self.assertIn(content, schema)

    def test_queries_and_recovery(self) -> None:
        query = self.tools["sql_db_query"]
        self.assertEqual(
            query.invoke({"query": "SELECT COUNT(*) FROM Artist"}), "[(2,)]"
        )
        self.assertTrue(
            query.invoke({"query": "SELECT missing FROM Artist"}).startswith("Error:")
        )
        self.assertEqual(
            query.invoke(
                {"query": "WITH a AS (SELECT * FROM Artist) SELECT COUNT(*) FROM a"}
            ),
            "[(2,)]",
        )
        self.assertEqual(
            query.invoke({"query": "SELECT * FROM Artist WHERE id = 99"}), "[]"
        )

    def test_writes_and_attachments_are_rejected(self) -> None:
        query = self.tools["sql_db_query"]
        for sql in (
            "DELETE FROM Artist",
            "DROP TABLE Artist",
            "ATTACH ':memory:' AS other",
        ):
            with self.subTest(sql=sql):
                self.assertTrue(query.invoke({"query": sql}).startswith("Error:"))
        self.assertEqual(
            query.invoke({"query": "SELECT COUNT(*) FROM Artist"}), "[(2,)]"
        )

    def test_unknown_table_returns_error(self) -> None:
        result = self.tools["sql_db_schema"].invoke(
            {"table_names": "Artist; DROP TABLE Artist"}
        )
        self.assertTrue(result.startswith("Error:"))

    def test_checker_returns_model_response(self) -> None:
        result = self.tools["sql_db_query_checker"].invoke(
            {"query": "SELECT COUNT() FROM Artist"}
        )
        self.assertEqual(result, "SELECT COUNT(*) FROM Artist")

    def test_missing_database_is_not_created(self) -> None:
        path = self.path.with_name("missing.db")
        with self.assertRaises(FileNotFoundError):
            create_sql_tools(path, FakeListChatModel(responses=["unused"]))
        self.assertFalse(path.exists())
