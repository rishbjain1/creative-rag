"""A text→SQL agent over a bundled SQLite analytics DB.

Multi-step tool-use shape: introspect the schema → ask the LLM for a single
read-only SQL query → guard it (reject any write/DDL) → execute it. Paired with
sqleval.py, which scores it by **execution accuracy** (does the predicted query
return the same rows as a reference query) — the standard text2SQL metric, robust
to SQL phrasing differences.

The DB is built from eval/sql/schema.sql (committed + seeded), so every run —
local or CI — hits an identical database.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from . import llm

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "eval" / "sql" / "schema.sql"

# Anything that mutates data or schema, or runs multiple statements, is rejected.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|"
    r"vacuum|reindex|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


def build_db(schema_path: Path | None = None) -> sqlite3.Connection:
    """In-memory DB built from the committed schema + seed."""
    sql = (schema_path or SCHEMA_PATH).read_text()
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)
    return conn


def schema_text(conn: sqlite3.Connection) -> str:
    """The CREATE statements, as the LLM prompt context."""
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
    ).fetchall()
    return "\n\n".join(r[0] for r in rows)


def is_safe_select(sql: str) -> bool:
    """True only for a single read-only SELECT/WITH query."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False
    if ";" in s:  # no multiple statements
        return False
    if _FORBIDDEN.search(s):
        return False
    return bool(re.match(r"(?is)^\s*(select|with)\b", s))


def _extract_sql(text: str) -> str:
    """Pull SQL out of a model reply (fenced ```sql block or bare)."""
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    return candidate.strip().rstrip(";").strip()


SYSTEM = """You are a text-to-SQL agent for SQLite. Given a database schema and a
question, return ONE read-only SQL query that answers it. Rules: SELECT only (no
INSERT/UPDATE/DELETE/DDL), a single statement, valid SQLite. Return ONLY the SQL,
optionally in a ```sql code block — no prose, no explanation."""


def generate_sql(question: str, schema: str, model: str | None = None) -> str:
    reply = llm.chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"SCHEMA:\n{schema}\n\nQUESTION: {question}\n\nSQL:"},
    ], model=model)
    return _extract_sql(reply)


def run_sql(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    """Execute a query and return its rows, order-normalized for comparison."""
    rows = conn.execute(sql).fetchall()
    return sorted(rows, key=lambda r: tuple(str(x) for x in r))


def answer(question: str, conn: sqlite3.Connection, model: str | None = None) -> dict:
    """Full agent step: schema → generate SQL → guard → execute."""
    sql = generate_sql(question, schema_text(conn), model=model)
    if not is_safe_select(sql):
        return {"sql": sql, "rows": None, "safe": False, "error": "unsafe or non-SELECT SQL"}
    try:
        return {"sql": sql, "rows": run_sql(conn, sql), "safe": True, "error": None}
    except sqlite3.Error as e:
        return {"sql": sql, "rows": None, "safe": True, "error": str(e)}
