"""Offline tests for the text→SQL agent + eval — no LLM, no network.

Builds the real bundled DB and exercises the safety guard, execution, result
matching, and reference-SQL validity.
"""
from pathlib import Path

from creative_rag import sqlagent, sqleval

QA = Path(__file__).resolve().parent.parent / "eval" / "sql" / "qa_sql.jsonl"


# --- DB + schema ---
def test_build_db_has_tables():
    conn = sqlagent.build_db()
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"customers", "products", "orders", "order_items"} <= names


def test_schema_text_includes_creates():
    conn = sqlagent.build_db()
    s = sqlagent.schema_text(conn)
    assert "CREATE TABLE customers" in s and "CREATE TABLE orders" in s


# --- safety guard ---
def test_safe_select_accepts_select_and_with():
    assert sqlagent.is_safe_select("SELECT * FROM customers")
    assert sqlagent.is_safe_select("WITH x AS (SELECT 1) SELECT * FROM x")
    assert sqlagent.is_safe_select("select count(*) from orders;")


def test_safe_select_rejects_writes_and_multistatement():
    assert not sqlagent.is_safe_select("DROP TABLE customers")
    assert not sqlagent.is_safe_select("DELETE FROM orders WHERE id=1")
    assert not sqlagent.is_safe_select("INSERT INTO customers VALUES (9,'x','US','2025-01-01')")
    assert not sqlagent.is_safe_select("SELECT 1; DROP TABLE customers")
    assert not sqlagent.is_safe_select("UPDATE products SET price=0")
    assert not sqlagent.is_safe_select("")


def test_extract_sql_strips_fence():
    assert sqlagent._extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert sqlagent._extract_sql("SELECT 1;") == "SELECT 1"


# --- execution + matching ---
def test_run_sql_known_answer():
    conn = sqlagent.build_db()
    # 2 US customers (Alice, Diego)
    assert sqlagent.run_sql(conn, "SELECT COUNT(*) FROM customers WHERE country='US'") == [(2,)]


def test_result_match_order_insensitive():
    assert sqleval.result_match([(2,), (1,)], [(1,), (2,)])
    assert not sqleval.result_match([(1,)], [(2,)])
    assert not sqleval.result_match(None, [(1,)])


def test_result_match_float_tolerant():
    assert sqleval.result_match([(129.49600001,)], [(129.496,)])


# --- reference SQL is valid (this is what CI gates on) ---
def test_all_gold_sql_safe_and_runnable():
    v = sqleval.validate_gold(QA)
    assert v["problems"] == [], v["problems"]
    assert v["n"] == 12
