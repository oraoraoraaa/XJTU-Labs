#!/usr/bin/env python3
"""Automated optimization experiment driver for openGauss.

Features:
1) Apply SQL scripts for indexes and optional partitioning.
2) Run EXPLAIN (ANALYZE, BUFFERS) query variants.
3) Collect structured results (JSON/CSV) and plan text files.

Example:
  python3 optimization_driver.py \
    --host 192.168.39.160 --port 7654 --dbname mydb \
    --user dbremote --password 'dbremote:399' \
    --apply-indexes --run-queries --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import psycopg2
import sqlparse


ROOT = Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"
DEFAULT_INDEXES = SQL_DIR / "indexes.sql"
DEFAULT_PARTITION = SQL_DIR / "partitioning.sql"
DEFAULT_QUERIES = SQL_DIR / "query_variants.sql"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimization experiments on openGauss")

    parser.add_argument("--host", default=os.getenv("OG_HOST", "192.168.39.160"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OG_PORT", "7654")))
    parser.add_argument("--dbname", default=os.getenv("OG_DB", "mydb"))
    parser.add_argument("--user", default=os.getenv("OG_USER", "dbremote"))
    parser.add_argument("--password", default=os.getenv("OG_PASS", "dbremote:399"))

    parser.add_argument("--indexes-sql", type=Path, default=DEFAULT_INDEXES)
    parser.add_argument("--partition-sql", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--queries-sql", type=Path, default=DEFAULT_QUERIES)

    parser.add_argument("--apply-indexes", action="store_true", help="Apply indexes SQL script")
    parser.add_argument("--apply-partition", action="store_true", help="Apply partition SQL script")
    parser.add_argument("--run-queries", action="store_true", help="Run EXPLAIN ANALYZE queries")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each query N times")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiment-results",
        help="Directory for experiment output",
    )
    parser.add_argument("--analyze-after-ddl", action="store_true", help="Run ANALYZE after DDL scripts")

    return parser.parse_args()


def sql_statements(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    chunks = sqlparse.split(text)
    statements: List[str] = []
    for chunk in chunks:
        stmt = chunk.strip()
        if not stmt:
            continue
        if stmt.startswith("--") and "\n" not in stmt:
            continue
        statements.append(stmt)
    return statements


def is_explain_statement(stmt: str) -> bool:
    compact = stmt.lstrip().upper()
    return compact.startswith("EXPLAIN")


def extract_execution_time_ms(plan_lines: List[str]) -> float:
    for line in plan_lines:
        m = re.search(r"Execution Time:\\s*([0-9.]+)\\s*ms", line)
        if m:
            return float(m.group(1))
    return -1.0


def connect(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )


def run_ddl_script(conn, script_path: Path, label: str) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    statements = sql_statements(script_path)
    for i, stmt in enumerate(statements, start=1):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(stmt)
        conn.commit()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results.append(
            {
                "type": "ddl",
                "label": label,
                "statement_index": i,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )
    return results


def run_analyze(conn) -> Dict[str, object]:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute('ANALYZE "public"."S799";')
        cur.execute('ANALYZE "public"."C799";')
        cur.execute('ANALYZE "public"."SC799";')
    conn.commit()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {"type": "maintenance", "label": "analyze", "elapsed_ms": round(elapsed_ms, 3)}


def run_query_variants(conn, queries_sql: Path, repeats: int, out_dir: Path) -> List[Dict[str, object]]:
    statements = [s for s in sql_statements(queries_sql) if is_explain_statement(s)]
    query_dir = out_dir / "plans"
    query_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for qidx, stmt in enumerate(statements, start=1):
        for r in range(1, repeats + 1):
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(stmt)
                rows = cur.fetchall()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            plan_lines = [row[0] for row in rows]
            exec_ms = extract_execution_time_ms(plan_lines)

            plan_file = query_dir / f"query_{qidx:02d}_run_{r:02d}.txt"
            plan_file.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

            results.append(
                {
                    "type": "query",
                    "query_index": qidx,
                    "run": r,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "plan_execution_ms": exec_ms,
                    "plan_file": str(plan_file),
                }
            )
    return results


def write_reports(events: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    fieldnames = sorted({k for e in events for k in e.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event)

    # Markdown summary
    md_path = out_dir / "summary.md"
    lines = ["# Optimization Experiment Summary", ""]
    lines.append(f"Total events: {len(events)}")
    lines.append("")

    query_events = [e for e in events if e.get("type") == "query"]
    if query_events:
        lines.append("## Query Results")
        by_query: Dict[int, List[Dict[str, object]]] = {}
        for e in query_events:
            by_query.setdefault(int(e["query_index"]), []).append(e)
        for qidx in sorted(by_query):
            runs = by_query[qidx]
            avg_elapsed = sum(float(r["elapsed_ms"]) for r in runs) / len(runs)
            plan_times = [float(r["plan_execution_ms"]) for r in runs if float(r["plan_execution_ms"]) >= 0]
            avg_plan = (sum(plan_times) / len(plan_times)) if plan_times else -1.0
            lines.append(f"- Query {qidx}: avg elapsed {avg_elapsed:.3f} ms, avg plan execution {avg_plan:.3f} ms")
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- {json_path.name}")
    lines.append(f"- {csv_path.name}")
    lines.append("- plans/*.txt")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not (args.apply_indexes or args.apply_partition or args.run_queries):
        raise SystemExit("No action selected. Use --apply-indexes and/or --apply-partition and/or --run-queries")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    events: List[Dict[str, object]] = []
    conn = connect(args)
    try:
        if args.apply_indexes:
            events.extend(run_ddl_script(conn, args.indexes_sql, "indexes"))

        if args.apply_partition:
            events.extend(run_ddl_script(conn, args.partition_sql, "partitioning"))

        if args.analyze_after_ddl and (args.apply_indexes or args.apply_partition):
            events.append(run_analyze(conn))

        if args.run_queries:
            events.extend(run_query_variants(conn, args.queries_sql, args.repeats, out_dir))
    finally:
        conn.close()

    write_reports(events, out_dir)
    print(f"Experiment completed. Results written to: {out_dir}")


if __name__ == "__main__":
    main()
