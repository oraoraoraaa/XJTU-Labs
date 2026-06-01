# Data Generator Utilities

This folder contains tools for data generation and optimization experiments.

## Files

- `DataGenerator.java`: multithreaded JDBC generator/inserter for S799/C799/SC799.
- `generate_names.py`: Python helper to fetch richer names and output SQL inserts.
- `optimization_driver.py`: automated optimization experiment driver.
- `requirements.txt`: Python dependencies.

## Install Python Dependencies

```bash
cd Database-System/JDBC/query-optimization
pip3 install -r requirements.txt
```

## Run Automated Optimization Experiment

```bash
cd Database-System/JDBC/query-optimization
python3 optimization_driver.py \
  --host 192.168.39.160 --port 7654 --dbname mydb \
  --user dbremote --password 'dbremote:399' \
  --apply-indexes --run-queries --repeats 3 --analyze-after-ddl
```

Optional partition experiment:

```bash
python3 optimization_driver.py \
  --host 192.168.39.160 --port 7654 --dbname mydb \
  --user dbremote --password 'dbremote:399' \
  --apply-partition --run-queries --repeats 3 --analyze-after-ddl
```

Results are written to:

- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.json`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.csv`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.md`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/plans/*.txt`

## Notes

- `partitioning.sql` is a template and may require adjustments before production usage.
- Use an isolated environment when applying partitioning/migration scripts.
