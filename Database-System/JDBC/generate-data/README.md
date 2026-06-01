# Data Generator Utilities

This folder contains tools for data generation and optimization experiments.

## Files

- `DataGenerator.java`: multithreaded JDBC generator/inserter for S799/C799/SC799.
- `generate_names.py`: Python helper to fetch richer names, emit SQL inserts, and write plain-text name lists for the Java generator.
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

Results are written to:

- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.json`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.csv`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.md`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/plans/*.txt`

## Use Python Names in Java

The Python helper can generate reusable name lists that the Java generator consumes.

```bash
cd Database-System/JDBC/generate-data
python3 generate_names.py --students 1000 --courses 100 --enrollments 20000 --outdir ./sql_out

# Then pass the generated text files into the Java generator.
cd Database-System/JDBC/generate-data
javac -cp "../opengauss-jdbc-6.0.0.jar" DataGenerator.java
java -cp ".:../opengauss-jdbc-6.0.0.jar" DataGenerator \
  --students 1000 --courses 100 --enrollments 20000 --threads 8 \
  --student-names ./sql_out/student_names.txt \
  --teacher-names ./sql_out/teacher_names.txt
```

If the name files are not provided, the Java generator falls back to synthetic names.

## Notes

- The driver is indexing-only; it does not apply any partition migration scripts.
