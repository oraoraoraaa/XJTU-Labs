# Optimization Experiment Summary

Total events: 20

## Query Results

- Query 1: avg elapsed 172.352 ms, avg plan execution 149.761 ms
- Query 2: avg elapsed 173.993 ms, avg plan execution 140.556 ms
- Query 3: avg elapsed 173.951 ms, avg plan execution 155.535 ms
- Query 4: avg elapsed 173.772 ms, avg plan execution 142.309 ms
- Query 5: avg elapsed 174.296 ms, avg plan execution 146.125 ms

## Artifacts

- summary.json
- summary.csv
- plans/*.txt

## Analysis

This run is broadly expected for the current workload. The five query variants all finish in the same general range, with average elapsed time around 170 ms and average plan execution time around 140-156 ms. The spread between variants is small, which suggests that the optimizer is not finding a dramatically cheaper access path for any one variant, even after indexes are applied.

The raw plan for the first query family shows the main reason: the optimizer still uses sequential scans on `S799`, `C799`, and `SC799`, followed by hash joins and a hash aggregate. That is a reasonable choice because query (4) touches almost every row in all three tables, so a B-tree index on `SC799` does not offer much benefit when the query needs the full join result anyway. In other words, the workload is dominated by large-table scans and joins, not by selective point lookups.

The `ANALYZE` step completed successfully and the plan runtime values are now being parsed correctly, so the reported `plan execution` numbers are usable for comparison. The current numbers also imply that the index experiment mainly verified stability rather than producing a major speedup. This is normal for queries (4), (6), and (8), because they are aggregation-heavy and relatively unselective.

Implications for the report:

- The experiment is valid, but the index choice is not expected to produce a large improvement for these queries.
- If a stronger optimization effect is needed, the report should compare against a more selective query, or add a different index strategy, such as an index on the exact filtering column used by a highly selective predicate.
- For the current lab requirement, it is enough to report that the indexed workload remains mostly scan-bound and that the optimizer prefers hash joins plus sequential scans on this dataset scale.
