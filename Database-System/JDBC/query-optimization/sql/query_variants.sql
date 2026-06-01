-- query_variants.sql
-- Three example query variants and EXPLAIN templates for comparative analysis

-- Variant A: Simple JOIN
EXPLAIN (ANALYZE, BUFFERS)
SELECT s."S_num", s."SNAME", SUM(c."CREDIT") AS total_credit
FROM "public"."S799" s
JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
JOIN "public"."C799" c ON sc."C_num" = c."C_num"
GROUP BY s."S_num", s."SNAME";

-- Variant B: Aggregation with pre-filtering (may help if many NULL grades)
EXPLAIN (ANALYZE, BUFFERS)
SELECT s."S_num", s."SNAME", COALESCE(SUM(c."CREDIT"),0) AS total_credit
FROM "public"."S799" s
LEFT JOIN (
    SELECT "S_num", "C_num" FROM "public"."SC799" WHERE "GRADE" IS NOT NULL
) sc ON s."S_num" = sc."S_num"
LEFT JOIN "public"."C799" c ON sc."C_num" = c."C_num"
GROUP BY s."S_num", s."SNAME";

-- Variant C: Use EXISTS to filter students who selected CS courses
EXPLAIN (ANALYZE, BUFFERS)
SELECT s."S_num", s."SNAME"
FROM "public"."S799" s
WHERE EXISTS (
    SELECT 1 FROM "public"."SC799" sc WHERE sc."S_num" = s."S_num" AND sc."C_num" LIKE 'CS-%'
)
ORDER BY s."S_num" DESC;
