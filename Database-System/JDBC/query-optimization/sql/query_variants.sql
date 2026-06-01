-- query_variants.sql
-- Query family A: Query (4) from the lab image
-- 查询每位学生的学号、学生姓名及其已选修课程的学分总数

EXPLAIN (ANALYZE, BUFFERS)
SELECT s."S_num", s."SNAME", COALESCE(SUM(c."CREDIT"), 0) AS "TOTAL_CREDIT"
FROM "public"."S799" s
LEFT JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
LEFT JOIN "public"."C799" c ON sc."C_num" = c."C_num"
GROUP BY s."S_num", s."SNAME";

-- Alternative form for Query (4): pre-filter selected courses before joining
EXPLAIN (ANALYZE, BUFFERS)
SELECT s."S_num", s."SNAME", COALESCE(SUM(c."CREDIT"), 0) AS "TOTAL_CREDIT"
FROM "public"."S799" s
LEFT JOIN (
    SELECT "S_num", "C_num"
    FROM "public"."SC799"
    WHERE "GRADE" IS NOT NULL
) sc ON s."S_num" = sc."S_num"
LEFT JOIN "public"."C799" c ON sc."C_num" = c."C_num"
GROUP BY s."S_num", s."SNAME";

-- Query family B: Query (6) from the lab image
-- 查询平均成绩低于“许澍”同学的学生学号、姓名和平均成绩，并按学号降序排列

EXPLAIN (ANALYZE, BUFFERS)
SELECT s."S_num", s."SNAME", AVG(sc."GRADE") AS "AVG_GRADE"
FROM "public"."S799" s
JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
GROUP BY s."S_num", s."SNAME"
HAVING AVG(sc."GRADE") < (
    SELECT AVG(sc2."GRADE")
    FROM "public"."S799" s2
    JOIN "public"."SC799" sc2 ON s2."S_num" = sc2."S_num"
    WHERE s2."SNAME" = '许澍'
)
ORDER BY s."S_num" DESC;

-- Query family C: Query (8) from the lab image
-- 查询选修了 3 门以上课程（包括 3 门）的学生中平均成绩最高的同学学号及姓名

EXPLAIN (ANALYZE, BUFFERS)
SELECT s."S_num", s."SNAME"
FROM "public"."S799" s
JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
GROUP BY s."S_num", s."SNAME"
HAVING COUNT(sc."C_num") >= 3
ORDER BY AVG(sc."GRADE") DESC
LIMIT 1;

-- Alternative form for Query (8): use a CTE to separate aggregation from ranking
EXPLAIN (ANALYZE, BUFFERS)
WITH student_avg AS (
    SELECT s."S_num", s."SNAME", COUNT(sc."C_num") AS course_count, AVG(sc."GRADE") AS avg_grade
    FROM "public"."S799" s
    JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
    GROUP BY s."S_num", s."SNAME"
)
SELECT "S_num", "SNAME"
FROM student_avg
WHERE course_count >= 3
ORDER BY avg_grade DESC
LIMIT 1;
