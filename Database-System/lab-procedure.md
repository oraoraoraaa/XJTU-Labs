# Detailed Procedures on Completing the Lab

## Create `mydb` database

```sql
CREATE DATABASE "mydb"
WITH
  OWNER = "dbremote"
;
```

## Create Tables

```sql
CREATE TABLE "public"."S799" (
  "S_num" varchar(20) NOT NULL,
  "SNAME" varchar(50) NOT NULL,
  "SEX" varchar(10),
  "BDATE" date,
  "HEIGHT" decimal(3,2),
  "DORM" varchar(100),
  PRIMARY KEY ("S_num")
);

CREATE TABLE "public"."C799" (
    "C_num" VARCHAR(20) NOT NULL,
    "CNAME" VARCHAR(100) NOT NULL,
    "PERIOD" INT,
    "CREDIT" DECIMAL(3,1), 
    "TEACHER" VARCHAR(50),
    PRIMARY KEY ("C_num")
);

CREATE TABLE "public"."SC799" (
    "S_num" VARCHAR(20),
    "C_num" VARCHAR(20),
    "GRADE" DECIMAL(4,1),
    PRIMARY KEY ("S_num", "C_num"),
    FOREIGN KEY ("S_num") REFERENCES "S799"("S_num"),
    FOREIGN KEY ("C_num") REFERENCES "C799"("C_num")
);
```

## Writing Data to Table

S799:

```sql
INSERT INTO "public"."S799" ("S_num", "SNAME", "SEX", "BDATE", "HEIGHT", "DORM") VALUES
('01032010', '王涛', '男', '2004-04-05', 1.72, '东6舍221'),
('01032023', '孙文', '男', '2005-06-10', 1.80, '东6舍221'),
('01032001', '张晓梅', '女', '2004-11-17', 1.58, '东1舍312'),
('01032005', '刘静', '女', '2004-01-10', 1.63, '东1舍312'),
('01032112', '许澍', '男', '2004-02-20', 1.71, '东6舍221'),
('03031011', '王倩', '女', '2005-09-20', 1.66, '东2舍104'),
('03031014', '赵思扬', '男', '2003-06-06', 1.85, '东18舍421'),
('03031051', '周剑', '男', '2003-05-08', 1.68, '东18舍422'),
('03031009', '田菲', '女', '2004-08-11', 1.60, '东2舍104'),
('03031033', '蔡明明', '男', '2004-03-12', 1.75, '东18舍423'),
('03031056', '曹子衿', '女', '2006-12-15', 1.65, '东2舍305');
```

C799:

```sql
INSERT INTO "public"."C799" ("C_num", "CNAME", "PERIOD", "CREDIT", "TEACHER") VALUES
('CS-01', '数据结构', 60, 3, '张军'),
('CS-02', '计算机组成原理', 80, 4, '王亚伟'),
('CS-04', '人工智能', 40, 2, '李蕾'),
('CS-05', '深度学习', 40, 2, '崔昀'),
('EE-01', '信号与系统', 60, 3, '张明'),
('EE-02', '数字逻辑电路', 100, 5, '胡海东'),
('EE-03', '光电子学与光子学', 40, 2, '石韬');
```

SC799:

```sql
INSERT INTO "public"."SC799" ("S_num", "C_num", "GRADE") VALUES
('01032010', 'CS-01', 82.0),
('01032010', 'CS-02', 91.0),
('01032010', 'CS-04', 83.5),
('01032001', 'CS-01', 77.5),
('01032001', 'CS-02', 85.0),
('01032001', 'CS-04', 83.0),
('01032005', 'CS-01', 62.0),
('01032005', 'CS-02', 77.0),
('01032005', 'CS-04', 82.0),
('01032023', 'CS-01', 55.0),
('01032023', 'CS-02', 81.0),
('01032023', 'CS-04', 76.0),
('01032112', 'CS-01', 88.0),
('01032112', 'CS-02', 91.5),
('01032112', 'CS-04', 86.0),
('01032112', 'CS-05', NULL),
('03031033', 'EE-01', 93.0),
('03031033', 'EE-02', 89.0),
('03031009', 'EE-01', 88.0),
('03031009', 'EE-02', 78.5),
('03031011', 'EE-01', 91.0),
('03031011', 'EE-02', 86.0),
('03031051', 'EE-01', 78.0),
('03031051', 'EE-02', 58.0),
('03031014', 'EE-01', 79.0),
('03031014', 'EE-02', 71.0);
```

## Query Tables and Attributes

```sql
SELECT 
    table_name AS "表名", 
    column_name AS "属性名", 
    data_type AS "数据类型", 
    character_maximum_length AS "最大长度",
    is_nullable AS "是否允许为空"
FROM 
    information_schema.columns
WHERE 
    table_schema = 'public' 
    AND table_name IN ('S799', 'C799', 'SC799')
ORDER BY 
    table_name, ordinal_position;
```

## 三 - 1

### (1) 查询计算机系（CS）所开课程的课程编号、课程名称及学分数

思路：直接查询课程表 C799，通过课程编号 C_num 的前缀或关键字来筛选计算机系课程。

```sql
SELECT "C_num", "CNAME", "CREDIT"
FROM "public"."C799"
WHERE "C_num" LIKE 'CS%';
```

### (2) 查询未选修课程“CS-05”的男生学号及其已选各课程编号、成绩

思路：先在子查询中找出选修了 CS-05 的学生学号，在外层查询中筛选出性别为“男”、且学号不在该子查询结果中的学生。连接 SC799 表获取他们已选的其他课程和成绩。

```SQL
SELECT s."S_num", sc."C_num", sc."GRADE"
FROM "public"."S799" s
JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
WHERE s."SEX" = '男'
  AND s."S_num" NOT IN (
      SELECT "S_num" 
      FROM "public"."SC799" 
      WHERE "C_num" = 'CS-05'
  );
```

### (3) 查询 2005 年～2006 年出生学生的基本信息

思路：利用 BETWEEN ... AND ... 筛选出生日期 BDATE 在 2005-01-01 到 2006-12-31 之间的学生。

```SQL
SELECT *
FROM "public"."S799"
WHERE "BDATE" BETWEEN '2005-01-01' AND '2006-12-31';
```

### (4) 查询每位学生的学号、学生姓名及其已选修课程的学分总数

思路：使用 LEFT JOIN 连接学生、选课和课程表（确保没选课的学生也能查到，学分总数显示为 0 或 NULL），然后按学生分组并对学分求和。

```SQL
SELECT s."S_num", s."SNAME", COALESCE(SUM(c."CREDIT"), 0) AS "TOTAL_CREDIT"
FROM "public"."S799" s
LEFT JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
LEFT JOIN "public"."C799" c ON sc."C_num" = c."C_num"
GROUP BY s."S_num", s."SNAME";
```

### (5) 查询选修课程“CS-01”的学生中成绩第二高的学生学号

思路：筛选出 CS-01 的成绩后，按成绩降序排列，利用 OFFSET 1 LIMIT 1（跳过第 1 名，取第 2 名）来获取。

```SQL
SELECT "S_num"
FROM "public"."SC799"
WHERE "C_num" = 'CS-01' AND "GRADE" IS NOT NULL
ORDER BY "GRADE" DESC
LIMIT 1 OFFSET 1;
```

### (6) 查询平均成绩低于“许澍”同学的学生学号、姓名和平均成绩，并按学号进行降序排列

思路：子查询算出“许澍”个人的平均成绩。外层查询对所有学生进行分组聚合算出各自的平均分，利用 HAVING 子句过滤出低于许澍平均分的人。

```SQL
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
```

### (7) 查询选修了计算机专业全部课程（课程编号为“CS-XX”）的学生姓名及已获得的学分总数

思路：计算机专业的全部课程数量可以通过统计 C799 中 C_num LIKE 'CS%' 的总数得到。只有当学生选修的 CS 课程数量等于该总数时，才说明选满了全部课程（注意：在计算“已获得学分”时，通常及格/有成绩才算获得，这里以通过 GRADE IS NOT NULL 或通识修完为准进行计算）。

```SQL
SELECT s."SNAME", SUM(c."CREDIT") AS "TOTAL_CREDIT"
FROM "public"."S799" s
JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
JOIN "public"."C799" c ON sc."C_num" = c."C_num"
GROUP BY s."S_num", s."SNAME"
HAVING COUNT(CASE WHEN sc."C_num" LIKE 'CS%' THEN 1 END) = (
    SELECT COUNT(*) 
    FROM "public"."C799" 
    WHERE "C_num" LIKE 'CS%'
);
```

### (8) 查询选修了 3 门以上课程（包括 3 门）的学生中平均成绩最高的同学学号及姓名

思路：先用 HAVING COUNT(course) >= 3 过滤出符合选课数量的学生，再按平均分降序排列，用 LIMIT 1 摘取第一名。

```SQL
SELECT s."S_num", s."SNAME"
FROM "public"."S799" s
JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
GROUP BY s."S_num", s."SNAME"
HAVING COUNT(sc."C_num") >= 3
ORDER BY AVG(sc."GRADE") DESC
LIMIT 1;
```
