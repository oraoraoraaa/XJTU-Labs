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

## 三 - 2

在 S799 和 C799 表中插入学生记录：

```sql
INSERT INTO "public"."S799" ("S_num", "SNAME", "SEX", "BDATE", "HEIGHT", "DORM") 
VALUES ('01032005', '刘竟', '男', '2003-12-10', 1.75, '东14舍312');

INSERT INTO "public"."C799" ("C_num", "CNAME", "PERIOD", "CREDIT", "TEACHER") 
VALUES ('CS-03', '离散数学', 64, 4, '陈建明');
```

## 三 - 3

思路：利用子查询，先在选课表 SC799 和课程表 C799 中通过 GROUP BY 计算每位学生的总学分，筛选出大于 20 的学号，再由 DELETE 语句进行删除。由于设置了外键级联关系，如果直接删除学生，可能需要确保没有外键约束报错（或者确保外键是 ON DELETE CASCADE，否则需要先删 SC799 里的对应选课记录）。

```sql
DELETE FROM "public"."S799"
WHERE "S_num" IN (
    SELECT sc."S_num"
    FROM "public"."SC799" sc
    JOIN "public"."C799" c ON sc."C_num" = c."C_num"
    GROUP BY sc."S_num"
    HAVING SUM(c."CREDIT") > 20
);
```

## 三 - 4

```sql
UPDATE "public"."C799"
SET "PERIOD" = 36, "CREDIT" = "CREDIT" + 1
WHERE "TEACHER" = '张明' AND "CNAME" = '数字电子技术';
```

## 三 - 5

### (1) 居住在“东18舍”的男生视图

```sql
CREATE VIEW "view_male_dorm18" AS
SELECT "S_num", "SNAME", "BDATE", "HEIGHT", "SEX", "DORM"
FROM "public"."S799"
WHERE "SEX" = '男' AND "DORM" LIKE '东18舍%';
```

### (2) “张明”老师所开设课程情况的视图（含平均成绩）

```sql
CREATE VIEW "view_teacher_zhang_courses" AS
SELECT c."C_num", c."CNAME", AVG(sc."GRADE") AS "AVG_GRADE"
FROM "public"."C799" c
LEFT JOIN "public"."SC799" sc ON c."C_num" = sc."C_num"
WHERE c."TEACHER" = '张明'
GROUP BY c."C_num", c."CNAME";
```

### (3) 所有选修了“人工智能”课程的学生视图

```sql
CREATE VIEW "view_ai_students" AS
SELECT s."S_num", s."SNAME", sc."GRADE"
FROM "public"."S799" s
JOIN "public"."SC799" sc ON s."S_num" = sc."S_num"
JOIN "public"."C799" c ON sc."C_num" = c."C_num"
WHERE c."CNAME" = '人工智能';
```

## 四 - 1

### (1) 使用 JDBC 测试数据库连接

思路：使用 openGauss JDBC 驱动包 `opengauss-jdbc-6.0.0.jar`，加载 `org.opengauss.Driver`，连接到 `mydb` 数据库后执行一条简单的查询语句，用来验证连接是否成功。

程序中使用的连接信息如下：

- 地址：`192.168.39.160`
- 端口：`7654`
- 数据库名：`mydb`
- 用户名：`dbremote`

Java 代码：[`Database-System/JDBC/generate-data/DataGenerator.java`](./JDBC/generate-data/DataGenerator.java)

运行步骤：

```bash
javac -cp "opengauss-jdbc-6.0.0.jar" ConnectionTest.java
java -cp ".:opengauss-jdbc-6.0.0.jar" ConnectionTest
```

手动指定连接参数：

```bash
java -cp ".:opengauss-jdbc-6.0.0.jar" ConnectionTest jdbc:opengauss://192.168.39.160:7654/mydb dbremote dbremote:399
```

### (2) 大规模数据生成与多线程写入

思路：为完成实验附件中提出的数据规模任务，采用一个多线程的 Java 程序负责生成并写入数据。程序位于 `Database-System/JDBC/generate-data/DataGenerator.java`。实现要点：

- 使用随机生成的学号、课程编号和成绩分布，保证数据格式与表定义一致。
- 对 S799、C799、SC799 三个表分别采用分批（batch）提交，避免单条插入造成的过多网络往返。
- 每个批次由独立线程执行，各线程使用各自的 JDBC 连接以减小同步开销。
- 对可能造成主键冲突的插入，使用 `ON CONFLICT ... DO NOTHING` 做幂等写入，方便重复执行测试。
- 删除步骤使用基于 `ctid` 的子查询方式删除满足条件（成绩 < 60 或 NULL）的固定数量记录，避免大事务回滚风险。

#### 数据生成、数据存取

数据生成

为便于生成更真实的姓名与教师信息，使用一个 Python 脚本 [`JDBC/generate-data/generate_names.py`](./JDBC/generate-data/generate_names.py)：

- 功能：调用 RandomUser API 获取真实感的姓名，生成 `students.sql`、`courses.sql`、`enrollments.sql`。
- 输出：位于 `JDBC/generate-data/sql_out/`（可通过 `--outdir` 指定），文件为纯 SQL，可用 `psql` 或 JDBC 逐条执行或批量执行导入。

The Python helper can generate reusable name lists that the Java generator consumes.

```bash
python generate_names.py --students 1000 --courses 100 --enrollments 20000 --outdir ./sql_out

# Then pass the generated text files into the Java generator.
javac -cp "../opengauss-jdbc-6.0.0.jar" DataGenerator.java
java -cp ".:../opengauss-jdbc-6.0.0.jar" DataGenerator \
  --students 1000 --courses 100 --enrollments 20000 --threads 8 \
  --student-names ./sql_out/student_names.txt \
  --teacher-names ./sql_out/teacher_names.txt
```

If the name files are not provided, the Java generator falls back to synthetic names.

数据存取（写入）

- 使用批量提交（PreparedStatement.addBatch + executeBatch）显著降低网络往返与事务开销。
- 通过并发线程分区写入（每线程独立 Connection）提高并发写入吞吐，但要避免过多并发导致数据库连接耗尽或锁竞争。
- 将插入分解为适中大小的事务（例如每 200-1000 条提交一次），在出现错误时易于回滚和重试。
- 在写入前后使用 `ANALYZE`（或 `VACUUM ANALYZE`）更新统计信息，帮助优化器选择合理的执行计划。

## 四 - 2

自动化实验驱动器

将“索引/分区 + 查询性能评估”流程标准化，实现 Python 自动化实验驱动器：`JDBC/query-optimization/optimization_driver.py`。当前驱动器覆盖了 3 个来自“三、1”的查询：第 (4) 题、第 (6) 题和第 (8) 题，并为其中部分查询提供了不同写法的对比版本。

实验步骤：

1. 先用 `DataGenerator.java` 或 `generate_names.py` 生成并写入更大规模的数据。
2. 运行自动化驱动器做“基线测试”（仅 `--run-queries`）。
3. 运行自动化驱动器做“索引优化测试”（`--apply-indexes --run-queries`）。
4. 对比 `summary.csv` 与 `plans/*.txt`，分析 `elapsed_ms` 与 `Execution Time` 变化趋势，形成结论。

驱动器能力：

- 自动执行 `sql/indexes.sql`（索引实验）。
- 自动执行 `sql/query_variants.sql` 中的 `EXPLAIN (ANALYZE, BUFFERS)` 语句。
- 按重复次数（`--repeats`）执行并收集指标，输出 JSON/CSV/Markdown 汇总与执行计划文本。
- 在执行 `ANALYZE` 时，脚本会临时切换到 autocommit 模式，避免 openGauss 报出 “ANALYZE cannot run inside a transaction block”。

运行示例（索引 + 查询对比）：

```bash
python optimization_driver.py \
    --host 192.168.39.160 --port 7654 --dbname mydb \
    --user dbremote --password 'dbremote:399' \
    --apply-indexes --run-queries --repeats 3 --analyze-after-ddl
```

结果输出目录（自动按时间戳创建）：

- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.json`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.csv`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/summary.md`
- `Database-System/JDBC/query-optimization/experiment-results/<timestamp>/plans/query_XX_run_YY.txt`
