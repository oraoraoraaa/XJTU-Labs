-- indexes.sql
-- Suggested indexes to improve common queries on SC799

-- Index on course id for filtering by course
CREATE INDEX IF NOT EXISTS idx_sc799_cnum ON "public"."SC799" ("C_num");

-- Index on student id for filtering by student
CREATE INDEX IF NOT EXISTS idx_sc799_snum ON "public"."SC799" ("S_num");

-- Composite index if queries often filter by course then student
CREATE INDEX IF NOT EXISTS idx_sc799_cnum_snum ON "public"."SC799" ("C_num", "S_num");

-- Consider a partial index for non-null grades if queries frequently exclude NULLs
CREATE INDEX IF NOT EXISTS idx_sc799_grade_notnull ON "public"."SC799" ("GRADE") WHERE "GRADE" IS NOT NULL;
