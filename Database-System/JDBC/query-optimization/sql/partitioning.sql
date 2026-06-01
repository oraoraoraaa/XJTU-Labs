-- partitioning.sql
-- Example: partition SC799 by hash on S_num to spread data across partitions
-- Note: adjust partition count and bounds for your environment.

-- Create a new partitioned table (template). Do NOT run if you already have SC799 with data.
-- DROP TABLE IF EXISTS public.sc799_partitioned CASCADE;

CREATE TABLE IF NOT EXISTS public.sc799_partitioned (
    "S_num" VARCHAR(20),
    "C_num" VARCHAR(20),
    "GRADE" DECIMAL(4,1)
)
PARTITION BY HASH ("S_num");

-- Create 8 hash partitions as an example
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'sc799_p0') THEN
    EXECUTE 'CREATE TABLE public.sc799_p0 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 0)';
    EXECUTE 'CREATE TABLE public.sc799_p1 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 1)';
    EXECUTE 'CREATE TABLE public.sc799_p2 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 2)';
    EXECUTE 'CREATE TABLE public.sc799_p3 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 3)';
    EXECUTE 'CREATE TABLE public.sc799_p4 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 4)';
    EXECUTE 'CREATE TABLE public.sc799_p5 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 5)';
    EXECUTE 'CREATE TABLE public.sc799_p6 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 6)';
    EXECUTE 'CREATE TABLE public.sc799_p7 PARTITION OF public.sc799_partitioned FOR VALUES WITH (modulus 8, remainder 7)';
  END IF;
END$$;

-- To migrate data:
-- INSERT INTO public.sc799_partitioned SELECT * FROM public."SC799";
