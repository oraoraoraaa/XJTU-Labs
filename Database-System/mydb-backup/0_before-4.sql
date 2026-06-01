--
-- openGauss database dump
--

SET statement_timeout = 0;
SET xmloption = content;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET session_replication_role = replica;
SET client_min_messages = warning;
SET enable_dump_trigger_definer = on;

SET search_path = public;

ALTER TABLE IF EXISTS public."SC799" DROP CONSTRAINT IF EXISTS "SC799_S_num_fkey";
ALTER TABLE IF EXISTS public."SC799" DROP CONSTRAINT IF EXISTS "SC799_C_num_fkey";
ALTER TABLE IF EXISTS public."SC799" DROP CONSTRAINT IF EXISTS "SC799_pkey";
ALTER TABLE IF EXISTS public."S799" DROP CONSTRAINT IF EXISTS "S799_pkey";
ALTER TABLE IF EXISTS public."C799" DROP CONSTRAINT IF EXISTS "C799_pkey";
DROP VIEW IF EXISTS public.view_teacher_zhang_courses;
DROP VIEW IF EXISTS public.view_male_dorm18;
DROP VIEW IF EXISTS public.view_ai_students;
DROP TABLE IF EXISTS public."SC799";
DROP TABLE IF EXISTS public."S799";
DROP TABLE IF EXISTS public."C799";
--
-- Name: BEHAVIORCOMPAT; Type: BEHAVIORCOMPAT; Schema: -; Owner: 
--

SET behavior_compat_options = '';


SET search_path = public;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: C799; Type: TABLE; Schema: public; Owner: dbremote; Tablespace: 
--

CREATE TABLE "C799" (
    "C_num" character varying(20) NOT NULL,
    "CNAME" character varying(100) NOT NULL,
    "PERIOD" integer,
    "CREDIT" numeric(3,1),
    "TEACHER" character varying(50)
)
WITH (orientation=row, compression=no);


ALTER TABLE public."C799" OWNER TO dbremote;

--
-- Name: S799; Type: TABLE; Schema: public; Owner: dbremote; Tablespace: 
--

CREATE TABLE "S799" (
    "S_num" character varying(20) NOT NULL,
    "SNAME" character varying(50) NOT NULL,
    "SEX" character varying(10),
    "BDATE" timestamp(0) without time zone,
    "HEIGHT" numeric(3,2),
    "DORM" character varying(100)
)
WITH (orientation=row, compression=no);


ALTER TABLE public."S799" OWNER TO dbremote;

--
-- Name: SC799; Type: TABLE; Schema: public; Owner: dbremote; Tablespace: 
--

CREATE TABLE "SC799" (
    "S_num" character varying(20) NOT NULL,
    "C_num" character varying(20) NOT NULL,
    "GRADE" numeric(4,1)
)
WITH (orientation=row, compression=no);


ALTER TABLE public."SC799" OWNER TO dbremote;

--
-- Name: view_ai_students; Type: VIEW; Schema: public; Owner: dbremote
--

CREATE VIEW view_ai_students("S_num","SNAME","GRADE") AS
    SELECT s."S_num", s."SNAME", sc."GRADE" FROM (("S799" s JOIN "SC799" sc ON (((s."S_num")::text = (sc."S_num")::text))) JOIN "C799" c ON (((sc."C_num")::text = (c."C_num")::text))) WHERE ((c."CNAME")::text = '人工智能'::text);


ALTER VIEW public.view_ai_students OWNER TO dbremote;

--
-- Name: view_male_dorm18; Type: VIEW; Schema: public; Owner: dbremote
--

CREATE VIEW view_male_dorm18("S_num","SNAME","BDATE","HEIGHT","SEX","DORM") AS
    SELECT "S799"."S_num", "S799"."SNAME", "S799"."BDATE", "S799"."HEIGHT", "S799"."SEX", "S799"."DORM" FROM "S799" WHERE ((("S799"."SEX")::text = '男'::text) AND (("S799"."DORM")::text ~~ '东18舍%'::text));


ALTER VIEW public.view_male_dorm18 OWNER TO dbremote;

--
-- Name: view_teacher_zhang_courses; Type: VIEW; Schema: public; Owner: dbremote
--

CREATE VIEW view_teacher_zhang_courses("C_num","CNAME","AVG_GRADE") AS
    SELECT c."C_num", c."CNAME", avg(sc."GRADE") AS "AVG_GRADE" FROM ("C799" c LEFT JOIN "SC799" sc ON (((c."C_num")::text = (sc."C_num")::text))) WHERE ((c."TEACHER")::text = '张明'::text) GROUP BY c."C_num", c."CNAME";


ALTER VIEW public.view_teacher_zhang_courses OWNER TO dbremote;

--
-- Data for Name: C799; Type: TABLE DATA; Schema: public; Owner: dbremote
--

COPY public."C799" ("C_num", "CNAME", "PERIOD", "CREDIT", "TEACHER") FROM stdin;
CS-01	数据结构	60	3.0	张军
CS-02	计算机组成原理	80	4.0	王亚伟
CS-04	人工智能	40	2.0	李蕾
CS-05	深度学习	40	2.0	崔昀
EE-01	信号与系统	60	3.0	张明
EE-02	数字逻辑电路	100	5.0	胡海东
EE-03	光电子学与光子学	40	2.0	石韬
CS-03	离散数学	64	4.0	陈建明
\.
;

--
-- Data for Name: S799; Type: TABLE DATA; Schema: public; Owner: dbremote
--

COPY public."S799" ("S_num", "SNAME", "SEX", "BDATE", "HEIGHT", "DORM") FROM stdin;
01032010	王涛	男	2004-04-05 00:00:00	1.72	东6舍221
01032023	孙文	男	2005-06-10 00:00:00	1.80	东6舍221
01032001	张晓梅	女	2004-11-17 00:00:00	1.58	东1舍312
01032005	刘静	女	2004-01-10 00:00:00	1.63	东1舍312
01032112	许澍	男	2004-02-20 00:00:00	1.71	东6舍221
03031011	王倩	女	2005-09-20 00:00:00	1.66	东2舍104
03031014	赵思扬	男	2003-06-06 00:00:00	1.85	东18舍421
03031051	周剑	男	2003-05-08 00:00:00	1.68	东18舍422
03031009	田菲	女	2004-08-11 00:00:00	1.60	东2舍104
03031033	蔡明明	男	2004-03-12 00:00:00	1.75	东18舍423
03031056	曹子衿	女	2006-12-15 00:00:00	1.65	东2舍305
\.
;

--
-- Data for Name: SC799; Type: TABLE DATA; Schema: public; Owner: dbremote
--

COPY public."SC799" ("S_num", "C_num", "GRADE") FROM stdin;
01032010	CS-01	82.0
01032010	CS-02	91.0
01032010	CS-04	83.5
01032001	CS-01	77.5
01032001	CS-02	85.0
01032001	CS-04	83.0
01032005	CS-01	62.0
01032005	CS-02	77.0
01032005	CS-04	82.0
01032023	CS-01	55.0
01032023	CS-02	81.0
01032023	CS-04	76.0
01032112	CS-01	88.0
01032112	CS-02	91.5
01032112	CS-04	86.0
01032112	CS-05	\N
03031033	EE-01	93.0
03031033	EE-02	89.0
03031009	EE-01	88.0
03031009	EE-02	78.5
03031011	EE-01	91.0
03031011	EE-02	86.0
03031051	EE-01	78.0
03031051	EE-02	58.0
03031014	EE-01	79.0
03031014	EE-02	71.0
\.
;

--
-- Name: C799_pkey; Type: CONSTRAINT; Schema: public; Owner: dbremote; Tablespace: 
--

ALTER TABLE "C799"
    ADD CONSTRAINT "C799_pkey" PRIMARY KEY  ("C_num");


--
-- Name: S799_pkey; Type: CONSTRAINT; Schema: public; Owner: dbremote; Tablespace: 
--

ALTER TABLE "S799"
    ADD CONSTRAINT "S799_pkey" PRIMARY KEY  ("S_num");


--
-- Name: SC799_pkey; Type: CONSTRAINT; Schema: public; Owner: dbremote; Tablespace: 
--

ALTER TABLE "SC799"
    ADD CONSTRAINT "SC799_pkey" PRIMARY KEY  ("S_num", "C_num");


--
-- Name: SC799_C_num_fkey; Type: FK CONSTRAINT; Schema: public; Owner: dbremote
--

ALTER TABLE "SC799"
    ADD CONSTRAINT "SC799_C_num_fkey" FOREIGN KEY ("C_num") REFERENCES "C799"("C_num");


--
-- Name: SC799_S_num_fkey; Type: FK CONSTRAINT; Schema: public; Owner: dbremote
--

ALTER TABLE "SC799"
    ADD CONSTRAINT "SC799_S_num_fkey" FOREIGN KEY ("S_num") REFERENCES "S799"("S_num");


--
-- Name: public; Type: ACL; Schema: -; Owner: opengauss
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM opengauss;
GRANT CREATE,USAGE ON SCHEMA public TO opengauss;
GRANT USAGE ON SCHEMA public TO PUBLIC;


--
-- openGauss database dump complete
--

