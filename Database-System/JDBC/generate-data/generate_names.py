#!/usr/bin/env python3
"""
generate_names.py

Fetches realistic person names from RandomUser API and emits SQL insert
statements for S799 (students) and C799 (courses) along with optional
SC799 enrollments. This is a lightweight alternative to the Java
generator and can produce richer `SNAME` and `TEACHER` values.

Usage:
  pip install requests
  python3 generate_names.py --students 1000 --courses 100 --enrollments 20000 --outdir ./sql_out

The script will write files: students.sql, courses.sql, enrollments.sql
which can be loaded via psql or executed through JDBC.
"""
import argparse
import json
import os
import random
import sys

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests", file=sys.stderr)
    raise


API = "https://randomuser.me/api/"


def fetch_names(count):
    # randomuser supports up to 5000 per request in practice; keep batches
    names = []
    left = count
    while left > 0:
        n = min(500, left)
        r = requests.get(API, params={"results": n, "nat": "us,gb,cn"})
        r.raise_for_status()
        data = r.json()
        for it in data.get("results", []):
            name = it["name"]
            full = "{} {}".format(name.get("first", ""), name.get("last", ""))
            names.append(full)
        left -= n
    return names


def write_students(names, outpath, start_id=1):
    path = os.path.join(outpath, "students.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write("-- Generated students inserts\n")
        for i, n in enumerate(names, start=start_id):
            sid = f"{i:08d}"
            sex = random.choice(["男", "女"])
            year = random.randint(2000, 2006)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            bdate = f"{year:04d}-{month:02d}-{day:02d}"
            height = round(1.50 + random.random() * 0.50, 2)
            dorm = f"Dorm{random.randint(1,200)}"
            sql = ("INSERT INTO \"public\".\"S799\" (\"S_num\",\"SNAME\",\"SEX\",\"BDATE\",\"HEIGHT\",\"DORM\") "
                   f"VALUES ('{sid}','{n.replace("'","''")}', '{sex}', '{bdate}', {height}, '{dorm}');\n")
            f.write(sql)
    return path


def write_courses(count, outpath, start_id=1):
    path = os.path.join(outpath, "courses.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write("-- Generated courses inserts\n")
        for i in range(start_id, start_id + count):
            cid = f"C{i:05d}"
            prefix = random.choice(["CS", "EE", "ME"])
            cnum = f"{prefix}-{cid}"
            cname = f"{prefix} Course {i}"
            period = random.randint(20, 120)
            credit = random.randint(1, 5)
            teacher = f"Prof {random.choice(['Li', 'Wang', 'Zhang', 'Smith', 'Johnson'])} {random.randint(1,200)}"
            sql = ("INSERT INTO \"public\".\"C799\" (\"C_num\",\"CNAME\",\"PERIOD\",\"CREDIT\",\"TEACHER\") "
                   f"VALUES ('{cnum}','{cname}', {period}, {credit}, '{teacher}');\n")
            f.write(sql)
    return path


def write_enrollments(students, courses, total, outpath):
    path = os.path.join(outpath, "enrollments.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write("-- Generated enrollments inserts\n")
        for _ in range(total):
            sid = f"{random.randint(1, students):08d}"
            cid = random.choice([f"CS-C{random.randint(1,courses):05d}", f"EE-C{random.randint(1,courses):05d}"])
            if random.random() < 0.08:
                grade = 'NULL'
            else:
                grade = f"{round(random.uniform(50, 100), 1)}"
            sql = f"INSERT INTO \"public\".\"SC799\" (\"S_num\",\"C_num\",\"GRADE\") VALUES ('{sid}','{cid}', {grade});\n"
            f.write(sql)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--students', type=int, default=1000)
    p.add_argument('--courses', type=int, default=100)
    p.add_argument('--enrollments', type=int, default=20000)
    p.add_argument('--outdir', type=str, default='sql_out')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Fetching {args.students} student names...")
    student_names = fetch_names(args.students)
    sfile = write_students(student_names, args.outdir)
    print(f"Wrote students SQL to {sfile}")

    print(f"Generating {args.courses} courses...")
    cfile = write_courses(args.courses, args.outdir)
    print(f"Wrote courses SQL to {cfile}")

    print(f"Generating {args.enrollments} enrollments...")
    efile = write_enrollments(args.students, args.courses, args.enrollments, args.outdir)
    print(f"Wrote enrollments SQL to {efile}")


if __name__ == '__main__':
    main()
