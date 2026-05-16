# Rust Lab-3: Database Tasks (SQLite, MySQL, PostgreSQL)

This folder contains four standalone Rust CLI programs. Each program focuses on one database task from the assignment.

## Local Databases (Docker)
If you do not have MySQL/PostgreSQL running locally, you can start both with Docker:
```bash
cd Rust/Lab-3
docker-compose up -d
```

On macOS:

```bash
brew install colima docker
colima start
docker context use colima
docker-compose up -d
```

Default URLs used by the demos when env vars are not set:
- MySQL: `mysql://root:root@127.0.0.1:3306/rust_lab`
- PostgreSQL: `postgres://postgres:postgres@127.0.0.1:5432/rust_lab`

## Cleanup

```bash
docker-compose down
docker-compose down --rmi all
docker system prune
```

## Task 1: User Registration (SQLite)
**Location**: `task1_user_registration_sqlite/`

**Features**:
- Creates `users.db` and `users` table
- Inserts at least 5 test users
- Registers a new user
- Queries user info by username

**Run**:
```bash
cd task1_user_registration_sqlite
cargo run -- demo
cargo run -- seed
cargo run -- register alice2 pass123
cargo run -- query alice
```

## Task 2: Log System (MySQL)
**Location**: `task2_log_mysql/`

**Features**:
- Creates `app_logs` table
- Inserts logs of different levels
- Queries ERROR logs in the last 24 hours

**Environment (optional)**:
```bash
export MYSQL_URL="mysql://root:root@127.0.0.1:3306/rust_lab"
```

**Run**:
```bash
cd task2_log_mysql
cargo run -- demo
cargo run -- init
cargo run -- seed
cargo run -- log ERROR "db down"
cargo run -- recent-errors
```

## Task 3: Inventory Management (PostgreSQL)
**Location**: `task3_inventory_postgres/`

**Features**:
- Creates `products` table
- Purchases items by reducing stock
- Queries products with stock below a threshold

**Environment (optional)**:
```bash
export POSTGRES_URL="postgres://postgres:postgres@127.0.0.1:5432/rust_lab"
```

**Run**:
```bash
cd task3_inventory_postgres
cargo run -- demo
cargo run -- init
cargo run -- seed
cargo run -- purchase 1 3
cargo run -- low-stock 10
```

## Task 4: Daily Sales Report (MySQL)
**Location**: `task4_report_mysql/`

**Features**:
- Creates `orders` table
- Builds view `daily_sales_report` for daily aggregates
- Queries report data by date range

**Environment (optional)**:
```bash
export MYSQL_URL="mysql://root:root@127.0.0.1:3306/rust_lab"
```

**Run**:
```bash
cd task4_report_mysql
cargo run -- demo
cargo run -- init
cargo run -- seed
cargo run -- report 2024-01-01 2024-12-31
```
