use mysql::{params, prelude::Queryable, Pool, PooledConn};
use std::env;
use std::error::Error;

fn get_mysql_url() -> String {
    env::var("MYSQL_URL")
        .unwrap_or_else(|_| "mysql://root:root@127.0.0.1:3306/rust_lab".to_string())
}

fn ensure_database(url: &str) -> Result<String, Box<dyn Error>> {
    let opts = mysql::Opts::from_url(url)?;
    let db_name = opts
        .get_db_name()
        .map(|name| name.to_string())
        .unwrap_or_else(|| "rust_lab".to_string());

    let builder = mysql::OptsBuilder::from_opts(opts).db_name(None::<String>);
    let pool = Pool::new(builder)?;
    let mut conn = pool.get_conn()?;
    conn.query_drop(format!("CREATE DATABASE IF NOT EXISTS `{}`", db_name))?;

    Ok(db_name)
}

fn connect() -> Result<PooledConn, Box<dyn Error>> {
    let url = get_mysql_url();
    let db_name = ensure_database(&url)?;
    let opts = mysql::Opts::from_url(&url)?;
    let builder = mysql::OptsBuilder::from_opts(opts).db_name(Some(db_name));
    let pool = Pool::new(builder)?;
    Ok(pool.get_conn()?)
}

fn init_db(conn: &mut PooledConn) -> mysql::Result<()> {
    conn.query_drop(
        "CREATE TABLE IF NOT EXISTS orders (
            order_id INT AUTO_INCREMENT PRIMARY KEY,
            amount DECIMAL(10, 2) NOT NULL,
            created_at DATETIME NOT NULL
        )",
    )?;

    conn.query_drop(
        "CREATE OR REPLACE VIEW daily_sales_report AS
         SELECT DATE(created_at) AS report_date,
                COUNT(*) AS order_count,
                SUM(amount) AS total_amount,
                AVG(amount) AS avg_amount
         FROM orders
         GROUP BY DATE(created_at)",
    )?;

    Ok(())
}

fn seed_orders(conn: &mut PooledConn) -> mysql::Result<()> {
    conn.query_drop(
        "INSERT INTO orders (amount, created_at)
         VALUES (120.50, NOW())",
    )?;
    conn.query_drop(
        "INSERT INTO orders (amount, created_at)
         VALUES (75.00, NOW() - INTERVAL 1 DAY)",
    )?;
    conn.query_drop(
        "INSERT INTO orders (amount, created_at)
         VALUES (260.00, NOW() - INTERVAL 1 DAY)",
    )?;
    conn.query_drop(
        "INSERT INTO orders (amount, created_at)
         VALUES (30.99, NOW() - INTERVAL 2 DAY)",
    )?;
    conn.query_drop(
        "INSERT INTO orders (amount, created_at)
         VALUES (410.20, NOW() - INTERVAL 3 DAY)",
    )?;
    Ok(())
}

fn query_report(
    conn: &mut PooledConn,
    start_date: &str,
    end_date: &str,
) -> mysql::Result<Vec<(String, u64, String, String)>> {
    conn.exec_map(
        "SELECT DATE_FORMAT(report_date, '%Y-%m-%d') AS report_date,
                order_count,
                CAST(total_amount AS CHAR) AS total_amount,
                CAST(avg_amount AS CHAR) AS avg_amount
         FROM daily_sales_report
         WHERE report_date BETWEEN :start_date AND :end_date
         ORDER BY report_date",
        params! {
            "start_date" => start_date,
            "end_date" => end_date,
        },
        |(report_date, order_count, total_amount, avg_amount)| {
            (report_date, order_count, total_amount, avg_amount)
        },
    )
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run -- init
  cargo run -- seed
    cargo run -- report <YYYY-MM-DD> <YYYY-MM-DD>
    cargo run -- demo"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut conn = connect()?;
    init_db(&mut conn)?;

    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("init") => {
            println!("Table orders and view daily_sales_report are ready.");
        }
        Some("seed") => {
            seed_orders(&mut conn)?;
            println!("Seeded sample orders.");
        }
        Some("report") => {
            let start_date = match args.next() {
                Some(value) => value,
                None => {
                    print_usage();
                    return Ok(());
                }
            };
            let end_date = match args.next() {
                Some(value) => value,
                None => {
                    print_usage();
                    return Ok(());
                }
            };

            let rows = query_report(&mut conn, &start_date, &end_date)?;
            if rows.is_empty() {
                println!("No report rows for {start_date} to {end_date}.");
            } else {
                for (report_date, order_count, total_amount, avg_amount) in rows {
                    println!(
                        "date={} orders={} total={} avg={}",
                        report_date, order_count, total_amount, avg_amount
                    );
                }
            }
        }
        Some("demo") => {
            seed_orders(&mut conn)?;
            let start_date = "2000-01-01";
            let end_date = "2100-01-01";

            let rows = query_report(&mut conn, start_date, end_date)?;
            if rows.is_empty() {
                println!("No report rows for {start_date} to {end_date}.");
            } else {
                for (report_date, order_count, total_amount, avg_amount) in rows {
                    println!(
                        "date={} orders={} total={} avg={}",
                        report_date, order_count, total_amount, avg_amount
                    );
                }
            }
        }
        Some(other) => {
            eprintln!("Unknown command: {other}");
            print_usage();
        }
        None => {
            print_usage();
        }
    }

    Ok(())
}
