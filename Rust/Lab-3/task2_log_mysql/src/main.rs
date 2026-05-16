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
        "CREATE TABLE IF NOT EXISTS app_logs (
            log_id INT AUTO_INCREMENT PRIMARY KEY,
            log_level ENUM('INFO','WARN','ERROR') NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
    )?;
    Ok(())
}

fn normalize_level(level: &str) -> Option<String> {
    let upper = level.to_ascii_uppercase();
    match upper.as_str() {
        "INFO" | "WARN" | "ERROR" => Some(upper),
        _ => None,
    }
}

fn insert_log(conn: &mut PooledConn, level: &str, content: &str) -> mysql::Result<()> {
    conn.exec_drop(
        "INSERT INTO app_logs (log_level, content) VALUES (:level, :content)",
        params! {
            "level" => level,
            "content" => content,
        },
    )?;
    Ok(())
}

fn seed_logs(conn: &mut PooledConn) -> mysql::Result<()> {
    conn.query_drop(
        "INSERT INTO app_logs (log_level, content, created_at)
         VALUES ('INFO', 'service started', NOW())",
    )?;
    conn.query_drop(
        "INSERT INTO app_logs (log_level, content, created_at)
         VALUES ('WARN', 'cache miss', NOW() - INTERVAL 2 HOUR)",
    )?;
    conn.query_drop(
        "INSERT INTO app_logs (log_level, content, created_at)
         VALUES ('ERROR', 'db timeout', NOW() - INTERVAL 1 HOUR)",
    )?;
    conn.query_drop(
        "INSERT INTO app_logs (log_level, content, created_at)
         VALUES ('INFO', 'health check OK', NOW() - INTERVAL 5 HOUR)",
    )?;
    conn.query_drop(
        "INSERT INTO app_logs (log_level, content, created_at)
         VALUES ('ERROR', 'legacy error', NOW() - INTERVAL 25 HOUR)",
    )?;
    Ok(())
}

fn query_recent_errors(conn: &mut PooledConn) -> mysql::Result<Vec<(u64, String, String, String)>> {
    conn.query_map(
        "SELECT log_id, log_level, content,
                DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') AS created_at
         FROM app_logs
         WHERE log_level = 'ERROR'
           AND created_at >= NOW() - INTERVAL 24 HOUR
         ORDER BY created_at DESC",
        |(log_id, log_level, content, created_at)| (log_id, log_level, content, created_at),
    )
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run -- init
  cargo run -- seed
  cargo run -- log <INFO|WARN|ERROR> <content>
    cargo run -- recent-errors
    cargo run -- demo"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut conn = connect()?;
    init_db(&mut conn)?;

    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("init") => {
            println!("Table app_logs is ready.");
        }
        Some("seed") => {
            seed_logs(&mut conn)?;
            println!("Seeded sample logs.");
        }
        Some("log") => {
            let level = match args.next() {
                Some(value) => value,
                None => {
                    print_usage();
                    return Ok(());
                }
            };
            let content = match args.next() {
                Some(value) => value,
                None => {
                    print_usage();
                    return Ok(());
                }
            };

            let level = match normalize_level(&level) {
                Some(value) => value,
                None => {
                    eprintln!("Invalid level: {level}");
                    return Ok(());
                }
            };

            insert_log(&mut conn, &level, &content)?;
            println!("Inserted log: level={level} content={content}");
        }
        Some("recent-errors") => {
            let rows = query_recent_errors(&mut conn)?;
            if rows.is_empty() {
                println!("No ERROR logs in the last 24 hours.");
            } else {
                for (log_id, log_level, content, created_at) in rows {
                    println!(
                        "log_id={} level={} created_at={} content={}",
                        log_id, log_level, created_at, content
                    );
                }
            }
        }
        Some("demo") => {
            seed_logs(&mut conn)?;
            insert_log(&mut conn, "INFO", "demo info")?;
            insert_log(&mut conn, "WARN", "demo warn")?;
            insert_log(&mut conn, "ERROR", "demo error")?;

            let rows = query_recent_errors(&mut conn)?;
            println!("Recent ERROR logs (last 24 hours):");
            if rows.is_empty() {
                println!("No ERROR logs in the last 24 hours.");
            } else {
                for (log_id, log_level, content, created_at) in rows {
                    println!(
                        "log_id={} level={} created_at={} content={}",
                        log_id, log_level, created_at, content
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
