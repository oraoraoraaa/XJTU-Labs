use rusqlite::{params, Connection};
use std::env;
use std::error::Error;

#[derive(Debug)]
struct User {
    id: i64,
    username: String,
    password: String,
    created_at: String,
}

fn init_db(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    Ok(())
}

fn seed_users(conn: &Connection) -> rusqlite::Result<usize> {
    let samples = [
        ("alice", "alice123"),
        ("bob", "bob123"),
        ("carol", "carol123"),
        ("dave", "dave123"),
        ("erin", "erin123"),
    ];

    let mut inserted = 0;
    for (username, password) in samples {
        let changes = conn.execute(
            "INSERT OR IGNORE INTO users (username, password, created_at)
             VALUES (?1, ?2, CURRENT_TIMESTAMP)",
            params![username, password],
        )?;
        inserted += changes;
    }

    Ok(inserted)
}

fn register_user(conn: &Connection, username: &str, password: &str) -> rusqlite::Result<bool> {
    let changes = conn.execute(
        "INSERT OR IGNORE INTO users (username, password, created_at)
         VALUES (?1, ?2, CURRENT_TIMESTAMP)",
        params![username, password],
    )?;
    Ok(changes > 0)
}

fn query_user(conn: &Connection, username: &str) -> rusqlite::Result<Option<User>> {
    let mut stmt = conn.prepare(
        "SELECT id, username, password, created_at
         FROM users
         WHERE username = ?1",
    )?;
    let mut rows = stmt.query(params![username])?;
    if let Some(row) = rows.next()? {
        Ok(Some(User {
            id: row.get(0)?,
            username: row.get(1)?,
            password: row.get(2)?,
            created_at: row.get(3)?,
        }))
    } else {
        Ok(None)
    }
}

fn list_users(conn: &Connection) -> rusqlite::Result<Vec<User>> {
    let mut stmt = conn.prepare(
        "SELECT id, username, password, created_at
         FROM users
         ORDER BY id ASC",
    )?;
    let users_iter = stmt.query_map([], |row| {
        Ok(User {
            id: row.get(0)?,
            username: row.get(1)?,
            password: row.get(2)?,
            created_at: row.get(3)?,
        })
    })?;

    let mut users = Vec::new();
    for user in users_iter {
        users.push(user?);
    }

    Ok(users)
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run -- seed
  cargo run -- register <username> <password>
  cargo run -- query <username>
    cargo run -- list
  cargo run -- demo"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let db_path = "users.db";
    let conn = Connection::open(db_path)?;
    init_db(&conn)?;

    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("seed") => {
            let inserted = seed_users(&conn)?;
            println!("Seeded {inserted} users into {db_path}.");
        }
        Some("register") => {
            let username = match args.next() {
                Some(value) => value,
                None => {
                    print_usage();
                    return Ok(());
                }
            };
            let password = match args.next() {
                Some(value) => value,
                None => {
                    print_usage();
                    return Ok(());
                }
            };

            if register_user(&conn, &username, &password)? {
                println!("User registered: {username}");
            } else {
                println!("Username already exists: {username}");
            }
        }
        Some("query") => {
            let username = match args.next() {
                Some(value) => value,
                None => {
                    print_usage();
                    return Ok(());
                }
            };

            match query_user(&conn, &username)? {
                Some(user) => println!(
                    "User: id={} username={} password={} created_at={}",
                    user.id, user.username, user.password, user.created_at
                ),
                None => println!("User not found: {username}"),
            }
        }
        Some("list") => {
            let users = list_users(&conn)?;
            if users.is_empty() {
                println!("No users found.");
            } else {
                for user in users {
                    println!(
                        "User: id={} username={} password={} created_at={}",
                        user.id, user.username, user.password, user.created_at
                    );
                }
            }
        }
        Some("demo") | None => {
            seed_users(&conn)?;
            let username = "alice";
            println!("Demo query for username: {username}");
            match query_user(&conn, username)? {
                Some(user) => println!(
                    "User: id={} username={} password={} created_at={}",
                    user.id, user.username, user.password, user.created_at
                ),
                None => println!("User not found: {username}"),
            }
        }
        Some(other) => {
            eprintln!("Unknown command: {other}");
            print_usage();
        }
    }

    Ok(())
}
