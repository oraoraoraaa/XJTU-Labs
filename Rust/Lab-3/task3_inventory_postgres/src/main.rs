use postgres::{Client, NoTls};
use std::env;
use std::error::Error;

fn get_postgres_url() -> String {
    env::var("POSTGRES_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@127.0.0.1:5432/rust_lab".to_string())
}

fn split_postgres_url(url: &str) -> Result<(String, String, String), Box<dyn Error>> {
    let (main, params) = match url.split_once('?') {
        Some((left, right)) => (left, right),
        None => (url, ""),
    };
    let slash_pos = main
        .rfind('/')
        .ok_or("POSTGRES_URL must include a database name")?;
    let base = &main[..slash_pos];
    let db_part = &main[slash_pos + 1..];
    let db_name = if db_part.is_empty() { "rust_lab" } else { db_part };
    let params_suffix = if params.is_empty() {
        String::new()
    } else {
        format!("?{}", params)
    };

    Ok((base.to_string(), db_name.to_string(), params_suffix))
}

fn build_postgres_url(base: &str, db_name: &str, params: &str) -> String {
    format!("{}/{}{}", base, db_name, params)
}

fn escape_identifier(value: &str) -> String {
    value.replace('"', "\"\"")
}

fn ensure_database(url: &str) -> Result<String, Box<dyn Error>> {
    let (base, db_name, params) = split_postgres_url(url)?;
    let admin_url = build_postgres_url(&base, "postgres", &params);
    let mut admin = Client::connect(&admin_url, NoTls)?;

    let rows = admin.query(
        "SELECT 1 FROM pg_database WHERE datname = $1",
        &[&db_name],
    )?;
    if rows.is_empty() {
        let escaped = escape_identifier(&db_name);
        admin.batch_execute(&format!("CREATE DATABASE \"{}\"", escaped))?;
    }

    Ok(db_name)
}

fn connect() -> Result<Client, Box<dyn Error>> {
    let url = get_postgres_url();
    let db_name = ensure_database(&url)?;
    let (base, _db_name, params) = split_postgres_url(&url)?;
    let db_url = build_postgres_url(&base, &db_name, &params);
    Ok(Client::connect(&db_url, NoTls)?)
}

fn init_db(client: &mut Client) -> Result<(), postgres::Error> {
    client.batch_execute(
        "CREATE TABLE IF NOT EXISTS products (
            product_id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            stock INT NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )",
    )?;
    Ok(())
}

fn seed_products(client: &mut Client) -> Result<(), postgres::Error> {
    let samples = [
        ("Keyboard", 25),
        ("Mouse", 8),
        ("Monitor", 12),
        ("USB Cable", 5),
        ("Laptop Stand", 9),
    ];

    for (name, stock) in samples {
        client.execute(
            "INSERT INTO products (name, stock) VALUES ($1, $2)",
            &[&name, &stock],
        )?;
    }

    Ok(())
}

fn purchase_product(client: &mut Client, product_id: i32, qty: i32) -> Result<(), postgres::Error> {
    let row = client.query_opt(
        "UPDATE products
         SET stock = stock - $1,
             updated_at = NOW()
         WHERE product_id = $2
           AND stock >= $1
         RETURNING product_id, name, stock, to_char(updated_at, 'YYYY-MM-DD HH24:MI:SS')",
        &[&qty, &product_id],
    )?;

    match row {
        Some(row) => {
            let id: i32 = row.get(0);
            let name: String = row.get(1);
            let stock: i32 = row.get(2);
            let updated_at: String = row.get(3);
            println!(
                "Purchase OK: id={} name={} stock={} updated_at={}",
                id, name, stock, updated_at
            );
        }
        None => {
            println!(
                "Purchase failed: product_id={} qty={} (insufficient stock or not found)",
                product_id, qty
            );
        }
    }

    Ok(())
}

fn query_low_stock(
    client: &mut Client,
    threshold: i32,
) -> Result<Vec<(i32, String, i32, String)>, postgres::Error> {
    let rows = client.query(
        "SELECT product_id, name, stock, to_char(updated_at, 'YYYY-MM-DD HH24:MI:SS')
         FROM products
         WHERE stock < $1
         ORDER BY stock ASC",
        &[&threshold],
    )?;

    Ok(rows
        .into_iter()
        .map(|row| {
            let id: i32 = row.get(0);
            let name: String = row.get(1);
            let stock: i32 = row.get(2);
            let updated_at: String = row.get(3);
            (id, name, stock, updated_at)
        })
        .collect())
}

fn print_usage() {
    eprintln!(
        "Usage:
  cargo run -- init
  cargo run -- seed
  cargo run -- purchase <product_id> <qty>
    cargo run -- low-stock <threshold>
    cargo run -- demo"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut client = connect()?;
    init_db(&mut client)?;

    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("init") => {
            println!("Table products is ready.");
        }
        Some("seed") => {
            seed_products(&mut client)?;
            println!("Seeded sample products.");
        }
        Some("purchase") => {
            let product_id: i32 = match args.next() {
                Some(value) => value.parse().unwrap_or(0),
                None => {
                    print_usage();
                    return Ok(());
                }
            };
            let qty: i32 = match args.next() {
                Some(value) => value.parse().unwrap_or(0),
                None => {
                    print_usage();
                    return Ok(());
                }
            };

            if product_id <= 0 || qty <= 0 {
                eprintln!("product_id and qty must be positive.");
                return Ok(());
            }

            purchase_product(&mut client, product_id, qty)?;
        }
        Some("low-stock") => {
            let threshold: i32 = match args.next() {
                Some(value) => value.parse().unwrap_or(10),
                None => {
                    print_usage();
                    return Ok(());
                }
            };

            let rows = query_low_stock(&mut client, threshold)?;
            if rows.is_empty() {
                println!("No products with stock below {threshold}.");
            } else {
                for (id, name, stock, updated_at) in rows {
                    println!(
                        "product_id={} name={} stock={} updated_at={}",
                        id, name, stock, updated_at
                    );
                }
            }
        }
        Some("demo") => {
            seed_products(&mut client)?;
            purchase_product(&mut client, 1, 20)?;

            let rows = query_low_stock(&mut client, 10)?;
            if rows.is_empty() {
                println!("No products with stock below 10.");
            } else {
                for (id, name, stock, updated_at) in rows {
                    println!(
                        "product_id={} name={} stock={} updated_at={}",
                        id, name, stock, updated_at
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
