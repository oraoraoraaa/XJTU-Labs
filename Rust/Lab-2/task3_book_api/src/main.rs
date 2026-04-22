use actix_web::{web, App, HttpServer, HttpResponse};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Book {
    pub id: String,
    pub title: String,
    pub author: String,
    pub year: u32,
}

#[derive(Debug, Deserialize)]
pub struct CreateBookRequest {
    pub title: String,
    pub author: String,
    pub year: u32,
}

pub type BookStore = Arc<Mutex<Vec<Book>>>;

async fn get_books(data: web::Data<BookStore>) -> HttpResponse {
    let books = data.lock().unwrap();
    HttpResponse::Ok().json(books.clone())
}

async fn add_book(
    book_req: web::Json<CreateBookRequest>,
    data: web::Data<BookStore>,
) -> HttpResponse {
    let new_book = Book {
        id: Uuid::new_v4().to_string(),
        title: book_req.title.clone(),
        author: book_req.author.clone(),
        year: book_req.year,
    };

    let mut books = data.lock().unwrap();
    books.push(new_book.clone());

    HttpResponse::Created().json(new_book)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let book_store = web::Data::new(Arc::new(Mutex::new(vec![
        Book {
            id: "1".to_string(),
            title: "The Rust Programming Language".to_string(),
            author: "Steve Klabnik and Carol Nichols".to_string(),
            year: 2018,
        },
        Book {
            id: "2".to_string(),
            title: "Programming in Rust".to_string(),
            author: "Jim Blandy and Jason Orendorff".to_string(),
            year: 2017,
        },
    ])));

    println!("Starting Book API server on http://127.0.0.1:8082");
    println!("GET  /books - Get all books");
    println!("POST /books - Add a new book");

    HttpServer::new(move || {
        App::new()
            .app_data(book_store.clone())
            .route("/books", web::get().to(get_books))
            .route("/books", web::post().to(add_book))
    })
    .bind("127.0.0.1:8082")?
    .run()
    .await
}
