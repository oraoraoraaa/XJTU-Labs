use actix_web::{web, App, HttpServer, HttpResponse};

async fn hello() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/plain; charset=utf-8")
        .body("Hello, Rust!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting HTTP server on http://127.0.0.1:8080");
    
    HttpServer::new(|| {
        App::new()
            .route("/hello", web::get().to(hello))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
