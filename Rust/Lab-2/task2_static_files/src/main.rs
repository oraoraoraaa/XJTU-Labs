use actix_web::{web, App, HttpServer, HttpResponse};
use actix_files as fs;

async fn welcome() -> HttpResponse {
    let html = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            text-align: center;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        h1 {
            color: #333;
            margin: 0;
        }
        p {
            color: #666;
            margin: 10px 0 0 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welcome to Rust Web Server!</h1>
        <p>Task 2: Serving Static Files</p>
    </div>
</body>
</html>
    "#;
    
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(html)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting HTTP server on http://127.0.0.1:8081");
    println!("Visit / for welcome page or /static for static files");
    
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(welcome))
            .service(fs::Files::new("/static", "./static"))
    })
    .bind("127.0.0.1:8081")?
    .run()
    .await
}
