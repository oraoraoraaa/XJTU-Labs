use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    web, App, HttpServer, HttpResponse, Error,
};
use futures::future::LocalBoxFuture;
use std::time::Instant;
use tracing::{info, Span};

async fn hello() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/plain; charset=utf-8")
        .body("Hello from Task 4!")
}

async fn about() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/plain; charset=utf-8")
        .body("This is the about page")
}

// Custom middleware for logging
pub struct LoggingMiddleware;

impl<S, B> Transform<S, ServiceRequest> for LoggingMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Transform = LoggingMiddlewareService<S>;
    type InitError = ();
    type Future = std::future::Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        std::future::ready(Ok(LoggingMiddlewareService { service }))
    }
}

pub struct LoggingMiddlewareService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for LoggingMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let start = Instant::now();
        let path = req.path().to_string();
        let method = req.method().to_string();

        let fut = self.service.call(req);

        Box::pin(async move {
            let res = fut.await?;
            let status = res.status();
            let duration = start.elapsed();

            info!(
                path = %path,
                method = %method,
                status = status.as_u16(),
                duration_ms = duration.as_millis(),
                "Request processed"
            );

            Ok(res)
        })
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting HTTP server with logging middleware on http://127.0.0.1:8083");

    HttpServer::new(|| {
        App::new()
            .wrap(LoggingMiddleware)
            .route("/", web::get().to(hello))
            .route("/hello", web::get().to(hello))
            .route("/about", web::get().to(about))
    })
    .bind("127.0.0.1:8083")?
    .run()
    .await
}
