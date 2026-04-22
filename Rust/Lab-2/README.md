# Rust Lab-2: Web Server Tasks

This directory contains four separate Rust web server projects, each implementing a specific feature.

## Task 1: Simple HTTP Server

**Location**: `task1_hello_server/`

**Description**: A basic HTTP server that responds to requests on the `/hello` endpoint with "Hello, Rust!" text.

**Technologies**: actix-web

**How to run**:
```bash
cd task1_hello_server
cargo run
```

**Expected output**:
```
Starting HTTP server on http://127.0.0.1:8080
```

**Test the endpoint**:
```bash
curl http://127.0.0.1:8080/hello
```

---

## Task 2: Static Files Server

**Location**: `task2_static_files/`

**Description**: Extends the basic HTTP server to serve static files (HTML/CSS) and displays a custom welcome page at the root path.

**Technologies**: actix-web, actix-files

**Features**:
- GET `/` - Returns a beautiful HTML welcome page with CSS styling
- GET `/static/*` - Serves static files from the `static/` directory

**How to run**:
```bash
cd task2_static_files
cargo run
```

**Expected output**:
```
Starting HTTP server on http://127.0.0.1:8081
Visit / for welcome page or /static for static files
```

**Test the endpoints**:
```bash
# View welcome page
curl http://127.0.0.1:8081/

# Access static CSS file
curl http://127.0.0.1:8081/static/style.css
```

---

## Task 3: Book Management API

**Location**: `task3_book_api/`

**Description**: A REST API for managing books with in-memory data storage. Supports retrieving and adding books.

**Technologies**: actix-web, serde, serde_json, uuid

**Features**:
- GET `/books` - Retrieve all books
- POST `/books` - Add a new book

**How to run**:
```bash
cd task3_book_api
cargo run
```

**Expected output**:
```
Starting Book API server on http://127.0.0.1:8082
GET  /books - Get all books
POST /books - Add a new book
```

**Test the endpoints**:
```bash
# Get all books
curl http://127.0.0.1:8082/books

# Add a new book
curl -X POST http://127.0.0.1:8082/books \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Hello",
    "author": "Li Xiangxi",
    "year": 2005
  }'
```

**Example Response**:
```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "title": "Learning Rust",
  "author": "John Doe",
  "year": 2024
}
```

---

## Task 4: Middleware Logging

**Location**: `task4_middleware_logging/`

**Description**: A web server with custom middleware that logs each request's path, HTTP status code, and response duration. Uses the `tracing` library for structured logging.

**Technologies**: actix-web, tracing, tracing-subscriber

**Features**:
- Custom logging middleware that wraps all requests
- Logs path, method, status code, and response duration in milliseconds
- GET `/` - Hello endpoint
- GET `/hello` - Hello endpoint
- GET `/about` - About endpoint

**How to run**:
```bash
cd task4_middleware_logging
cargo run
```

**Expected output**:
```
Starting HTTP server with logging middleware on http://127.0.0.1:8083
```

**Test the endpoints** (each request will be logged):
```bash
curl http://127.0.0.1:8083/
curl http://127.0.0.1:8083/hello
curl http://127.0.0.1:8083/about
```

**Example Log Output**:
```
2024-04-22T10:30:45.123Z  INFO: Request processed path="/" method="GET" status=200 duration_ms=1
2024-04-22T10:30:46.456Z  INFO: Request processed path="/hello" method="GET" status=200 duration_ms=0
2024-04-22T10:30:47.789Z  INFO: Request processed path="/about" method="GET" status=200 duration_ms=1
```

---

## Building All Projects

To build all projects:
```bash
cd task1_hello_server && cargo build --release
cd ../task2_static_files && cargo build --release
cd ../task3_book_api && cargo build --release
cd ../task4_middleware_logging && cargo build --release
```
