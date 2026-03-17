import socket
import sys
import time
import os
import re
import hashlib
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser


class EmbeddedResourceParser(HTMLParser):
    """Collect embedded HTTP resources from HTML tags."""

    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag in {"img", "script", "iframe", "source", "audio", "video"}:
            src = attrs_dict.get("src")
            if src:
                self.urls.add(src)

        if tag == "video":
            poster = attrs_dict.get("poster")
            if poster:
                self.urls.add(poster)

        if tag == "link":
            href = attrs_dict.get("href")
            if href:
                self.urls.add(href)


def split_http_response(response_bytes):
    """Split raw HTTP response into header bytes and body bytes."""
    separator = b"\r\n\r\n"
    idx = response_bytes.find(separator)
    if idx == -1:
        return b"", response_bytes
    return response_bytes[:idx], response_bytes[idx + len(separator) :]


def parse_headers(header_bytes):
    headers = {}
    if not header_bytes:
        return headers

    lines = header_bytes.decode("iso-8859-1", errors="replace").split("\r\n")
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return headers


def extract_charset(content_type):
    if not content_type:
        return "utf-8"
    for part in content_type.split(";"):
        part = part.strip().lower()
        if part.startswith("charset="):
            return part.split("=", 1)[1].strip()
    return "utf-8"


def css_embedded_urls(css_text):
    """Extract URLs used in CSS url(...) declarations."""
    urls = set()
    for match in re.findall(r"url\((.*?)\)", css_text, flags=re.IGNORECASE):
        candidate = match.strip().strip('"').strip("'")
        if candidate and not candidate.lower().startswith("data:"):
            urls.add(candidate)
    return urls


def make_local_path(output_dir, url):
    parsed = urlparse(url)
    raw_path = parsed.path or "/"

    if raw_path.endswith("/"):
        raw_path += "index.html"

    relative_path = raw_path.lstrip("/")
    if not relative_path:
        relative_path = "index.html"

    # Avoid collisions for resources that differ only by query string.
    if parsed.query:
        stem, ext = os.path.splitext(relative_path)
        query_hash = hashlib.md5(parsed.query.encode("utf-8")).hexdigest()[:10]
        relative_path = f"{stem}_{query_hash}{ext}"

    local_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path


def make_http_request(hostname, port, path):
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {hostname}\r\n"
        f"User-Agent: Li Xiangxi\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )

    response = b""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    try:
        sock.connect((hostname, port))
        connected = True
        sock.sendall(request.encode())
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
    finally:
        sock.close()
    return response, connected


def download_resource(url, base_hostname, base_port, output_dir, visited):
    absolute_url = url
    parsed = urlparse(absolute_url)

    if parsed.scheme not in {"", "http"}:
        return [], 0

    hostname = parsed.hostname or base_hostname
    port = parsed.port or base_port

    # Keep this lab focused on one HTTP site.
    if hostname != base_hostname:
        return [], 0

    path = parsed.path if parsed.path else "/"
    if parsed.query:
        path += f"?{parsed.query}"

    visit_key = f"http://{hostname}:{port}{path}"
    if visit_key in visited:
        return [], 0
    visited.add(visit_key)

    try:
        response, connected = make_http_request(hostname, port, path)
    except Exception as e:
        print(f"Skip {absolute_url}: {e}")
        return [], 0

    header_bytes, body_bytes = split_http_response(response)
    headers = parse_headers(header_bytes)
    content_type = headers.get("content-type", "")

    local_path = make_local_path(output_dir, visit_key)
    with open(local_path, "wb") as f:
        f.write(body_bytes)

    discovered_urls = []
    if "text/css" in content_type:
        charset = extract_charset(content_type)
        css_text = body_bytes.decode(charset, errors="replace")
        discovered_urls.extend(css_embedded_urls(css_text))

    return [urljoin(visit_key, u) for u in discovered_urls], (1 if connected else 0)


def download_complete_page(received_data, hostname, port):
    """Save main HTML and download all directly embedded elements."""
    print("\n[4]  Download Complete Page")
    output_dir = "downloaded_page"
    os.makedirs(output_dir, exist_ok=True)

    header_bytes, body_bytes = split_http_response(received_data)
    headers = parse_headers(header_bytes)
    content_type = headers.get("content-type", "")
    charset = extract_charset(content_type)

    main_page_path = os.path.join(output_dir, "index.html")
    with open(main_page_path, "wb") as f:
        f.write(body_bytes)

    html_text = body_bytes.decode(charset, errors="replace")
    parser = EmbeddedResourceParser()
    parser.feed(html_text)

    base_url = f"http://{hostname}:{port}/"
    pending = [urljoin(base_url, u) for u in parser.urls]
    visited = set()

    downloaded = 0
    tcp_connection_count = 0
    while pending:
        resource_url = pending.pop(0)
        newly_found, resource_tcp_connections = download_resource(
            resource_url,
            hostname,
            port,
            output_dir,
            visited,
        )
        tcp_connection_count += resource_tcp_connections
        downloaded += 1 if newly_found is not None and resource_url in visited else 0
        pending.extend(newly_found)

    print(f"Main page saved: {main_page_path}")
    print(f"Embedded resources downloaded: {len(visited)}")
    return tcp_connection_count


def resolve_dns(hostname):
    """
    DNS Resolve. Get the IP address based on the hostname.
    Args:
        hostname: The domain (hostname) of the website to be resolved.
    Returns:
        tuple: (ip_address, dns_time)
    """
    print("\n[1]  DNS Resolve")
    print(f"Hostname: {hostname}")

    try:
        start_time = time.time()
        ip_address = socket.gethostbyname(hostname)
        dns_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"IPV4: {ip_address}")
        return (ip_address, dns_time)

    except Exception as e:
        print(f"Error: resolve_dns(): {e}")
        sys.exit(1)


def tcp_connection(ip_address, port):
    """
    Make TCP connection to the given hostname and port.
    Args:
        ip_address: The IPV4 address of the website to be connected.
        port: The port to connect
    Returns:
        tuple: (socket_object, tcp_time)
    """
    print("\n[2]  TCP Connection")
    print(f"Target: {ip_address}:{port}")
    try:
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )  # New socket object with IPV4 connection via TCP

        start_time = time.time()
        sock.connect((ip_address, port))
        tcp_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return sock, tcp_time
    except Exception as e:
        print(f"Error: tcp_connection(): {e}")
        sys.exit(1)


def http_request(sock, hostname, port):
    """
    Send HTTP request to and receive from the target host.
    Args:
        sock: A socket object on which to be made HTTP request
        hostname: The domain (hostname) of the website to connect
        port: The port to connect
    Returns:
        tuple: (received_data, http_time)
    """
    print("\n[3]  HTTP Request")

    request = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {hostname}\r\n"
        f"User-Agent: Li Xiangxi\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    )

    received_data = b""

    try:
        start_time = time.time()
        sock.sendall(request.encode())
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            received_data += chunk

        http_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return (received_data, http_time)
    except Exception as e:
        print(f"Error: http_request(): {e}")
        sys.exit(1)
    finally:
        sock.close()
        print("Connection closed.")


def main():
    # verify arguments
    if len(sys.argv) < 2:
        print("Usage: python http_client.py <server_name> [port]")
        print("Example: python http_client.py www.example.com 80")
        sys.exit(1)

    hostname = sys.argv[1]
    port = int(sys.argv[2])

    default_choice = (
        input(f"Make connection to {hostname}:{port}? (y/n): ").strip().lower()
    )
    if default_choice == "":
        default_choice = "y"

    if default_choice != "y":
        print("Operation cancelled by user.")
        sys.exit(0)

    print("Connecting...")

    start_time = time.time()

    # STEP 1: Resolve DNS
    ip_address, dns_time = resolve_dns(hostname)

    # STEP 2: Make TCP Connection
    sock, tcp_time = tcp_connection(ip_address, port)

    # STEP 3: HTTP Request
    received_data, http_time = http_request(sock, hostname, port)

    total_time = (time.time() - start_time) * 1000

    # STEP 4: Download main page and embedded resources
    resource_tcp_connections = download_complete_page(received_data, hostname, port)
    total_tcp_connections = 1 + resource_tcp_connections

    # print summary
    print("-" * 80)
    print(f"SUMMARY")
    print("-" * 80)
    print(f"DNS Time: {dns_time} ms")
    print(f"TCP Time: {tcp_time} ms")
    print(f"HTTP Time: {http_time} ms")
    print(f"TCP Connection Count: {total_tcp_connections}")
    print(f"Total Time: {total_time} ms")
    print("-" * 80 + "\n")

    # print http response
    print("-" * 80)
    print("HTTP RESPONSE")
    print("-" * 80)
    try:
        print(received_data.decode("utf-8"))
    except Exception as e:
        print(f"Error: main(): {e}")


if __name__ == "__main__":
    main()
