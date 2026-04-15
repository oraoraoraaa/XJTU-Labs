import socket
import sys
import time


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


def http_request(sock, hostname, port, path="/"):
    """
    Send HTTP request to and receive from the target host.
    Args:
        sock: A socket object on which to be made HTTP request
        hostname: The domain (hostname) of the website to connect
        port: The port to connect
        path: The path to request
    Returns:
        tuple: (received_data, http_time)
    """
    print("\n[3]  HTTP Request")

    request = (
        f"GET {path} HTTP/1.1\r\n"
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

    hostname = "47.106.213.176"  # Fixed IP for the server
    port = 80  # Fixed port
    file_name = "download_capture.pcap"
    path = "/" + file_name  # Path to the file

    default_choice = (
        input(f"Download file from {hostname}:{port}{path}? (y/n): ").strip().lower()
    )
    if default_choice == "":
        default_choice = "y"

    if default_choice != "y":
        print("Operation cancelled by user.")
        sys.exit(0)

    print("Connecting...")

    start_time = time.time()

    # STEP 1: Resolve DNS (though it's an IP, this will work)
    ip_address, dns_time = resolve_dns(hostname)

    # STEP 2: Make TCP Connection
    sock, tcp_time = tcp_connection(ip_address, port)

    # STEP 3: HTTP Request
    received_data, http_time = http_request(sock, hostname, port, path)

    total_time = (time.time() - start_time) * 1000

    # print summary
    print(f"{"-" * 80}")
    print(f"SUMMARY")
    print(f"{"-" * 80}")
    print(f"DNS Time: {dns_time} ms")
    print(f"TCP Time: {tcp_time} ms")
    print(f"HTTP Time: {http_time} ms")
    print(f"Total Time: {total_time} ms")
    print(f"{"-" * 80}\n")

    # Parse and save the file
    print(f"{"-" * 80}")
    print("SAVING FILE")
    print(f"{"-" * 80}")
    try:
        # Find the end of headers
        header_end = received_data.find(b"\r\n\r\n")
        if header_end == -1:
            print("Invalid HTTP response: no header-body separator")
            return
        headers = received_data[:header_end].decode("utf-8", errors="ignore")
        body = received_data[header_end + 4 :]  # Skip \r\n\r\n

        if "200 OK" in headers:
            with open(f"{file_name}", "wb") as f:
                f.write(body)
            print(f"File downloaded successfully as {file_name}.")
            print(f"File size: {len(body)} bytes")
        else:
            print("Failed to download file. Response status not 200.")
            print("Response headers:")
            print(headers)
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
