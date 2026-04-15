import socket


def get_private_server_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def start_server(host="0.0.0.0", port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server listening on {host}:{port}", flush=True)
        conn, addr = server_socket.accept()
        with conn:
            client_ip, client_port = addr
            server_private_ip = get_private_server_ip()
            print(f"Client public IP: {client_ip}", flush=True)
            print(f"Server private IP: {server_private_ip}", flush=True)
            print(f"Connected by {addr}", flush=True)

            data = conn.recv(1024)
            if not data:
                print(
                    "No filename received from client, closing connection.", flush=True
                )
                return

            file_name = data.decode().strip()
            if not file_name:
                print("Received empty filename, closing connection.", flush=True)
                return

            print(f"Requested file: {file_name}", flush=True)
            try:
                with open(file_name, "rb") as f:
                    print(f"Starting transfer of {file_name}.", flush=True)
                    while chunk := f.read(1024):
                        try:
                            conn.sendall(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            print(
                                "Client disconnected before transfer finished.",
                                flush=True,
                            )
                            return
                    try:
                        conn.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    print("File transfer complete. Connection closed.", flush=True)
            except FileNotFoundError:
                print(f"File not found: {file_name}", flush=True)
                try:
                    conn.sendall(b"File not found")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return


if __name__ == "__main__":
    start_server()
