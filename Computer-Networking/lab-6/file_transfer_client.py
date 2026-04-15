import socket


def request_file(file_name, host="47.106.213.176", port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        client_socket.sendall(file_name.encode())
        first_chunk = client_socket.recv(1024)
        if first_chunk == b"File not found":
            print("File not found on server.")
            return

        output_path = f"received_{file_name}"
        with open(output_path, "wb") as f:
            f.write(first_chunk)
            while data := client_socket.recv(1024):
                f.write(data)

        print(f"File {file_name} received successfully.")


if __name__ == "__main__":
    request_file("file2.txt")
