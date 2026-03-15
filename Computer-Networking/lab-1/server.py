import socket
import threading
from datetime import datetime


def handle_client(conn, addr):
    """Handle single client connection"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 新连接来自：{addr}")
    print(f"客户端 IP：{addr[0]}，端口：{addr[1]}")

    try:
        while True:
            data = conn.recv(2048)
            if not data:
                print(f"客户端 {addr} 已断开连接。")
                break

            message = data.decode("utf-8")
            print(f"from {addr}: {message}")

            conn.send(data)
    except Exception as e:
        print(f"处理客户端 {addr} 时出错：{e}")
    finally:
        conn.close()
        print(f"与 {addr} 的连接已关闭。")


def main():
    host = "0.0.0.0"  # listen all the devices
    port = 3939

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((host, port))
        server.listen(5)
        print(f"TCP 服务器已启动，监听{host}:{port} ...")
        print("等待客户端连接...\n")

        while True:
            conn, addr = server.accept()
            client_thread = threading.Thread(
                target=handle_client, args=(conn, addr), daemon=True
            )
            client_thread.start()
            print(f"当前活跃连接数：{threading.active_count() - 1}")

    except KeyboardInterrupt:
        print("服务器关闭中……")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        server.close()
        print("服务器已关闭")


if __name__ == "__main__":
    main()
