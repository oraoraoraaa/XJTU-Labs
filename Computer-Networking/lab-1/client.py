import socket


def main():
    target_ip = "127.0.0.1"
    target_port = 3939
    line_count = 0

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((target_ip, target_port))
        print(f"已连接到服务器 {target_ip}:{target_port}")
        print(f"第一行输入姓名、学号和班级，至少输入5行内容。\n")
        while True:
            message = input(f"第{line_count + 1}行> ")
            if message.lower() == "exit":
                print("用户中断连接")
                break
            client.send(message.encode("utf-8"))
            line_count += 1
            message = (client.recv(2048)).decode("utf-8")

            print(f"服务器回复：{message}")

            if line_count >= 5:
                if input(f"已输入5行，是否继续？(y/n): ").lower() != "y":
                    break

    except ConnectionRefusedError:
        print(f"拒绝连接 检查服务器{target_ip}:{target_port}")
    except Exception as e:
        print(f"发生错误：{e}")
    finally:
        client.close()
        print("会话结束")


if __name__ == "__main__":
    main()
