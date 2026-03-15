import sys
import socket
import struct
from scapy.all import ARP, Ether, srp, conf


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    local_ip = None
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"Local ip: {local_ip}")
        return local_ip

    except socket.error as se:
        print(f"Socket error: {se}")
    except Exception as e:
        print(f"Error when fetching local IP address: {e}")


def get_local_network(local_ip: str):
    local_network = None
    try:
        netmask = input(f"Input netmask: ")
        ip_int = struct.unpack("!I", socket.inet_aton(local_ip))[0]
        mask_int = struct.unpack("!I", socket.inet_aton(netmask))[0]
        cidr = bin(mask_int).count("1")

        local_network = (
            f"{socket.inet_ntoa(struct.pack("!I", ip_int & mask_int))}/{cidr}"
        )
        return local_network
    except Exception as e:
        print(f"Error when fetching local network address: {e}")


def scan_arp(local_network: str, timeout: int, verbose: bool = False):
    arp_request = ARP(pdst=local_network)
    broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = broadcast / arp_request

    print(
        f"Fetching ARP lists for local network: {local_network}, timeout: {timeout} secs"
    )

    answered, _ = srp(packet, timeout=timeout, verbose=verbose)

    results = []

    for sent, received in answered:
        results.append(
            {
                "ip": received.psrc,
                "mac": received.hwsrc,
            }
        )
    return results


def print_table(results: list):
    if not results:
        print("No host recorded. Make sure you input the correct netmask")
        exit(0)
    col_ip = max(len(r["ip"]) for r in results)
    col_mac = max(len(r["mac"]) for r in results)
    col_ip = max(col_ip, 15)
    col_mac = max(col_mac, 17)

    sep = f"+{'-' * (col_ip + 2)}+{'-' * (col_mac + 2)}+"
    fmt = f"| {{:<{col_ip}}} | {{:<{col_mac}}} |"

    print(sep)
    print(fmt.format("IP Address", "MAC Address"))
    print(sep)

    for r in sorted(results, key=lambda x: socket.inet_aton(x["ip"])):
        print(fmt.format(r["ip"], r["mac"]))
        print(sep)


def main():
    local_ip = get_local_ip()
    if local_ip is None:
        print("Fatal: cannot determine local ip address")
        exit(1)

    local_network = get_local_network(local_ip)
    if local_network is None:
        print("Fatal: cannot determine local network address")
        exit(1)

    print_table(scan_arp(local_network, 2))


if __name__ == "__main__":
    main()
