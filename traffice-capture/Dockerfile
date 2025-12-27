FROM ubuntu:22.04

RUN apt update && apt install -y tcpdump iproute2 && rm -rf /var/lib/apt/lists/*

WORKDIR /pcap
CMD ["tcpdump", "-i", "any", "-w", "/pcap/egress.pcap"]