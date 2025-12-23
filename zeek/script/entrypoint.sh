#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar nullglob

RAW_DIR="/raw/normal"
OUT_DIR="/output/normal"
CONFIG="/config/zeek_tls_json.zeek"

echo "[+] Zeek version:"
zeek --version || true
echo "[+] zkg packages loaded:"
zkg list || true

echo "[+] Scanning PCAPs under $RAW_DIR ..."
found_any=false

for pcap in $RAW_DIR/**/Monday-cutp1.{pcap,pcapng}; do
  [ -e "$pcap" ] || continue
  found_any=true

  dataset="$(basename "$(dirname "$pcap")")"
  ds_out="$OUT_DIR/$dataset"
  mkdir -p "$ds_out"

  echo "[+] Processing dataset=$dataset file=$(basename "$pcap")"
  ZEEK_LOG_DIR="$ds_out" zeek -r "$pcap" "$CONFIG"
done

if [ "$found_any" = false ]; then
  echo "[!] No .pcap/.pcapng found under $RAW_DIR/*/"
  exit 0
fi

echo "[✓] Done."