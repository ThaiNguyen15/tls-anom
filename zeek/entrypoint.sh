#!/usr/bin/env bash
set -euo pipefail
shopt -s globstar nullglob

RAW_DIR="/raw/botnet"          # Thay thành /raw/normal khi cần
OUT_DIR="/outputs/botnet"       # Thay thành /output/normal khi cần
CONFIG="/config/zeek_tls_json.zeek"
ALL_DIR="$OUT_DIR/all"

ALL_CONN="$ALL_DIR/conn.log"
ALL_SSL="$ALL_DIR/ssl.log"
ALL_X509="$ALL_DIR/x509.log"

mkdir -p "$OUT_DIR"
mkdir -p "$ALL_DIR"
echo "[+] Zeek version:"
zeek --version || true

echo "[+] Processing all PCAP files in $RAW_DIR ..."

found_any=false

for pcap in "$RAW_DIR"/**/*.{pcap,pcapng}; do
  [ -e "$pcap" ] || continue
  found_any=true

  # Lấy tên file PCAP KHÔNG có đuôi .pcap hoặc .pcapng
  filename="$(basename "$pcap")"
  dataset="${filename%.pcap}"     # xóa .pcap nếu có
  dataset="${dataset%.pcapng}"    # xóa .pcapng nếu còn

  ds_out="$OUT_DIR/$dataset"
  mkdir -p "$ds_out"

  echo
  echo "[+] Processing PCAP: $filename"
  echo "    → Output folder: $ds_out"

  # Chạy Zeek, ghi log thẳng vào thư mục riêng theo tên file
  zeek -C -r "$pcap" "$CONFIG" Log::default_logdir="$ds_out"

  echo "    [*] Logs generated:"
  ls -lh "$ds_out"/*.log 2>/dev/null || echo "    (no logs generated for this PCAP)"
done

if [ "$found_any" = false ]; then
  echo "[!] No .pcap or .pcapng files found in $RAW_DIR"
  exit 0
fi

echo
echo "[✓] DONE!"
echo "[+] Merging all logs into 3 big files..."

# Xóa file cũ nếu tồn tại (để chạy lại nhiều lần không bị append thừa)
rm -f "$ALL_CONN" "$ALL_SSL" "$ALL_X509"


for folder in "$OUT_DIR"/*/; do
  [ -d "$folder" ] || continue

  # Bỏ qua thư mục all để tránh tự gộp lại
  if [ "$(basename "$folder")" = "all" ]; then
    continue
  fi

  echo "    Including logs from: $(basename "$folder")"

  # Gộp conn.json (tên file có thể là conn.log nhưng nội dung JSON)
  if [ -f "$folder/conn.log" ]; then
    cat "$folder/conn.log" >> "$ALL_CONN"
  fi

  if [ -f "$folder/ssl.log" ]; then
    cat "$folder/ssl.log" >> "$ALL_SSL"
  fi

  if [ -f "$folder/x509.log" ]; then
    cat "$folder/x509.log" >> "$ALL_X509"
  fi
done

# Báo kết quả
echo
echo "[✓] Merging completed!"