@load base/protocols/conn
@load base/protocols/ssl

@load ja3

redef Log::default_logdir = "/outputs/mix";

# Xuất JSON-line cho an toàn
redef LogAscii::use_json = T;

# Không dùng CSV nữa
# redef LogAscii::separator = ",";   # ← BỎ DÒNG NÀY

redef SSL::disable_analyzer_after_detection = F;