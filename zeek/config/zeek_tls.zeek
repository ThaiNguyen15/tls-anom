# =========================
# Zeek TLS Extraction Config (for Zeek 7.x)
# =========================

@load base/protocols/conn
@load base/protocols/ssl

# JA3 fingerprinting (zkg installed)
@load ja3

redef Log::default_logdir = "/outputs";
redef LogAscii::separator = ",";
redef SSL::disable_analyzer_after_detection = F;
