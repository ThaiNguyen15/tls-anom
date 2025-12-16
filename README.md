# TLS Anomaly – Quickstart

```bash
# 1) Install (editable)
pip install -e .

# 2) Run (dev)
tls-anom run --dataset data/raw/normal.csv --name normal --config config/default.yaml

# 3) Choose stages
TLS_STAGES=extract,label,featurize,preprocess,train,predict,evaluate \
  tls-anom run --dataset data/raw/mix.csv --name mix --stages $TLS_STAGES

# 4) Docker
docker build -t tls-anom:dev .
docker run --rm -v $PWD/data:/app/data tls-anom:dev tls-anom run \
  --dataset data/raw/botnet.csv --name botnet --config config/default.yaml
```
