cd zeek
tcpdump -r /raw/botnet/2018-04-04_win20.pcap -w /tmp/botnet/2018-04-04_win20_fixed.pcap

./run-zeek.sh
./script/entrypoint.sh

source .venv/bin/activate
pip install -e .

input → processed
tls-anom run \
  --dataset data/processed/zeek3/mix \
  --name mix \
  --stages extract \
  --config config/default.yaml


Step: label
tls-anom run --dataset data/raw/mix.csv --name mix --stages label

tls-anom run \
  --dataset data/processed/zeek3/normal \
  --name normal \
  --stages label,featurize \
  --config config/default.yaml

tls-anom run \
  --dataset data/processed/zeek3/normal \
  --name normal \
  --stages preprocess \
  --config config/default.yaml

tls-anom run \
  --dataset data/features \
  --name botnet \
  --stages train \
  --config config/default.yaml

tls-anom run \
  --dataset data/features \
  --name botnet \
  --stages predict


| Dataset | Vai trò                  |
| ------- | ------------------------ |
| normal  | **TRAIN model**          |
| botnet  | **PREDICT / EVALUATE**   |
| mix     | **PREDICT (real-world)** |

normal
tls-anom run \
  --dataset data/processed/zeek3/normal \
  --name normal \
  --stages featurize,preprocess  \
  --config config/default.yaml