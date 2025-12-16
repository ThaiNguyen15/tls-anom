
#!/bin/bash
docker run --rm -it   -v "$(pwd)/../data/raw:/raw"   -v "$(pwd)/script:/script"   -v "$(pwd)/config:/config"   -v "$(pwd)/../data/processed/zeek3:/outputs"   tls-anom-zeek:1.3
