#/bin/bash
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs); do
    if ps -p "$pid" -o comm= | grep -qi python; then
        sudo kill -9 "$pid"
    fi
done