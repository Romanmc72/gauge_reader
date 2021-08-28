#!/usr/bin/env bash

set -euo pipefail

main() {
    docker run -p 8086:8086 \
        -v influxdb:/var/lib/influxdb \
        -v influxdb2:/var/lib/influxdb2 \
        -e DOCKER_INFLUXDB_INIT_MODE=upgrade \
        -e DOCKER_INFLUXDB_INIT_USERNAME=romanmc72 \
        -e DOCKER_INFLUXDB_INIT_PASSWORD=!Q2w#E4r \
        -e DOCKER_INFLUXDB_INIT_ORG=grill \
        -e DOCKER_INFLUXDB_INIT_BUCKET=grill-bucket \
        influxdb:2.0.6-alpine
}

main
