#!/bin/sh -e

CUR_DIR="$(dirname $(readlink -f "${0}"))"
C_NAME='hnsw_index'

docker build "--tag=${C_NAME}" "${CUR_DIR}/"
#docker run --rm --publish 6000:5000 ${C_NAME}
