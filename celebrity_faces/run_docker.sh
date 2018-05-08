#!/bin/sh -e

CUR_DIR="$(dirname $(readlink -f "${0}"))"
C_NAME='celebrity_faces'

docker build "--tag=${C_NAME}" "${CUR_DIR}/"
#docker run --rm --publish 8080:8080 ${C_NAME} -v /home/sgjurano/work/shad/sem4/lsml/img_align_celeba:/app/img_align_celeba:ro
