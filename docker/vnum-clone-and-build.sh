#!/usr/bin/env bash

# Syntax: ./vnum-clone-and-build.sh [DEV_IMG] [VNUM_VERSION]

DEV_IMG=${1:-"false"}
VNUM_VERSION=${2:-"latest"}
VSL_VERSION=${3:-"latest"}

# Install vsl
BRANCH="v${VSL_VERSION}"
if [ "${VSL_VERSION}" = "latest" ]; then
  BRANCH="master"
fi

git clone -b $BRANCH --single-branch --depth 1 https://github.com/vlang/vsl.git /opt/vlang/v/vlib/vsl

if [ "${DEV_IMG}" = "true" ]; then
  exit 0
fi

BRANCH="v${VNUM_VERSION}"
if [ "${VNUM_VERSION}" = "latest" ]; then
  BRANCH="main"
fi

git clone -b $BRANCH --single-branch --depth 1 https://github.com/vlang/vnum.git /opt/vlang/v/vlib/vnum
