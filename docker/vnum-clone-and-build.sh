#!/usr/bin/env bash

# Syntax: ./vtl-clone-and-build.sh [DEV_IMG] [VTL_VERSION]

DEV_IMG=${1:-"false"}
VTL_VERSION=${2:-"latest"}
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

BRANCH="v${VTL_VERSION}"
if [ "${VTL_VERSION}" = "latest" ]; then
  BRANCH="main"
fi

git clone -b $BRANCH --single-branch --depth 1 https://github.com/vlang/vtl.git /opt/vlang/v/vlib/vtl
