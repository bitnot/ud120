#!/usr/bin/env bash

echo "running conda env export"
conda env export -n ud120 -f ../ud120.yml --no-builds

