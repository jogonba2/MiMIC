#!/usr/bin/env bash

set -e
set -x

mypy "mimic"
flake8 "mimic" --ignore=E501,W503,E203,E402,E704
black "mimic" --check -l 80
