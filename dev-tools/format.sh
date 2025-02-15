#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "mimic" --exclude=__init__.py
isort "mimic"
black "mimic" -l 80