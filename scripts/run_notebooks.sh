#!/bin/bash

shopt -s nullglob
FILES="../notebooks/2017_07_03_*.ipynb"

for notebook in $FILES
do
  echo "Processing $notebook file..."
  jupyter nbconvert --to notebook $notebook --output $notebook
done
