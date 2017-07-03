#!/bin/bash

shopt -s nullglob
FILES=($@)

for notebook in "${FILES[@]}"
do
  echo "Processing $notebook file..."
  jupyter nbconvert --to notebook $notebook --output $notebook
done
