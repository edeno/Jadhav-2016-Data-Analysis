#!/bin/bash

shopt -s nullglob
FILES=($@)

for notebook in "${FILES[@]}"
do
  echo "Processing $notebook file..."
  jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 $notebook 
done
