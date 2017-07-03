#!/bin/bash
# Examples
# --------
# bash run_notebooks.sh '../notebooks/2017_07_03_Group_Delay_Analysis.ipynb' '../notebooks/2017_07_03_Canonical_Coherence.ipynb'
# bash run_notebooks.sh "../notebooks/2017_07_03_*.ipynb"
shopt -s nullglob
FILES=($@)

for notebook in "${FILES[@]}"
do
  echo "Processing $notebook file..."
  jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 $notebook 
done
