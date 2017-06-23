[![Coverage Status](https://coveralls.io/repos/github/edeno/Jadhav-2016-Data-Analysis/badge.svg?branch=master)](https://coveralls.io/github/edeno/Jadhav-2016-Data-Analysis?branch=master)

### Data Description ###

Data is from:
> Jadhav, S. P., Rothschild, G., Roumis, D. K. & Frank, L. M. Coordinated Excitation and Inhibition of Prefrontal Ensembles during Awake Hippocampal Sharp-Wave Ripple Events. Neuron 90, 113–127 (2016).

### Raw Data Format ###
Data is in the Matlab format (.mat files). See the [Loren Frank Data Format Description Repository](https://github.com/edeno/Loren-Frank-Data-Format--Description) for more information.

### Installation ###

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash:
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Go to the local repository (`.../Jadhav-2016-Data-Analysis`) and install the anaconda environment for the repository. Type into bash:
```bash
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda env create -f environment.yml
source activate Jadhav-2016-Data-Analysis
python setup.py develop
```

3. Finally, to verify that the code has been installed correctly, run the tests:
```bash
pytest
```
