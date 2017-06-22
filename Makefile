
make install:
    pip install conda
	conda config --set always_yes yes --set changeps1 no
	conda update -q conda
    conda env create -f environment.yml
    source activate Jadhav-2016-Data-Analysis
	python setup.py develop
