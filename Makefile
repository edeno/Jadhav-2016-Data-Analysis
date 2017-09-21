
make install:
    pip install conda
	conda update -q conda
    conda env create -f environment.yml
    source activate Jadhav-2016-Data-Analysis
	python setup.py develop
