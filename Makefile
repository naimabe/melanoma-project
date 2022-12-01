env:
ifndef PYENV_VIRTUAL_ENV
	pyenv virtualenv melanoma-project
	pyenv local melanoma-project
endif
	pip install -e .
