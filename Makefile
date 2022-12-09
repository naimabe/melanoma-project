env:
ifndef PYENV_VIRTUAL_ENV
	pyenv virtualenv melanoma-project
	pyenv local melanoma-project
endif
	pip install -e .


run:
	python main.py

setup: requirements.txt
	pip install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf venv


venv/bin/activate: requirements.txt
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

run: venv/bin/activate
	./venv/bin/python3 app.py


run_api:
	uvicorn src.api.fast:app --reload
