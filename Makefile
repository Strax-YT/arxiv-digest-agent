.PHONY: install install-min test doctor graph clean

install:
	python -m pip install -r requirements.txt && python -m pip install -e .

install-min:
	python -m pip install -r requirements-min.txt && python -m pip install -e .

test:
	python -m pytest -q

doctor:
	python -m arxiv_agent doctor

graph:
	python -m arxiv_agent graph

clean:
	rm -rf .pytest_cache **/__pycache__ build dist *.egg-info
