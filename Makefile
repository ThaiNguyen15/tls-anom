.PHONY: venv install lint test run

venv:
	python -m venv .venv
	./.venv/Scripts/pip install -U pip || ./.venv/bin/pip install -U pip

install:
	pip install -e .
	pip install -r requirements-dev.txt || true

lint:
	pre-commit run --all-files || true

test:
	pytest -q

run:
	tls-anom run --dataset data/raw/normal.csv --name normal --config config/default.yaml
