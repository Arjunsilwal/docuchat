.PHONY: install ingest app eval test lint

install:
	pip install -r requirements-dev.txt

ingest:
	python -m ragbot.ingest

app:
	streamlit run app.py

eval:
	python -m ragbot.evaluate --k 5

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .