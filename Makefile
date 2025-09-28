.PHONY: all install run clean

all: install run

install:
	python3 -m venv venv || true
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && ./run.sh

clean:
	rm -rf venv __pycache__ outputs/*
