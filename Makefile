.PHONY: test-core test-awareness test-all smoke-core

test-core:
	pytest -q tests

test-awareness:
	cd apps/awareness_studio && pytest -q tests

test-all: test-core test-awareness

smoke-core:
	python main.py --mode synthetic
