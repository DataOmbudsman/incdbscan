isort:
	isort .

isort-check:
	isort . -c -v

lint:
	pylint \
	--disable=missing-class-docstring \
	--disable=missing-function-docstring \
	--disable=missing-module-docstring \
	--disable=too-few-public-methods \
	--fail-under 5 \
	--fail-on E,F \
	incdbscan/*

test:
	python -m pytest -m "not slow" incdbscan/tests/*

test-slow:
	python -m pytest -m slow incdbscan/tests/*

profile:
	mkdir -p profiling
	python profiling.py $(tag)