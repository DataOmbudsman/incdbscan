test:
	python -m pytest -m "not slow" incdbscan/tests/*

test-slow:
	python -m pytest -m slow incdbscan/tests/*
