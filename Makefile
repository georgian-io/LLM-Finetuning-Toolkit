test-coverage:
	pytest --cov=llmtune tests/

fix-format:
	ruff check --fix
	ruff format

build-release:
	rm -rf dist
	rm -rf build
	poetry build
	poetry publish
