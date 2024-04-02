build-release:
	rm -rf dist
	rm -rf build
	poetry build
	twine upload --repository pypi dist/*
