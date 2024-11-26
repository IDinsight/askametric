#!make
SHELL := /bin/bash
.PHONY : tests

PROJECT_NAME = ask_a_metric
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

clean:
	find .. -type f -name "*.py[co]" -delete
	find .. -type d -name "__pycache__" -delete
	find .. -type d -name ".pytest_cache" -delete

fresh-env :
	conda remove --name $(PROJECT_NAME) --all -y
	conda create --name $(PROJECT_NAME) python==3.11 -y

	$(CONDA_ACTIVATE) $(PROJECT_NAME); \
	python -m pip install -r requirements-dev.txt --ignore-installed; \
	pre-commit install

# Run tests
tests: run-tests

run-tests:
	@docker compose -f ./deployment/docker-compose/docker-compose.unittest.yml run --build --rm core_backend

# Run validation
validate:
	@cd validation && python validate.py --path_to_data_sources "../databases" && cd ..
