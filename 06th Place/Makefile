.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = drivendata-cleaning-time-sol
PYTHON_INTERPRETER = python3

RAW_DATA = $(wildcard data/raw/*.*)

ifneq (,$(wildcard /opt/local/bin/virtualenvwrapper.sh-3.6))
	VIRTUALENV_WRAPPER = /opt/local/bin/virtualenvwrapper.sh-3.6
else
	VIRTUALENV_WRAPPER = /usr/local/bin/virtualenvwrapper.sh
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

create_dirs:
	mkdir -p data/raw data/interim data/processed data/external
	mkdir -p models/logs models/submissions

## Make Dataset

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src


## Set up python interpreter environment
create_environment:
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource $(VIRTUALENV_WRAPPER)\n"
	@bash -c "source $(VIRTUALENV_WRAPPER);mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

data: requirements processed_data

gen_submission: models/final-submission.csv

interim_data: data/interim/train_test.hdf5

processed_data: data/processed/train_test.hdf5

data/interim/train_test.hdf5: $(RAW_DATA) src/data/make_interim_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_interim_dataset.py data/raw data/interim/train_test.hdf5

data/processed/train_test.hdf5: data/interim/train_test.hdf5 src/data/make_dataset.py src/features/build_features.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py \
		data/interim/train_test.hdf5 \
		data/processed/train_test.hdf5

models/final-submission.csv: models/submission-part1.csv models/submission-part2.csv
	./src/scripts/blend.py -i models/submission-part1.csv -i models/submission-part2.csv \
		-o models/final-submission.csv

models/submission-part1.csv: processed_data
	./e.sh --min-validate-date=None --k-fold --lib=keras --scaling --validation-plot \
		--model ContestBlendedModelV1 --validation-plot --gen-submission --epochs=200 \
		--random-state=0 --submission-file=models/submission-part1.csv

models/submission-part2.csv: processed_data
	./e.sh --min-validate-date=None --k-fold --lib=keras --scaling --validation-plot \
		--model ContestBlendedModelV1 --validation-plot --gen-submission --epochs=200 \
		--random-state=1 --submission-file=models/submission-part2.csv


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
