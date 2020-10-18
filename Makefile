SHELL := /bin/bash

ROOTDIR := $(shell pwd)
CODEDIR := code

SAMPLE := X77
REPLACE := no

# ------------------------------------------------------------------------------
# Clean raw data and perform sample selection
.PHONY: data
data:
	@cd $(CODEDIR); python3 -m src.data.make_data $(SAMPLE)


# -----------------------------------------------------------------------------
# create database for targets, features, outcomes, and predictions
.PHONY: db
# .DELETE_ON_ERROR: hello

db:
	@echo 'Creating database...'
	@cd $(CODEDIR); python3 -m src.db.create_db $(SAMPLE) $(REPLACE)
	@echo 'Adding decisions...'
	@cd $(CODEDIR); python3 -m src.db.add_decisions $(SAMPLE) $(REPLACE)
	@echo 'Adding features...'
	@cd $(CODEDIR); python3 -m src.db.add_features $(SAMPLE) $(REPLACE)
	@echo 'Database for sample $(SAMPLE) built.'


# ------------------------------------------------------------------------------
# Produce summary statistics table
.PHONY: sumstats
sumstats:
	@cd $(CODEDIR); python3 -m src.data.sumstats $(SAMPLE)


# -----------------------------------------------------------------------------
.PHONY: model
# .DELETE_ON_ERROR: hello

model:
	@python3 $(CODEDIR)/model.py $(SAMPLE)


# -----------------------------------------------------------------------------
.PHONY: analysis
# .DELETE_ON_ERROR: hello

analysis:
	@python3 $(CODEDIR)/analysis.py $(SAMPLE)
