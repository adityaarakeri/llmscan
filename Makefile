PYTHON ?= 3.12

.PHONY: all lint format typecheck test ci

all: ci

ci: lint typecheck test
	@echo "================================="
	@echo "running target: $@"
	@echo "================================="

lint:
	@echo "================================="
	@echo "running target: $@"
	@echo "================================="
	uv run --python $(PYTHON) --with ruff ruff check .
	uv run --python $(PYTHON) --with ruff ruff format --check .

format:
	@echo "================================="
	@echo "running target: $@"
	@echo "================================="
	uv run --python $(PYTHON) --with ruff ruff format .

typecheck:
	@echo "================================="
	@echo "running target: $@"
	@echo "================================="
	uv run --python $(PYTHON) --with mypy mypy llmscan/

test:
	@echo "================================="
	@echo "running target: $@"
	@echo "================================="
	uv run --python $(PYTHON) --extra test pytest --tb=short -q
