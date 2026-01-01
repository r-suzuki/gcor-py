.PHONY: sync test debug build install-check readme check clean

# Install dependencies (including development extras)
sync:
	uv sync --extra dev

# Run tests quietly
test:
	uv run pytest -q

# Run tests with stdout/stderr (useful for debugging)
debug:
	uv run pytest -s

# Build source distribution and wheel (equivalent to R CMD build)
build:
	uv run python -m build

# Render README.qmd to README.md using Quarto
readme:
	quarto render README.qmd

# Full check before release (rough equivalent of R CMD CHECK)
check: sync test build readme
	@echo "✔ All checks completed successfully."

# Remove build artifacts and temporary environments
clean:
	rm -rf dist build *.egg-info .venv-check
