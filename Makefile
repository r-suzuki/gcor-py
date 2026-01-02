.PHONY: sync test testv build install-check readme-user docs check clean

# Install dependencies (including development extras)
sync:
	uv sync --extra dev

# Run tests quietly
test:
	uv run pytest -q

# Run tests with stdout/stderr
testv:
	uv run pytest -s

# Build source distribution and wheel (equivalent to R CMD build)
build:
	uv run python -m build

# Render README_USER.qmd to README_USER.md using Quarto
readme-user:
	uv run quarto render README_USER.qmd

# Build HTML docs with pdoc
# - Adjust "gcor" to your top-level package/module name.
# - "docs" is a convenient output dir for GitHub Pages.
docs: readme-user
	rm -rf docs
	uv run pdoc --docformat numpy -o docs gcor

# Full check before release (rough equivalent of R CMD CHECK)
check: sync test build readme-user docs
	@echo "✔ All checks completed successfully."

# Remove build artifacts and temporary environments
clean:
	rm -rf dist build *.egg-info .venv-check docs