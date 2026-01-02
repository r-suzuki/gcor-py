.PHONY: sync test testv build install-check docs check clean

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

# Build HTML docs with pdoc
# - Adjust "gcor" to your top-level package/module name.
# - "docs" is a convenient output dir for GitHub Pages.
docs:
	uv run quarto render README_USER.qmd
	@VERSION=$$(uv run python -c "import tomllib; from pathlib import Path; d=tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8')); print(d['project']['version'])"); \
	rm -rf docs; \
	uv run pdoc --docformat numpy --footer-text "gcor v$${VERSION}" -o docs/gcor-py gcor

# Full check before release (rough equivalent of R CMD CHECK)
check: sync test build docs
	@echo "✔ All checks completed successfully."

# Remove build artifacts and temporary environments
clean:
	rm -rf dist build *.egg-info .venv-check docs