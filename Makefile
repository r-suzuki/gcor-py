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

# Build HTML docs with pdoc
# - "docs" is a convenient output dir for GitHub Pages.
docs:
	uv run quarto render README_USER.qmd
	@VERSION=$$(uv run python -c "import tomllib; from pathlib import Path; d=tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8')); print(d['project']['version'])"); \
	rm -rf docs; \
	uv run pdoc --docformat numpy --math --footer-text "gcor v$${VERSION}" -o docs gcor

# Build source distribution and wheel (equivalent to R CMD build)
build: docs
	rm -rf dist/
	uv run python -m build

# Full check before release (rough equivalent of R CMD CHECK)
check: sync test build
	@echo "✔ All checks completed successfully."

# Remove build artifacts and temporary environments
clean:
	rm -rf dist build *.egg-info .venv-check docs