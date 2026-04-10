set dotenv-load := false

# Default recipe: show help
help:
    @echo "=========================================="
    @echo "Business Data Science"
    @echo "------------------------------------------"
    @just --list
    @echo "=========================================="

###
### Repo
###

# Show repo structure
tree:
    @tree -a -I '.git|.venv|__pycache__|*.pyc|.ruff_cache|__marimo__|.ipynb_checkpoints|node_modules' -L 3

###
### Course scaffolding
###

# Create a new course project from template
new name:
    @if [ -d "courses/{{name}}" ]; then echo "Error: courses/{{name}} already exists"; exit 1; fi
    @cp -r templates/course-template "courses/{{name}}"
    @sed -i '' 's/course-template/{{name}}/g' "courses/{{name}}/pyproject.toml"
    @sed -i '' 's/Course Title/{{name}}/g' "courses/{{name}}/README.md"
    @echo "Created courses/{{name}}/"
    @echo "Next steps:"
    @echo "  cd courses/{{name}}"
    @echo "  uv python pin <version>"
    @echo "  uv add <dependencies>"
    @echo "  uv lock"

###
### Notebooks
###

# Convert a Jupyter notebook to marimo format
convert path:
    @echo "Converting {{path}} to marimo..."
    @uv run marimo convert "{{path}}"

# Open a marimo notebook for editing
edit path:
    @uv run marimo edit "{{path}}"

# Export a marimo notebook to HTML
export-html path:
    @uv run marimo export html "{{path}}" -o "{{without_extension(path)}}.html"
    @echo "Exported to {{without_extension(path)}}.html"

###
### Quality (run from a course directory)
###

# Lint a course project
lint dir:
    @echo "Linting {{dir}}..."
    @cd "{{dir}}" && uv run ruff check .

# Format a course project
format dir:
    @echo "Formatting {{dir}}..."
    @cd "{{dir}}" && uv run ruff format .
