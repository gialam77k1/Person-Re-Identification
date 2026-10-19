#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ruff check .
python src/validate_repository.py
python src/check_repository_hygiene.py
python -m unittest discover -v tests
python -c "from src.api.main import app; schema = app.openapi(); expected = {'/health', '/ready', '/model-info', '/embed', '/search'}; missing = expected - set(schema['paths']); assert not missing, f'Missing endpoints: {sorted(missing)}'; print('OpenAPI smoke: PASS')"

