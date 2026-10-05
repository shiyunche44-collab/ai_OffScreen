.PHONY: sync fmt lint type test contracts schemas openapi api-types api-types-check web-install web-check check
BE = cd backend &&

sync:
	$(BE) uv sync --group dev

fmt:
	$(BE) uv run ruff format src tests && uv run ruff check --fix src tests

lint:
	$(BE) uv run ruff format --check src tests && uv run ruff check src tests

type:
	$(BE) uv run mypy

test:
	$(BE) uv run pytest -q

contracts:
	$(BE) uv run lint-imports

schemas:
	$(BE) uv run python ../scripts/export_schemas.py

openapi:
	$(BE) uv run python ../scripts/export_openapi.py

api-types: openapi
	cd frontend && npm run api-types

# The committed frontend types must be what the OpenAPI document generates.
api-types-check:
	cd frontend && npx openapi-typescript ../docs/openapi.json -o .schema.check.d.ts >/dev/null \
	  && diff -q .schema.check.d.ts src/api/schema.d.ts >/dev/null \
	  || { rm -f frontend/.schema.check.d.ts .schema.check.d.ts; echo "frontend/src/api/schema.d.ts is stale: run make api-types"; exit 1; }
	rm -f frontend/.schema.check.d.ts

web-install:
	cd frontend && npm ci

web-check: api-types-check
	@test -d frontend/node_modules || { echo "run make web-install first"; exit 1; }
	cd frontend && npm run typecheck && npm test && npm run build

check: lint type test contracts web-check
