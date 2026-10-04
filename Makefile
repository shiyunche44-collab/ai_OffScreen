.PHONY: sync fmt lint type test contracts check
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

check: lint type test contracts
