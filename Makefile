.PHONY: test validate validate-gpu audit replay package clean

DAGGERTRACE_OUTPUT_DIR ?= artifacts/final
DAGGERTRACE_TARGET ?= liger-rmsnorm
DAGGERTRACE_TARGET_ROOT ?=

test:
	PYTHONPATH=src python -m pytest -q

validate:
	./scripts/validate.sh

validate-gpu:
	DAGGERTRACE_TARGET="$(DAGGERTRACE_TARGET)" \
	DAGGERTRACE_OUTPUT_DIR="$(DAGGERTRACE_OUTPUT_DIR)" \
	DAGGERTRACE_TARGET_ROOT="$(DAGGERTRACE_TARGET_ROOT)" \
	./scripts/validate.sh --gpu

audit:
	PYTHONPATH=src python -m daggertrace audit "$(DAGGERTRACE_TARGET)" --gpu --output "$(DAGGERTRACE_OUTPUT_DIR)" $(if $(DAGGERTRACE_TARGET_ROOT),--target-root "$(DAGGERTRACE_TARGET_ROOT)")

replay:
	PYTHONPATH=src python -m daggertrace replay "$(DAGGERTRACE_OUTPUT_DIR)"

package:
	python scripts/build_dist.py

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .coverage htmlcov
	find src tests evidence scripts -type d -name __pycache__ -prune -exec rm -rf {} +
