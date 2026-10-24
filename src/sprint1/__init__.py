# Clean-branch layout: shared code lives in common/sprint1; experiment-specific modules (sprint1.math.*,
# sprint1.moonshot.*, sprint1.scripts.ladder_analyze) live under experiments/<x>/code/sprint1/. extend_path merges
# every `sprint1/` directory on sys.path, so PYTHONPATH=common:experiments/<x>/code makes them importable.
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
