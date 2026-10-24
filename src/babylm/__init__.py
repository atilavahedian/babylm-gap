# Shared BabyLM package. extend_path lets an experiment folder add sub-packages (babylm/a4, babylm/moonshot,
# babylm/winner, ...) or override modules by putting its own babylm/ directory on PYTHONPATH; see README.md.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
