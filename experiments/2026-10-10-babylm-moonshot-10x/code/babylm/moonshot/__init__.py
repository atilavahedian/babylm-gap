# Namespace bridge (clean-repo addition): lets babylm.* modules under experiments/*/code merge with common/babylm
# when both are on PYTHONPATH. The original file was empty.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
