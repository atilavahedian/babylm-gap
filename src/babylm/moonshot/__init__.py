# Clean-tree note: only kd.py (imported by babylm.harness.train) lives in common/; the rest of babylm.moonshot lives in
# the moonshot experiment folder's code/babylm/moonshot/ and is found through extend_path when it is on PYTHONPATH.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
