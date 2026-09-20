"""Portable locations and read-only protection for completed scientific evidence."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
PATHS = json.loads((ROOT / 'config/resource_paths.json').read_text())

def resource(name):
    """Resolve a named saved artifact; no dependence on current working directory."""
    key = str(name).replace('\\', '/')
    if key not in PATHS:
        raise FileNotFoundError(f'No registered project resource: {key}')
    return ROOT / PATHS[key]

def report(name):
    return resource('published/' + str(name))

def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required local artifact missing: {path.relative_to(ROOT)}. See data/README.md and reports/methodology/reproducibility.md. No model will be fitted automatically.')
    return path

def protect_completed_evidence():
    """Refuse script writes outside artifacts/reproduction; never overwrite evidence."""
    allowed = (ROOT / 'artifacts/reproduction').resolve()
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            mode = args[1]
            flags = args[2]
            import os
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not Path(args[0]).resolve().is_relative_to(allowed):
                raise PermissionError('Completed evidence is read-only; no overwrite authorized.')
    sys.addaudithook(guard)
