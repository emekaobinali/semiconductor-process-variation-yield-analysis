"""Community Cloud entrypoint; run from the repository root.

    python -m streamlit run deployment/streamlit_app.py

Select Python 3.12 to match the locally verified app environment.
"""
from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    # Ensure the approved root app and its package resolve before this directory.
    sys.path.insert(0, str(root))
    try:
        runpy.run_path(str(root / 'streamlit_app.py'), run_name='__main__')
    finally:
        sys.path.pop(0)


if __name__ == '__main__':
    main()
