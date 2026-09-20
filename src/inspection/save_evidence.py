"""Check the archived notebook structure without executing its inspection cells."""
import json
from src.paths import ROOT
def main():
    book=json.loads((ROOT/'notebooks/01_dataset_inspection.ipynb').read_text())
    assert book['nbformat']==4
    assert all(not c.get('outputs') for c in book['cells'] if c['cell_type']=='code')
    print('Archived notebook structure verified; no cells executed.')
if __name__=='__main__':
    main()
