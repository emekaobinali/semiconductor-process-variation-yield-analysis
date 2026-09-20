"""Portability checks use synthetic data or saved metadata, never model fitting."""
import ast
import importlib
import json
import subprocess
import sys
import unittest
from src.paths import ROOT, resource
from src.integrity import verify_preprocessing_provenance

class PortabilityTests(unittest.TestCase):
    def test_all_source_imports_without_analysis(self):
        # A fresh subprocess rejects any scientific-input access on import.
        code = '''import sys,pathlib,importlib
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  p=str(args[0]).replace(chr(92),'/')
  if '/data/raw/' in p or '/data/splits/' in p or p.endswith(('.joblib','.npz')):
   raise AssertionError('Scientific input accessed during import')
sys.addaudithook(guard)
for p in pathlib.Path('src').rglob('*.py'):
 if p.name!='__init__.py': importlib.import_module('.'.join(p.with_suffix('').parts))
'''
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_paths_are_contained_and_cwd_independent(self):
        paths=json.loads((ROOT/'config/resource_paths.json').read_text())
        for key in paths:
            self.assertTrue(resource(key).resolve().is_relative_to(ROOT))
        self.assertEqual(resource('inspection/raw'), ROOT/'data/raw')
        self.assertEqual(resource('splits/development_rows.csv'),ROOT/'data/splits/development_rows.csv')

    def test_frozen_evaluation_refuses_replay(self):
        result=subprocess.run([sys.executable,'-B','-m','src.evaluation.final_test_once'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Replay is disabled',result.stderr)

    def test_notebook_is_clean_and_code_parses(self):
        book=json.loads((ROOT/'notebooks/01_dataset_inspection.ipynb').read_text())
        for cell in book['cells']:
            self.assertEqual(cell['metadata'],{})
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
                ast.parse(''.join(cell['source']))

    def test_transformer_science_unchanged(self):
        freeze=json.loads((ROOT/'config/frozen_method.json').read_text())
        verify_preprocessing_provenance(freeze['preprocessing_source_sha256'])

if __name__=='__main__': unittest.main()
