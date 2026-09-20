"""Validate original source provenance without changing frozen historical hashes."""
import ast,hashlib,json
from src.paths import ROOT
def verify_preprocessing_provenance(expected_original):
    record=json.loads((ROOT/'config/preprocessing_provenance.json').read_text())
    tree=ast.parse((ROOT/'src/preprocessing.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    assert record['original_preprocessing_sha256']==expected_original
    assert hashlib.sha256(ast.dump(cls,include_attributes=False).encode()).hexdigest()==record['transformer_ast_sha256']
