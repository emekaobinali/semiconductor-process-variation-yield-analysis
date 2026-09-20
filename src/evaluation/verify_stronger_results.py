from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import csv,json,collections

def main():
    protect_completed_evidence()
    p=resource('stronger_comparison')
    a=json.loads((p/'aggregate.json').read_text())
    d=json.loads((p/'diagnostics.json').read_text())
    pred=list(csv.DictReader((require_file(resource('stronger_comparison/development_oof_predictions.csv'))).open()))
    old=list(csv.DictReader(resource('baseline_comparison/fold_metrics.csv').open()))
    new=list(csv.DictReader(report('secom-stronger-model-fold-metrics.csv').open()))
    assert len(d)==30 and not any(r['warnings'] for r in d)
    print('30 fits; no warnings.')
    for r in a:
        s=[z for z in pred if z['model']==r['model'] and z['indicators']==str(r['indicators'])]
        c=collections.Counter((int(z['failure']),int(z['prediction'])) for z in s)
        assert [c[(0,0)],c[(0,1)],c[(1,0)],c[(1,1)]]==[r[k] for k in ['TN','FP','FN','TP']]
        delta=[float(z['AP'])-float(next(b['AP'] for b in old if b['model']=='tree' and b['indicators']==str(r['indicators']) and b['fold']==z['fold'])) for z in new if z['model']==r['model'] and z['indicators']==str(r['indicators'])]
        print(r['model'],r['indicators'],'verified counts; AP improvements vs tree:',sum(v>0 for v in delta),'of 5 folds')


if __name__ == '__main__':
    main()
