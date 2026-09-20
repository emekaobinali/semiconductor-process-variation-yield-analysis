from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import csv,collections,json,hashlib

def main():
    protect_completed_evidence()
    """Fixed intermediate thresholds; stdlib only; saved development scores only."""
    p = ROOT
    out=ROOT / 'artifacts/reproduction'
    out.mkdir(parents=True, exist_ok=True)
    sources=[resource('stronger_comparison/development_oof_predictions.csv'),resource('baseline_comparison/development_oof_predictions.csv')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    with sources[0].open(newline='') as f:
        stronger=[r for r in csv.DictReader(f) if r['model'] in ('small_forest','shallow_boosting')]
    with sources[1].open(newline='') as f:
        baseline=[r for r in csv.DictReader(f) if r['model']=='tree' and r['indicators']=='False']
    def metrics(rows,t):
        c=collections.Counter((int(r['failure']),int(float(r['score'])>=t)) for r in rows)
        tn,fp,fn,tp=[c[k] for k in [(0,0),(0,1),(1,0),(1,1)]]
        recall=tp/(tp+fn);spec=tn/(tn+fp)
        return dict(threshold=t,recall=recall,precision=tp/(tp+fp) if tp+fp else None,FP=fp,FN=fn,specificity=spec,balanced_accuracy=(recall+spec)/2,TP=tp,TN=tn,flagged=tp+fp)
    base=metrics(baseline,0.5)
    assert [base[k] for k in ('TN','FP','FN','TP')]==[811,359,35,48]
    membership={r['source_row_1based']:(r['failure'],r['fold']) for r in baseline}
    results=[];folds=[]
    for model in ('small_forest','shallow_boosting'):
        for flags in ('False','True'):
            rows=[r for r in stronger if r['model']==model and r['indicators']==flags]
            assert len(rows)==len(membership)==1253
            assert {r['source_row_1based']:(r['failure'],r['fold']) for r in rows}==membership
            assert all(0<=float(r['score'])<=1 for r in rows)
            group=[]
            for t in (0.25,0.30,0.35,0.40,0.45):
                r=dict(model=model,indicators=flags,**metrics(rows,t))
                r.update(detections_vs_tree=r['TP']-base['TP'],false_alarms_vs_tree=r['FP']-base['FP'])
                group.append(r)
                local=[dict(model=model,indicators=flags,fold=i,**metrics([z for z in rows if int(z['fold'])==i],t)) for i in range(1,6)]
                for k in ('TN','FP','FN','TP'):assert sum(z[k] for z in local)==r[k]
                folds.extend(local)
            assert all(a['TP']>=b['TP'] and a['FP']>=b['FP'] for a,b in zip(group,group[1:]))
            results.extend(group)
    assert all(hashlib.sha256(f.read_bytes()).hexdigest()==hashes[str(f)] for f in sources)
    def save(name,data):
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    save('secom-intermediate-thresholds.csv',results)
    save('secom-intermediate-threshold-folds.csv',folds)
    print(json.dumps(results,indent=2))
    def pct(v):return 'undefined' if v is None else f'{v*100:.1f}%'
    table='\n'.join(f"| {r['model']} | {r['indicators']} | {r['threshold']:.2f} | {pct(r['recall'])} | {pct(r['precision'])} | {r['FP']} | {r['FN']} | {pct(r['specificity'])} | {pct(r['balanced_accuracy'])} | {r['TP']} | {r['flagged']} |" for r in results)
    report=f'''# SECOM intermediate threshold comparison

    Only saved out-of-fold development scores were used. No models were loaded or trained; no raw measurements or reserved test files were accessed. Fixed candidates: 0.25, 0.30, 0.35, 0.40, 0.45. Both missingness variants are included, so no indicator choice is implicit.

    Population: 1,253 development records, 83 failures and 1,170 passes, with identical saved five-fold memberships. Each record contributes one held-out prediction per model configuration. Rates below pool out-of-fold counts. Flag a record if its score is at least the threshold. Class-weighted scores are not calibrated probabilities.

    Reference: shallow tree at 0.50 detects 48 failures, misses 35, and generates 359 false alarms (811 true negatives). Recall 57.8%, precision 11.8%, specificity 69.3%, balanced accuracy 63.6%; total flagged 407. Its indicator variants have identical decisions at this threshold.

    | Model | Indicators | Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Detected failures | Total flagged |
    |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    {table}

    Validation: record/fold/outcome membership matches the baseline; no duplicate or missing records; summed per-fold confusion counts match the totals; scores are finite in [0,1]; lower thresholds do not reduce detection or false-alarm counts; the baseline reproduces the prior totals; source prediction files are unchanged.

    These are exploratory development operating points evaluated after earlier model and threshold comparisons, not independent confirmation or population performance estimates. No confidence claim or operational optimum is established. Review capacity, missed-failure costs, dependence between production records, and measurement availability remain unresolved. Flags mean additional review of a production record, not automatic rejection or an identified physical defect. No final model or threshold is selected.
    '''
    (out/'secom-intermediate-threshold-comparison.md').write_text(report,encoding='utf-8')


if __name__ == '__main__':
    main()
