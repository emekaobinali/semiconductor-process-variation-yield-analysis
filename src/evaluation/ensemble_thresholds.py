from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import csv,json,hashlib,collections

def main():
    protect_completed_evidence()
    """Three fixed threshold candidates; saved OOF predictions only; standard library."""
    project = ROOT
    out=ROOT / 'artifacts/reproduction'
    out.mkdir(parents=True, exist_ok=True)
    paths=[resource('stronger_comparison/development_oof_predictions.csv'),resource('baseline_comparison/development_oof_predictions.csv')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    records=[]
    for p in paths:
        with p.open(newline='') as f:
            records.extend(r for r in csv.DictReader(f) if r['model'] in ['small_forest','shallow_boosting','tree'])
    def calc(rows,t):
        c=collections.Counter((int(r['failure']),int(float(r['score'])>=t)) for r in rows)
        tn,fp,fn,tp=[c[k] for k in [(0,0),(0,1),(1,0),(1,1)]]
        recall=tp/(tp+fn);specificity=tn/(tn+fp)
        return dict(threshold=t,recall=recall,precision=tp/(tp+fp) if tp+fp else None,FP=fp,FN=fn,specificity=specificity,balanced_accuracy=(recall+specificity)/2,TP=tp,TN=tn,flagged=tp+fp)
    results=[];fold_results=[];memberships=[]
    for model in ['tree','small_forest','shallow_boosting']:
        for flags in ['False','True']:
            rows=[r for r in records if r['model']==model and r['indicators']==flags]
            assert len(rows)==1253 and len({r['source_row_1based'] for r in rows})==1253
            assert sum(int(r['failure']) for r in rows)==83
            assert all(0<=float(r['score'])<=1 for r in rows)
            assert all(int(float(r['score'])>=0.5)==int(r['prediction']) for r in rows)
            memberships.append({r['source_row_1based']:(r['fold'],r['failure']) for r in rows})
            for t in [0.2,0.5,0.8]:
                result=dict(model=model,indicators=flags,**calc(rows,t));results.append(result)
                local=[dict(model=model,indicators=flags,fold=f,**calc([r for r in rows if int(r['fold'])==f],t)) for f in range(1,6)]
                fold_results.extend(local)
                for key in ['TP','TN','FP','FN']:assert sum(r[key] for r in local)==result[key]
            a,b,c=results[-3:]
            assert a['TP']>=b['TP']>=c['TP'] and a['FP']>=b['FP']>=c['FP']
    assert all(m==memberships[0] for m in memberships)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in paths)
    def save(path,rows):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'secom-ensemble-threshold-summary.csv',results)
    save(out/'secom-ensemble-threshold-folds.csv',fold_results)
    print(json.dumps(results,indent=2))
    def pct(x):return 'undefined' if x is None else f'{100*x:.1f}%'
    table='\n'.join(f"| {r['model']} | {r['indicators']} | {r['threshold']:.2f} | {pct(r['recall'])} | {pct(r['precision'])} | {r['FP']} | {r['FN']} | {pct(r['specificity'])} | {pct(r['balanced_accuracy'])} | {r['TP']} | {r['flagged']} |" for r in results)
    report=f'''# SECOM ensemble threshold comparison

    Only saved out-of-fold development predictions were used. No model training, raw-data inspection, or reserved-test access occurred. Thresholds 0.20, 0.50, and 0.80 were fixed before examining the results. Both missingness variants remain in scope. Scores at or above a threshold trigger a flag.

    All configurations cover the same 1,253 development records, including 83 failures and 1,170 passes, using the same five folds. Each record contributes once to pooled counts. Rates below are computed from pooled counts; per-fold metrics are provided separately. These are exploratory development comparisons, not independent test performance or evidence of a statistically established advantage. Weighted-model scores are not calibrated probabilities and the same numeric threshold does not imply the same review workload across models.

    | Model | Indicators | Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Detected failures | Total flagged |
    |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    {table}

    Within each model, 0.80 is the conservative candidate, 0.50 the middle candidate, and 0.20 the aggressive candidate. These names describe the relative alert rates of this fixed candidate set, not an optimized economic balance. Flags represent records proposed for further review, not confirmed individual wafers or instructions to scrap product. No final model or threshold is selected.

    Validation: all configurations have identical record, fold, and outcome memberships; the stored 0.50 predictions are reproduced; fold confusion counts reconcile to pooled totals; lowering thresholds never reduces detections or false alarms. Source prediction files are unchanged. No threshold search was conducted beyond these three candidates.
    '''
    (out/'secom-ensemble-threshold-comparison.md').write_text(report,encoding='utf-8')


if __name__ == '__main__':
    main()
