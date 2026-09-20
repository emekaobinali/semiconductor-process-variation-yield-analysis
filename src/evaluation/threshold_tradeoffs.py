from src.paths import ROOT, resource, report, require_file, protect_completed_evidence
from pathlib import Path
import csv,json,hashlib

def main():
    protect_completed_evidence()
    """Fixed threshold comparison from existing development OOF predictions only.
    Uses the Python standard library; no raw files, test files, or models are loaded.
    """

    project = ROOT
    source=resource('baseline_comparison/development_oof_predictions.csv')
    out=ROOT / 'artifacts/reproduction'
    out.mkdir(parents=True, exist_ok=True)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='') as f:
        records=[r for r in csv.DictReader(f) if r['model'] in ('logistic','tree')]
    thresholds=[0.20,0.50,0.80]  # Fixed before reviewing threshold results.
    def evaluate(rows,t):
        tp=fp=fn=tn=0
        for r in rows:
            actual=int(r['failure']); predicted=float(r['score'])>=t
            tp+=int(actual==1 and predicted); fp+=int(actual==0 and predicted)
            fn+=int(actual==1 and not predicted); tn+=int(actual==0 and not predicted)
        recall=tp/(tp+fn); specificity=tn/(tn+fp)
        return {'threshold':t,'records':len(rows),'failure_recall':recall,
            'precision':tp/(tp+fp) if tp+fp else None,'false_alarms':fp,
            'missed_failures':fn,'specificity':specificity,
            'balanced_accuracy':(recall+specificity)/2,
            'detected_failures':tp,'correct_passes':tn,'total_flagged':tp+fp}
    pooled=[]; folds=[]
    for model in ('logistic','tree'):
        for indicators in ('False','True'):
            rows=[r for r in records if r['model']==model and r['indicators']==indicators]
            assert len(rows)==1253 and len({r['source_row_1based'] for r in rows})==1253
            assert sum(int(r['failure']) for r in rows)==83
            assert set(int(r['fold']) for r in rows)==set(range(1,6))
            for t in thresholds:
                result={'model':model,'indicators':indicators,**evaluate(rows,t)}
                pooled.append(result)
                local=[]
                for fold in range(1,6):
                    local.append({'model':model,'indicators':indicators,'fold':fold,**evaluate([r for r in rows if int(r['fold'])==fold],t)})
                folds.extend(local)
                for key in ('false_alarms','missed_failures','detected_failures','correct_passes','total_flagged'):
                    assert sum(r[key] for r in local)==result[key]
            values=pooled[-3:]
            assert all(a['detected_failures']>=b['detected_failures'] and a['false_alarms']>=b['false_alarms'] for a,b in zip(values,values[1:]))
            # Reproduce the already-saved 0.5 baseline decisions exactly.
            assert all(int(float(r['score'])>=0.5)==int(r['prediction']) for r in rows)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    for name,rows in [('secom-threshold-tradeoffs.csv',pooled),('secom-threshold-fold-metrics.csv',folds)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(pooled,indent=2))
    def pct(v): return 'undefined' if v is None else f'{v*100:.1f}%'
    sections=[]
    for model in ('logistic','tree'):
        for indicators in ('False','True'):
            group=[r for r in pooled if r['model']==model and r['indicators']==indicators]
            table='\n'.join(f"| {r['threshold']:.2f} | {pct(r['failure_recall'])} | {pct(r['precision'])} | {r['false_alarms']} | {r['missed_failures']} | {pct(r['specificity'])} | {pct(r['balanced_accuracy'])} | {r['total_flagged']} |" for r in group)
            descriptions='\n\n'.join(f"**Threshold {r['threshold']:.2f}:** flag {r['total_flagged']} records for additional review: {r['detected_failures']} actually failed and {r['false_alarms']} passed. Leave {r['missed_failures']} actual failures unflagged. These are retrospective screening analogies, not instructions to scrap product or proof of real deployment performance." for r in group)
            sections.append(f"## {model.title()} — {'with' if indicators=='True' else 'without'} missingness indicators\n\n| Threshold | Recall | Precision | False alarms | Missed failures | Specificity | Balanced accuracy | Total flagged |\n|---|---:|---:|---:|---:|---:|---:|---:|\n{table}\n\n{descriptions}")
    report='''# SECOM threshold tradeoffs — development validation only

    The fixed candidate thresholds were 0.20, 0.50, and 0.80, selected before reviewing results. A record is flagged when its saved failure score is at least the threshold. These are three illustrative operating points, not optimized thresholds. Both missingness variants are included so no feature decision is silently made.

    The only input was the existing development out-of-fold prediction file. Each of 1,253 records (83 failures and 1,170 passes) was previously scored by a model that did not train on it. No raw measurements, timestamps, reserved test files, model objects, or training routines were opened or used. No new models were fitted. The input file remained unchanged.

    Rates below use pooled counts across the existing five folds. Recall is the share of failures caught; precision is the share of flagged records that failed; specificity is the share of passes left unflagged; balanced accuracy averages recall and specificity. False alarms and missed failures are record counts. Full per-fold results are supplied separately. Both models used balanced training class weights; these scores are not established as calibrated probabilities, so 0.20 does not mean a known 20% failure risk.

    '''+ '\n\n'.join(sections)+'''

    ## Options for review

    - Conservative candidate: 0.80, emphasizing fewer false alarms among these candidates, at the cost of missed failures.
    - Middle candidate: 0.50, the original reference decision rule. This is a compromise candidate, not a claim that manufacturing costs are balanced or that balanced accuracy is maximized.
    - Aggressive candidate: 0.20, intended to prioritize failure detection and accept more screening work. A shallow tree has only a few distinct leaf scores, so different thresholds may produce identical decisions; assess the actual counts rather than the label.

    Choosing an operating point requires an agreed cost or review-capacity tradeoff. These development comparisons are exploratory and do not establish an optimal threshold, model, or indicator choice. No final decision has been made. The screening analogy concerns production records, not confirmed individual wafers or physical root causes. Unknown entity dependence and acquisition timing remain limitations. The reserved test set remains untouched.

    ## Verification

    Every model/indicator group has exactly one saved validation prediction per development record. Summed fold confusion counts match pooled counts at each threshold. The 0.50 decisions exactly reproduce the baseline prediction file. Lowering thresholds never reduced detections or false alarms. No threshold search, repeat fitting, or additional experiments were run.
    '''
    (out/'secom-threshold-tradeoffs.md').write_text(report,encoding='utf-8')


if __name__ == '__main__':
    main()
