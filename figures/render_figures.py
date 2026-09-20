"""Render saved aggregate results only; no raw data, estimators, or predictions.

Optional figure tooling: reportlab and pypdfium2. Run from any working directory.
"""
from pathlib import Path
import csv, json, hashlib, importlib.metadata
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
NAVY='#162E46'; BLUE='#2267A0'; TEAL='#157A75'; GOLD='#B46720'; GRAY='#566677'; LIGHT='#E6EBF0'
sources={}
def read(name):
    p=ROOT/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def rows(name):
    with read(name).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
trade=rows('reports/results/comparisons/secom-intermediate-thresholds.csv')
models=rows('reports/results/comparisons/secom-model-comparison-summary.csv')
importance=rows('reports/feature_interpretation/secom-feature-all_feature_importances.csv')
metrics=json.loads(read('reports/final_evaluation/secom-final-test-metrics.json').read_text())
read('reports/summaries/technical-engineering-summary.md')
read('config/frozen_method.json')

def canvas(h,title,sub):
    d=Drawing(1000,h); d.add(Rect(0,0,1000,h,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    text(d,40,43,'SECOM  /  SEMICONDUCTOR PROCESS SCREENING',11,GRAY,bold=True)
    text(d,40,81,title,26,NAVY,bold=True)
    text(d,40,108,sub,13,GRAY)
    return d
def text(d,x,y,s,size=13,color=NAVY,bold=False,anchor='start'):
    d.add(String(x,d.height-y,s,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
def line(d,x1,y1,x2,y2,color=LIGHT,width=1):
    d.add(Line(x1,d.height-y1,x2,d.height-y2,strokeColor=HexColor(color),strokeWidth=width))
def box(d,x,y,w,h,title,lines,color=BLUE):
    d.add(Rect(x,d.height-y-h,w,h,fillColor=HexColor('#F5F8FB'),strokeColor=HexColor(LIGHT),strokeWidth=1))
    d.add(Rect(x,d.height-y-h,5,h,fillColor=HexColor(color),strokeColor=None))
    text(d,x+18,y+25,title,15,color,True)
    for i,s in enumerate(lines):text(d,x+18,y+49+i*18,s,12)
def arrow(d,x1,y1,x2,y2,color=GRAY):
    line(d,x1,y1,x2,y2,color,1.5)
    if y1==y2:pts=[x2,d.height-y2,x2-7,d.height-y2+4,x2-7,d.height-y2-4]
    else:pts=[x2,d.height-y2,x2-4,d.height-y2+7,x2+4,d.height-y2+7]
    d.add(Polygon(pts,fillColor=HexColor(color),strokeColor=None))
def save(d,name):
    renderSVG.drawToFile(d,str(OUT/(name+'.svg')))
    pdf=renderPDF.drawToString(d)
    doc=pdfium.PdfDocument(pdf);page=doc[0]
    page.render(scale=2.5).to_pil().save(OUT/(name+'.png'),dpi=(180,180))
    page.close();doc.close()

# 1. Workflow with the reserved branch visibly separated from development.
d=canvas(830,'From process measurements to an honest final test','Completed workflow  |  All selection decisions used development data')
box(d,40,135,920,76,'1  Inspect SECOM',['1,567 records  |  590 anonymous measurements  |  104 failures; missing values preserved'])
arrow(d,320,211,320,231)
box(d,40,231,920,76,'2  Save the stratified split',['Development: 1,253 records (83 failures)  |  Reserved test: 314 records (21 failures)'])
arrow(d,320,307,320,330)
box(d,40,330,570,104,'3  Develop and compare',['Same five folds; preprocessing fitted within training folds','Baselines + constrained ensembles; compare missingness indicators','Use saved out-of-fold scores to assess threshold tradeoffs'])
arrow(d,790,307,790,330)
box(d,650,330,310,210,'RESERVED TEST HELD ASIDE',['No model selection','No threshold selection','No preprocessing fitting','','Access only after method freeze'],GOLD)
arrow(d,320,434,320,457)
box(d,40,457,570,83,'4  Freeze the approved method',['Random Forest  |  No missingness indicators  |  Threshold 0.35','Fit once on all development records'],TEAL)
arrow(d,320,540,320,565);arrow(d,790,540,790,565,GOLD)
box(d,40,565,920,86,'5  Evaluate the reserved test once',['14 failures caught  |  7 missed  |  102 false alarms  |  191 passes correctly identified','No tuning or method changes after seeing the result'],TEAL)
arrow(d,500,651,500,675)
box(d,40,675,920,82,'6  Interpret and document',['Anonymous-feature importance: in-sample development diagnostic; no causal attribution','Useful screening signal, but not production-ready'])
text(d,40,798,'Sources: saved technical summary, frozen-method record and final-test metrics. Chronological sensitivity remains future work.',11,GRAY)
save(d,'secom-project-workflow')

# 2. Four saved candidate points; no interpolation, optimization or new thresholds.
d=canvas(670,'Catching failures has a review-workload cost','Development out-of-fold results  |  1,253 records: 83 failures, 1,170 passes')
x0,y0,w,h=95,490,825,330
X=lambda n:x0+n/450*w
Y=lambda n:y0-n/83*h
for y in [0,20,40,60,83]:
    line(d,x0,Y(y),x0+w,Y(y));text(d,x0-12,Y(y)+4,str(y),12,GRAY,anchor='end')
for x in [0,100,200,300,400,450]:text(d,X(x),y0+23,str(x),12,GRAY,anchor='middle')
line(d,x0,y0,x0+w,y0,GRAY);line(d,x0,Y(83),x0,y0,GRAY)
text(d,x0,142,'Failures caught (of 83)',13,NAVY,True)
text(d,510,540,'False alarms: passing records sent for review',14,NAVY,anchor='middle')
def candidate(model,t):return next(r for r in trade if r['model']==model and r['indicators']=='False' and float(r['threshold'])==t)
rf35=candidate('small_forest',.35);rf40=candidate('small_forest',.40);gb=candidate('shallow_boosting',.25)
tree=next(r for r in models if r['model']=='tree' and r['indicators']=='False')
points=[(gb,'Gradient boosting 0.25',BLUE,-25,33),(rf40,'Random Forest 0.40',BLUE,-70,35),(tree,'Shallow tree 0.50',GRAY,-25,-30),(rf35,'Random Forest 0.35  SELECTED',TEAL,-244,-35)]
for r,label,col,dx,dy in points:
    fp,tp=int(r['FP']),int(r['TP']);x,y=X(fp),Y(tp)
    if r is rf35:
        d.add(Polygon([x,d.height-y+9,x+9,d.height-y,x,d.height-y-9,x-9,d.height-y],fillColor=HexColor(col),strokeColor=HexColor(col)))
    else:d.add(Circle(x,d.height-y,6,fillColor=HexColor('#FFFFFF'),strokeColor=HexColor(col),strokeWidth=2))
    text(d,x+dx,y+dy,label,13,col,True)
    text(d,x+dx,y+dy+17,f'{tp} caught / {fp} false alarms',12,GRAY)
text(d,40,585,'Selected RF vs tree: +17 failures caught, +28 false alarms.',15,TEAL,True)
text(d,40,609,'Selected RF vs RF 0.40: +17 caught, +139 false alarms. All candidates shown exclude indicators.',12,GRAY)
text(d,40,640,'Source: saved intermediate-threshold and model-comparison tables. Selection evidence, not independent validation.',11,GRAY)
save(d,'secom-development-screening-tradeoffs')

# 3. Exact saved confusion counts, categorical shading rather than distorted intensity.
assert [metrics[k] for k in ['TN','FP','FN','TP']]==[191,102,7,14]
d=canvas(690,'Final test: useful signal, substantial false alarms','Frozen Random Forest  |  No missingness indicators  |  Score threshold 0.35')
text(d,605,154,'MODEL SCREENING DECISION',12,GRAY,True,anchor='middle')
text(d,425,185,'Predicted pass',17,NAVY,True,anchor='middle');text(d,775,185,'Flagged for review',17,NAVY,True,anchor='middle')
text(d,40,266,'Actual pass',16,NAVY,True);text(d,40,288,'293 records',12,GRAY)
text(d,40,417,'Actual failure',16,NAVY,True);text(d,40,439,'21 records',12,GRAY)
for x,y,key,label,col,bg in [(250,205,'TN','Correctly identified passes',TEAL,'#E7F2F0'),(600,205,'FP','False alarms',GOLD,'#FCF0E5'),(250,360,'FN','Missed failures',GOLD,'#FCF0E5'),(600,360,'TP','Failures caught',TEAL,'#E7F2F0')]:
    d.add(Rect(x,d.height-y-145,340,145,fillColor=HexColor(bg),strokeColor=HexColor('#FFFFFF')))
    text(d,x+170,y+70,str(metrics[key]),46,col,True,'middle')
    text(d,x+170,y+108,label,15,col,False,'middle')
text(d,40,550,'Recall 66.7%   |   Precision 12.1%   |   Specificity 65.2%',17,NAVY,True)
text(d,40,581,'Balanced accuracy 65.9%   |   Average precision 0.1872',16,NAVY)
text(d,40,616,'116 of 314 records flagged; only 14 flags were failures. Not production-ready.',14,GOLD,True)
text(d,40,654,'Source: saved one-time reserved-test metrics. Counts, not row-normalized percentages. No test replay performed.',11,GRAY)
save(d,'secom-reserved-test-confusion-matrix')

# 4. Distinct scales, each method's own top ten; visual comparison without conflating units.
d=canvas(770,'Which anonymous measurements mattered to this model?','Frozen Random Forest  |  Built-in importance vs in-sample development permutation importance')
topbuilt=sorted(importance,key=lambda r:float(r['builtin_importance']),reverse=True)[:10]
topperm=sorted(importance,key=lambda r:float(r['permutation_AP_loss']),reverse=True)[:10]
shared=set(r['feature'] for r in topbuilt)&set(r['feature'] for r in topperm)
assert len(shared)==4
for left,rs,key,maxv,title,subtitle in [(40,topbuilt,'builtin_importance',.06,'Built-in importance','Share of total weighted impurity decrease (%)'),(525,topperm,'permutation_AP_loss',.022,'Permutation importance','Mean decrease in average precision (AP)')]:
    text(d,left,153,title,18,NAVY,True);text(d,left,176,subtitle,11,GRAY)
    bx=left+112; bw=310
    for tick in ([0,.02,.04,.06] if key=='builtin_importance' else [0,.005,.010,.015,.020]):
        px=bx+tick/maxv*bw;line(d,px,192,px,591)
        text(d,px,613,f'{tick*100:.0f}' if key=='builtin_importance' else f'{tick:.3f}',11,GRAY,anchor='middle')
    for i,r in enumerate(rs):
        y=213+i*38;v=float(r[key]);col=TEAL if r['feature'] in shared else BLUE
        text(d,bx-12,y+4,'M'+r['feature'].split('_')[1],13,NAVY,r['feature'] in shared,'end')
        d.add(Rect(bx,d.height-y-9,v/maxv*bw,18,fillColor=HexColor(col),strokeColor=None))
        if key=='permutation_AP_loss':
            sd=float(r['permutation_shuffle_SD']);a=bx+(v-sd)/maxv*bw;b=bx+(v+sd)/maxv*bw
            line(d,a,y,b,y,NAVY,1.2);line(d,a,y-5,a,y+5,NAVY,1.2);line(d,b,y-5,b,y+5,NAVY,1.2)
            text(d,b+7,y+4,f'{v:.4f}',10,NAVY)
        else:text(d,bx+v/maxv*bw+7,y+4,f'{100*v:.2f}%',11,NAVY)
text(d,40,650,'M = anonymous measurement. Bold teal labels appear in both top-ten lists (4 of 10).',13,TEAL,True)
text(d,40,674,'Permutation error bars: ±1 shuffle SD over five repeats; not confidence intervals. Separate panel scales.',12,GRAY)
text(d,40,698,'Training-data reliance only; correlated features and imputation can affect ranks. Association is not causation.',12,GOLD,True)
text(d,40,735,'Source: saved feature-importance table. No physical names or units inferred; no new importance calculations.',11,GRAY)
save(d,'secom-anonymous-feature-importance')

readme=ROOT/'README.md';s=readme.read_text(encoding='utf-8');old=s
changes=[('> Figure placeholder: workflow diagram. No figure has been generated yet.','![Completed SECOM workflow with a reserved-test branch held aside until the method is frozen.](figures/secom-project-workflow.png)'),('> Figure placeholder: development threshold tradeoff chart using the saved results only.','![Development failures caught versus false alarms for four saved screening candidates, highlighting Random Forest at 0.35.](figures/secom-development-screening-tradeoffs.png)'),('> Figure placeholder: visualization of this saved confusion matrix.','![Reserved-test confusion matrix: 191 true negatives, 102 false positives, 7 false negatives, and 14 true positives.](figures/secom-reserved-test-confusion-matrix.png)')]
for a,b in changes:
    if a in s:
        assert s.count(a)==1;s=s.replace(a,b)
    else:
        assert s.count(b)==1
anchor='## Feature-interpretation summary\n'
assert s.count(anchor)==1
feature='\n![Top ten anonymous measurements by built-in and in-sample permutation importance; predictive associations only.](figures/secom-anonymous-feature-importance.png)\n'
if feature not in s:
    s=s.replace(anchor,anchor+feature)
readme.write_text(s,encoding='utf-8')
for name,h in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
(OUT/'figure-provenance.json').write_text(json.dumps({'sources_sha256':sources,'source_data_modified':False,'models_loaded':False,'new_scientific_analysis':False,'formats':['PNG (2500 px wide)','SVG (vector)'],'renderer_versions':{p:importlib.metadata.version(p) for p in ['reportlab','pypdfium2']},'README_edits':'Three figure placeholder replacements plus one figure immediately under Feature-interpretation summary. No prose changes.'},indent=2))
print('Saved four PNG/SVG pairs; source hashes unchanged; README figure references inserted.')
