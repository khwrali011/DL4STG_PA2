"""Produce measured tables and an explicitly marked report until review is complete."""
import argparse,csv,json,math
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

O=Path('outputs');O.mkdir(exist_ok=True)
styles=getSampleStyleSheet();styles['BodyText'].fontSize=9;styles['BodyText'].leading=13

def fmt(x,scientific=False):
    if isinstance(x,str):return x
    if isinstance(x,int):return str(x)
    if scientific or (x!=0 and abs(x)<1e-3):return f'{x:.2e}'
    if x==0:return '0'
    digits=2-math.floor(math.log10(abs(x)))
    return f'{x:.{max(0,digits)}f}' if digits>=0 else str(int(round(x,digits)))

def metrics(name,seed):
    p=O/f'{name}_seed{seed}_metrics.json'
    return json.loads(p.read_text()) if p.exists() else None

def value(name,key,sci=False):
    a=metrics(name,0)
    if not a:return 'PENDING'
    if name in ['Blank','Copy-last']:return fmt(a[key],sci)
    b=metrics(name,1)
    if not b:return 'PENDING (seed 1)'
    if key=='parameters':return str(a[key])
    x,y=a[key],b[key]
    return f'{fmt((x+y)/2,sci)} ({fmt(x,sci)}, {fmt(y,sci)})'

def make_tables():
    table1=json.loads((O/'table1.json').read_text())
    tables={'1':[['Check','Expected','Measured']]+[[r['check'],r['expected'],r['yours']] for r in table1]}
    specs={
      '2':(['A','Blank','Copy-last'],['MSE1','MSE2','MSE3','MSE4','MSE5','E5_1','parameters']),
      '3':(['A','Blank','Copy-last'],['E5_1','CM','EM','CR','ER']),
      '4':(['A','Copy-last'],['E2_1','E2_2','E2_3','E5_1','E5_2']),
      '5':(list('ABCD'),['E5_1','CM','EM','CR','ER','E2_1','E2_2','E2_3','parameters','train_seconds','inference_seconds']),
      '6':(['symB','symD'],['E5_1','CM','EM','CR','ER','E2_1','E2_2','E2_3','parameters','train_seconds','inference_seconds'])}
    for n,(names,keys) in specs.items():
        # Transposed for readable two-seed entries, preserving every requested field.
        tables[n]=[['Metric']+names]+[[key]+[value(name,key,n=='6' and key in ['CM','CR']) for name in names] for key in keys]
    for n,rows in tables.items():
        with (O/f'table{n}.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    return tables

def table_block(rows):
    cells=[[Paragraph(escape(str(c)),styles['BodyText']) for c in row] for row in rows]
    widths=[72]+[(475-72)/(len(rows[0])-1)]*(len(rows[0])-1)
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf5')),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.lightgrey),
                          ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return t

def main(strict=False):
    tables=make_tables();review=json.loads(Path('review.json').read_text())
    missing=[]
    for name in ['A','B','C','D','symB','symD']:
        for seed in [0,1]:
            if metrics(name,seed) is None:missing.append(f'{name} seed {seed} evaluation')
    for name in ['Blank','Copy-last']:
        if metrics(name,0) is None:missing.append(name+' evaluation')
    for p in ['figure1.png','figure4.png','figure5.png','figure6b.png']:
        if not (O/p).exists():missing.append(p)
    for pattern,count in [('figure2_video*.png',3),('figure3_*.png',4),('figure3_*.gif',4),('figure6a_*.png',4),('figure6a_*.gif',4)]:
        if len(list(O.glob(pattern)))!=count:missing.append(pattern)
    for p in ['Q2_5.txt','Q2_11.txt']:
        if not Path('predictions',p).exists():missing.append('Pre-experiment prediction '+p)
    sel=json.loads((O/'selection.json').read_text()) if (O/'selection.json').exists() else {}
    if not sel.get('reviewed'):missing.append('Visual confirmation of overlap/separation examples')
    if not review.get('github_url','').startswith('https://github.com/'):missing.append('Public GitHub URL')
    if not review.get('repository_public_confirmed'):missing.append('Public repository confirmation')
    for q in ['Q2.1','Q2.4','Q2.7','Q2.8','Q2.9','Q2.13','Q2.14','Q2.15','Q2.16','Q2.17']:
        if not review.get('answers',{}).get(q,'').strip():missing.append('Result-based answer '+q)
    if len(review.get('answers',{}).get('Q2.17','').split())>150:missing.append('Q2.17 exceeds 150 words')
    if not review.get('ai_edits','').strip():missing.append('AI disclosure: actual edits/review')
    if not review.get('predictions_reviewed'):missing.append('Prediction texts and Git history reviewed')
    status={'submission_ready':not missing,'missing':missing,
      'note':'The full assignment requires Q1 and Q2 together in one PDF; this package covers Q2 only.'}
    (O/'completion_status.json').write_text(json.dumps(status,indent=2))
    if strict and missing:raise SystemExit('Submission blocked:\n- '+'\n- '.join(missing))
    theory={
    'Q2.2':('Mirroring maps horizontal displacement dx to -dx and leaves dy unchanged. Anticlockwise rotation maps screen-coordinate (dx,dy) to (dy,-dx). Both preserve speed and the validity of straight motion and wall bounces, but alter digit appearance. Whether their direction statistics are familiar must be checked in Figure 1; physical validity alone does not prove distributional symmetry. Speed-up changes temporal displacement and can hide intermediate bounces, while retaining digit appearance in sampled frames. Together these perturbations help separate appearance sensitivity from speed generalisation.'),
    'Q2.3':('A folded grid cell represents 4 original pixels centre-to-centre. Each 3x3 convolution reaches one cell per axis. The longest within-step path is new input -> layer 1 -> layer 2 -> output, or previous layer-1 hidden state -> layer 1 -> layer 2 -> output: two spatial convolutions. Thus reach is 2 x 1 x 4 = 8 pixels per axis (Chebyshev distance), or 8 sqrt(2), approximately 11.3 pixels, at a diagonal corner. Previous layer-2 state traverses only one convolution. Gates are parallel channel groups, not four serial convolutions. Elementwise cell updates, the 1x1 head and folding/unfolding add no reach. Earlier recurrence increases accumulated history but not the stipulated one-step reach. This is a dependency bound, not a guaranteed tracking threshold.'),
    'Q2.6':('Blank and Copy-last commute exactly with mirror and rotation because zero is unchanged by pixel permutations, and copying the last frame commutes with applying the same permutation to every frame. Both therefore have zero consistency gaps even when their forecasts are wrong. C_T measures agreement between two processing routes; E_T measures agreement with the transformed true future. Reporting only C_T could reward a useless blank predictor.'),
    'Q2.10':('Baseline results alone confound unfamiliar speed with limited architectural reach. Compare A with C: their architectures match and only speed training changes. Then compare C with D: their speed distribution and training protocol match, while D has larger reach. Run both seeds, score the same first two horizons on the same test videos, and use validation only for design choices. Improvement from A to C supports a distribution-coverage explanation. Additional improvement from C to D supports a reach-related explanation, although changed spatial sampling and optimisation are alternative contributors; it is not definitive proof of a unique cause.'),
    'Q2.12':('D changes the dilation of both 3x3 ConvLSTM convolutions from 1 to 2, with padding 2. All channels, biases, folding, head, initial states, nine-step rollout and training settings remain unchanged. The two-layer within-step path now reaches 2 x 2 x 4 = 16 original pixels per axis, or 16 sqrt(2), approximately 22.6 pixels, diagonally. D has 480,784 trainable parameters, unchanged from A and below the limit of 721,176. Layer 1: 4 x 64 x (16+64) x 9 + 256 = 184,576. Layer 2: 4 x 64 x (64+64) x 9 + 256 = 295,168. Head: 16 x 64 + 16 = 1,040. This fixed design was chosen before test evaluation. Dilated kernels sample sparsely, so increased reach need not improve accuracy.'),
    'Symmetry proof':('For any a in G, define h = g a. Right multiplication permutes the finite group. Therefore fbar(aX) = (1/8) sum_g g^-1 f(g a X) = (1/8) sum_h a h^-1 f(hX) = a fbar(X). These transformations permute pixels and are linear, so a can move outside the average. Taking a=M or R gives exact consistency in real arithmetic for any forecaster f, regardless of accuracy. Floating-point summation can leave tiny nonzero gaps; report measured values rather than replacing them with zero. The final model is symmetrised D, as prescribed; optional D+B is not used.')}
    story=[]
    def p(text):story.append(Paragraph(escape(text).replace('\n','<br/>'),styles['BodyText']));story.append(Spacer(1,6))
    def heading(text):story.append(Paragraph(escape(text),styles['Heading2']))
    story.append(Paragraph('Question 2 - Motion or Memory?',styles['Title']))
    p('Khawar Ali | AI651 - Deep Learning for Space, Time and Graphs | Assignment 2')
    p('GitHub: '+(review.get('github_url') or 'PENDING - public repository URL required'))
    p('STATUS: '+('Q2 requirements recorded; combine with Q1 for the final LMS report.' if not missing else 'INCOMPLETE DRAFT - not ready for submission. No missing experimental results have been invented.'))
    if missing:p('Outstanding: '+'; '.join(missing))
    heading('Protocol and interpretation')
    p('Use all 10,000 Moving MNIST videos with split=None. Train indices 0-7999; validation 8000-8999; test 9000-9999. Normalise uint8 frames to float32 /255. Train A-D for 30 epochs with Adam, learning rate 0.001, batch 32, no schedule, seeds 0 and 1, retaining final weights. Outputs always contain five frames; score available targets only, with equal per-video loss. No teacher forcing or detached feedback is used. Test metrics use all 1,000 test videos at t=1. Tables use mean (seed 0, seed 1); references use one deterministic value. Scientific notation is used below 0.001. Parameters are exact integers.')
    p('Metric key: E5_1 = five-horizon error at 1x; E5_2 = five-horizon error at 2x. E2_k = first-two-horizon error at speed k. CM/CR = mirror/rotation consistency gaps; EM/ER = transformed errors. Timings are seconds. Training time is the mean over all 30 epochs; inference time sums only timed model computation after each batch reaches the device. Symmetrised inference includes transforms, all eight predictions, inverse transforms and averaging.')
    hardware={metrics(n,s)['hardware'] for n in 'ABCD' for s in [0,1] if metrics(n,s)}
    p('Measured training hardware: '+(', '.join(sorted(hardware)) if hardware else 'PENDING - no full training run yet. CPU implementation checks completed separately.'))
    heading('Table 1 - Transformation and window checks');story.append(table_block(tables['1']))
    for n in range(1,18):
        q=f'Q2.{n}';heading(q)
        if n in [5,11]:
            f=Path('predictions')/f'Q2_{n}.txt';p(f.read_text() if f.exists() else 'PENDING: must be written and committed before the relevant experiment. The pipeline creates this record before execution.')
        elif q in theory:p(theory[q])
        else:p(review.get('answers',{}).get(q) or 'PENDING: inspect the corresponding measured tables and figures, then enter the evidence-based answer in review.json. No result is assumed.')
        if n==1:
            p('Figure 1 uses exactly the prescribed Farneback settings and foreground/magnitude mask. Its caption reports the retained-pixel count and median.')
        if n==12:heading('Exact symmetrisation');p(theory['Symmetry proof'])
    for n in range(2,7):
        story.append(PageBreak());heading(f'Table {n}');story.append(table_block(tables[str(n)]))
    for n in range(1,7):
        files=sorted(O.glob(f'figure{n}*.png'))
        if files:
            for file in files:
                story.append(PageBreak());heading(f'Figure {n} - {file.stem}')
                from PIL import Image as PILImage
                with PILImage.open(file) as im:w,h=im.size
                scale=min(475/w,610/h);story.append(Image(str(file),width=w*scale,height=h*scale))
                gif=file.with_suffix('.gif')
                if gif.exists():p('Animation path: outputs/'+gif.name)
                if n in [2,5,6]:p('Selection confirmation: '+('visually reviewed' if sel.get('reviewed') else 'PENDING visual review of overlap/separation requirements'))
        else:heading(f'Figure {n}');p('PENDING - generated after the required data or trained models are available.')
    story.append(PageBreak());heading('AI-use disclosure')
    p('Prompt: "complete question 2 and give submission and completion status". Output: ChatGPT generated the Python experiment implementation, Kaggle runner, synthetic checks, theoretical explanations and report scaffolding. At package creation, no Moving MNIST model had been trained in the assistant environment and no model-performance values were generated. The executable scripts produce empirical outputs only after real training/evaluation. Prediction texts generated by the pipeline are AI-assisted forecasts and must be reviewed before their experiments.')
    p('Edits and checks by the student: '+(review.get('ai_edits') or 'PENDING - describe the edits and verification actually performed; do not claim unperformed work.'))
    heading('References')
    p('AI651, Fall 2026, Assignment 2, Question 2: Motion or Memory?, pp. 4-13. Shi et al. (2015), Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, NeurIPS. Srivastava et al. (2015), Unsupervised Learning of Video Representations using LSTMs, ICML.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawString(42,26,'AI651 | Question 2');canvas.drawRightString(553,26,str(doc.page))
    SimpleDocTemplate(str(O/'Q2_Report.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps(status,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--strict',action='store_true');main(p.parse_args().strict)
