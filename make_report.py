import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
R=json.load(open('results.json')); m=R['main']; cm_=R['confusion_matrix']
ss=getSampleStyleSheet()
B=ParagraphStyle('B',parent=ss['BodyText'],fontSize=10,leading=14.5,spaceAfter=6)
H1=ParagraphStyle('H1',parent=ss['Heading1'],keepWithNext=1,fontSize=15,spaceBefore=12,spaceAfter=6,textColor=colors.HexColor('#1f4e79'))
H2=ParagraphStyle('H2',parent=ss['Heading2'],keepWithNext=1,fontSize=11.5,spaceBefore=8,spaceAfter=4,textColor=colors.HexColor('#1f4e79'))
T=ParagraphStyle('T',parent=ss['Title'],fontSize=21,leading=26,textColor=colors.HexColor('#1f4e79'))
SM=ParagraphStyle('SM',parent=B,fontSize=8.5,leading=11,textColor=colors.HexColor('#444444'))
C=ParagraphStyle('C',parent=B,fontSize=8.5,leading=11,alignment=1)
BL=ParagraphStyle('BL',parent=B,leftIndent=14,bulletIndent=2,spaceAfter=3)
def bl(t): return Paragraph(t,BL,bulletText='•')
def img(f,w=16.5):
    from PIL import Image as PI
    iw,ih=PI.open(f).size; return Image(f,width=w*cm,height=w*cm*ih/iw)
def tbl(data,cw=None,hdr=True):
    t=Table(data,colWidths=cw,repeatRows=1)
    st=[('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#bbbbbb')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]
    if hdr: st+=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1f4e79')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')]
    t.setStyle(TableStyle(st)); return t
P=lambda t: Paragraph(t,SM)

s=[]
s+=[Paragraph("Anomaly Detection in Industrial Equipment Sensor Data",T),
    Paragraph("<b>Assignment ID:</b> ASGN2703 &nbsp;|&nbsp; <b>Method:</b> Isolation Forest &nbsp;|&nbsp; <b>Data:</b> industrial_sensor_data.csv (3,000 records, 10 columns)",SM),Spacer(1,8)]
s+=[Paragraph("1. Executive summary",H1),
 Paragraph(f"An Isolation Forest was trained on eight standardised features (seven sensor channels plus a derived service-age feature) to detect anomalous equipment readings. "
 f"The model works as designed: it ranks readings by how unusual they are and flags {m['n_flagged']} of {m['n_eval']} labelled records at 5% contamination. "
 f"<b>However, the flagged readings do not line up with the dataset's <font face='Courier'>FaultFlag</font> label</b> (precision {m['precision']:.2f}, recall {m['recall']:.2f}, F1 {m['f1']:.2f}, ROC-AUC {m['roc_auc']:.2f}). "
 f"A supervised Random Forest also fails to predict the label (cross-validated ROC-AUC {R['rf_cv_auc_mean']:.2f}), and no sensor correlates with it (max |r| = {R['max_abs_corr_with_fault']:.2f}). "
 f"We therefore conclude that, in this dataset, <font face='Courier'>FaultFlag</font> carries no signal recoverable from the sensor readings, so the weak scores reflect the data rather than a defect in the model. "
 f"The key recommendation is to validate how <font face='Courier'>FaultFlag</font> is defined and recorded, and to use the anomaly score as a prioritised inspection list whose outcomes become trustworthy labels.",B)]

s+=[Paragraph("2. Dataset and preprocessing",H1),
 Paragraph(f"The dataset contains {R['rows_total']:,} records: SensorID, Temperature, Pressure, Vibration, Humidity, FlowRate, PowerConsumption, RuntimeHours, LastServiceDate and the binary label FaultFlag "
 f"({100*R['fault_rate']:.1f}% faulty among labelled rows). Preprocessing steps:",B),
 bl(f"<b>Missing values.</b> {R['missing_total_cells']} cells ({R['rows_with_any_missing']} rows) were missing, about 1-2% per column. Sequential SensorIDs were reconstructed from row position (verified on all known IDs); missing service dates were filled from neighbouring rows (the data is date-ordered); sensor values were median-imputed (robust to outliers); the label was never imputed, and unlabelled rows were excluded from the metrics only."),
 bl("<b>Feature engineering.</b> LastServiceDate was converted to DaysSinceService relative to the latest date in the data."),
 bl(f"<b>Noise / extremes.</b> The IQR rule found no outliers in any feature; mild winsorising at the 0.5/99.5 percentiles ({R['values_winsorised']} values) limited the influence of isolated spikes. A rolling median (window 25) was used for plotting only."),
 bl("<b>Normalisation.</b> Z-score standardisation of all features."),
 Spacer(1,4), img('figures/01_missing_values.png',11)]

s+=[Paragraph("3. Model and evaluation",H1),
 Paragraph("<b>Model.</b> scikit-learn IsolationForest with 300 trees, max_samples='auto', contamination = 0.05 (a conventional anomaly rate, as the true rate is unknown) and a fixed random seed (42). "
 "Anomalies receive shorter average path lengths and hence higher anomaly scores.",B),
 Paragraph("<b>Metrics.</b> FaultFlag = 1 was treated as the ground-truth 'anomaly'. Precision, recall and F1 depend on the threshold; ROC-AUC uses the continuous score.",B)]
s+=[tbl([['Metric','Value'],['Precision',f"{m['precision']:.3f}"],['Recall',f"{m['recall']:.3f}"],['F1-score',f"{m['f1']:.3f}"],['ROC-AUC',f"{m['roc_auc']:.3f}"],
         ['Confusion matrix [[TN, FP],[FN, TP]]',f"[[{cm_[0][0]}, {cm_[0][1]}], [{cm_[1][0]}, {cm_[1][1]}]]"]],cw=[8*cm,5*cm]),Spacer(1,6)]
sw=[['Contamination','Flagged','Precision','Recall','F1']]+[[f"{r['contamination']:.3f}",r['flagged'],f"{r['precision']:.3f}",f"{r['recall']:.3f}",f"{r['f1']:.3f}"] for r in R['sweep']]
s+=[Paragraph("Sensitivity to the contamination parameter",H2),tbl(sw,cw=[3.2*cm]*5),Spacer(1,4),
 Paragraph(f"Precision stays at 0.38-0.46, essentially the 48% base rate, at every threshold, and F1 rises only because more points are flagged. Even when contamination equals the fault rate the model reaches F1 = {R['sweep'][-1]['f1']:.2f}, which is what random guessing achieves. "
 f"The supervised sanity check (Random Forest, 5-fold CV ROC-AUC = {R['rf_cv_auc_mean']:.3f} &plusmn; {R['rf_cv_auc_std']:.3f}) confirms that the features contain no usable information about the label.",B),
 img('figures/04_evaluation.png',15.5)]

s+=[PageBreak(),Paragraph("4. Visualisation of results",H1),
 Paragraph("Rows are ordered by SensorID (and service date), so record order serves as a time axis. Red points are readings flagged as anomalous.",B),
 img('figures/05_timeseries.png',16.5),
 Paragraph("<i>Figure: sensor time-series with rolling median and flagged anomalies.</i>",C),PageBreak(),
 img('figures/07_scatter_pairs.png',17),Paragraph("<i>Figure: normal (blue) vs anomalous (red) readings for three sensor pairs.</i>",C),Spacer(1,6),
 img('figures/08_pca.png',16),Paragraph("<i>Figure: 2-D PCA projection coloured by anomaly score and by class.</i>",C)]
a=R['mean_extreme_features_anom']; n=R['mean_extreme_features_normal']
s+=[Paragraph("Interpretation",H2),
 Paragraph(f"The sensor features are close to uniformly distributed, independent of one another, and contain no IQR outliers. Isolation Forest therefore flags readings in the <i>corners</i> of the operating envelope: combinations in which several features are simultaneously far from typical. "
 f"Flagged readings have on average {a:.1f} features with |z| &gt; 1.5 compared with {n:.1f} for normal readings, and are farther from the fleet average on every feature, with no single sensor dominating. "
 "The only temporal pattern is at the edges: about 13% of records in the first and last 250 are flagged versus 1-5% in between. DaysSinceService falls steadily from 124 to 0 days across the record order, so the longest-unserviced and most recently serviced units sit at the extremes of that feature and are isolated more easily (25% of flagged units have a service age of 5 days or less, or 120 days or more, versus 8% of normal units).",B),
 img('figures/10_feature_profile.png',11)]

s+=[PageBreak(),Paragraph("5. Potential causes of anomalies",H1),
 Paragraph("The data alone cannot establish causation, so the following are engineering hypotheses to verify during inspection:",B),
 tbl([[Paragraph('<font color=white><b>Signal pattern</b></font>',SM),Paragraph('<font color=white><b>Plausible cause</b></font>',SM)]]+[[P(a),P(b) ] for a,b in [
  ('High vibration + high temperature','Bearing wear, misalignment, imbalance, lubrication failure'),
  ('High pressure with low flow rate','Blockage, fouled filter, partly closed valve'),
  ('Low pressure with high flow rate','Leak, seal or pipe failure, pump cavitation'),
  ('High power consumption at normal output','Motor inefficiency, friction, electrical fault'),
  ('High temperature + high humidity','Cooling / ventilation failure, condensation'),
  ('Long time since service + abnormal readings','Overdue maintenance, accumulated wear'),
  ('Isolated single-sensor spikes','Sensor drift, calibration error, loose wiring (measurement fault, not equipment fault)')]],cw=[6.5*cm,10*cm])]
s+=[Paragraph("6. Recommendations for maintenance and process optimisation",H1)]
for t in [
 "<b>Validate FaultFlag first.</b> A ~48% fault rate that no sensor explains suggests the flag is defined differently from what the sensors measure (e.g. an inspection or work-order flag) or is mis-recorded. Confirm its definition with the maintenance team before building models on it.",
 "<b>Use the anomaly score for triage.</b> Inspect the top-scoring units (listed in anomaly_results.csv) and record the findings. Confirmed outcomes become reliable labels for supervised models.",
 "<b>Check instrumentation.</b> Cross-check flagged readings against neighbouring sensors and recalibrate channels showing isolated spikes.",
 "<b>Adopt condition-based maintenance.</b> Set joint alert thresholds on vibration, temperature and power consumption, and prioritise service for units that are both long overdue and flagged.",
 "<b>Improve data quality.</b> Monitor the collection pipeline for the 1-2% missing values per column and record true timestamps so drift and trends can be modelled.",
 "<b>Operationalise.</b> Re-fit the model periodically, set contamination (e.g. 2-5%) according to inspection capacity, and track the precision of reviewed alerts over time."]: s.append(bl(t))
s+=[Paragraph("7. Limitations",H1),
 Paragraph("Contamination is a tuning choice; record order is only a proxy for time; and the dataset appears to contain no real fault structure (uniform, independent features), so the causes in Section 5 are hypotheses rather than findings.",B),
 Paragraph("<b>Deliverables:</b> anomaly_detection_isolation_forest.ipynb (code, comments, visualisations), this report, README.md, requirements.txt, anomaly_results.csv.",SM)]
def foot(c,d):
    c.saveState(); c.setFont('Helvetica',8); c.setFillColor(colors.grey); c.drawCentredString(A4[0]/2,1.2*cm,f"ASGN2703 - Isolation Forest anomaly detection - page {d.page}"); c.restoreState()
SimpleDocTemplate('ASGN2703_Report.pdf',pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=1.8*cm,bottomMargin=2*cm,title="ASGN2703 Anomaly Detection Report").build(s,onFirstPage=foot,onLaterPages=foot)
