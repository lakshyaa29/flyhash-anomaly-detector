"""Build the three-page report from verified baseline results and figures."""

import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
WIDTH, HEIGHT = A4
CONTENT_WIDTH = WIDTH - 84
INK = colors.HexColor("#172A42")
MUTED = colors.HexColor("#4C5968")


def main():
    summary = json.loads((ROOT / "results/baseline_summary.json").read_text(encoding="utf-8"))
    standard = summary["scenarios"]["Standard NSL-KDD test"]
    heldout = summary["scenarios"]["Held-out neptune"]
    example = json.loads((ROOT / "results/raw_baseline_seed42.json").read_text(encoding="utf-8"))
    heldout_runs = [json.loads((ROOT / f"results/heldout/neptune/raw_baseline_seed{s}.json").read_text(encoding="utf-8")) for s in summary["seeds"]]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle("Deck", fontName="Helvetica", fontSize=10, leading=14, textColor=MUTED, spaceAfter=12))
    styles.add(ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=INK, spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle("Text", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=7))
    styles.add(ParagraphStyle("SmallText", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=6))
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.4, leading=11))
    styles.add(ParagraphStyle("HeadCell", fontName="Helvetica-Bold", fontSize=8.4, leading=11, textColor=colors.white))
    styles.add(ParagraphStyle("Number", fontName="Helvetica", fontSize=8.3, leading=11, alignment=TA_CENTER))
    story = []

    def p(text, style="Text"):
        return Paragraph(text, styles[style])

    def add(text, style="Text"):
        story.append(p(text, style))

    def heading(text):
        add(text, "Section")

    def picture(name, caption):
        picture = Image(str(REPORT / "figures" / name))
        picture.drawHeight *= CONTENT_WIDTH / picture.drawWidth
        picture.drawWidth = CONTENT_WIDTH
        story.append(picture)
        add(caption, "SmallText")

    def table(rows, widths, numeric=False):
        formatted = [[p(str(value), "HeadCell" if i == 0 else ("Number" if numeric and j else "Cell"))
                      for j, value in enumerate(row)] for i, row in enumerate(rows)]
        tab = Table(formatted, colWidths=widths, hAlign="LEFT", repeatRows=1)
        tab.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#D9D9D9")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([tab, Spacer(1, 7)])

    def metric(entry, name, percent=True):
        stats = entry["metrics"][name]
        factor = 100 if percent else 1
        decimals = 2 if percent else 3
        return f"{factor * stats['mean']:.{decimals}f}<br/>± {factor * stats['sample_sd']:.{decimals}f}"

    add("FlyHash Network<br/>Anomaly Detector", "ReportTitle")
    add("Team sprint | Siddhesh, Hansika and Lakshya<br/>Mentor: Ashutosh Gopale | Baseline report | 17 October 2026", "Deck")
    add(f"The raw-feature perceptron benchmark is complete. Across three seeds, standard-test accuracy is <b>{100 * standard['metrics']['accuracy']['mean']:.2f}%</b>, attack recall is <b>{100 * standard['metrics']['recall']['mean']:.2f}%</b> and false-positive rate is <b>{100 * standard['metrics']['false_positive_rate']['mean']:.2f}%</b>. The FlyHash comparison awaits Siddhesh's encoder.")
    heading("Approach and implementation")
    add("The project will compare a perceptron on preprocessed features with the same classifier on sparse binary FlyHash codes. The completed baseline uses a NumPy perceptron with zero initial weights and bias, online misclassification updates, learning rate 0.01, at most 100 epochs and shuffle seeds 42, 43 and 44. Scores at least zero predict an attack.")
    add("Hansika's preprocessing design uses median imputation and standardization for 38 numeric columns, and most-frequent imputation plus one-hot encoding for protocol, service and flag. Label and difficulty are removed. Fitting uses training records only; the fitted transformation produces 122 float32 features for both partitions.")
    table([
        ["Evaluation", "Training records", "Test records", "Test normal / attack"],
        ["Standard NSL-KDD", "125,973", "22,544", "9,711 / 12,833"],
        ["Held-out neptune", "84,759", "14,368", "9,711 / 4,657"],
    ], [138, 112, 101, CONTENT_WIDTH - 351])
    heading("Held-out attack protocol")
    add("The individual attack type <b>neptune</b> was fixed from Hansika's example before evaluation. All 41,214 such training rows are removed before fitting a new preprocessor and three fresh models. Evaluation uses only original KDDTest+ normal and neptune rows. The held-out attack never enters training or parameter selection; other test attacks are outside this scenario.")
    heading("Verification and individual contributions")
    add("A separate synthetic check trained on 225 records and correctly classified all 75 test records: confusion matrix [[38, 0], [0, 37]], with zero false positives. This easy check verifies basic classifier behavior; the measured NSL-KDD results follow on page 2.")
    add("<b>Hansika:</b> preprocessing design and reference scripts. <b>Lakshya:</b> perceptron, metric evaluation, repeated baselines, validation, plots, report and runnable integration harness. <b>Siddhesh:</b> FlyHash encoder, pending integration.")

    story.append(PageBreak())
    add("Measured baseline results", "ReportTitle")
    add("Three fixed training seeds on each fixed test population", "Deck")
    table([
        ["Raw-feature model", "Accuracy", "Precision", "Recall", "FPR", "ROC AUC"],
        ["Standard test"] + [metric(standard, name) for name in ["accuracy", "precision", "recall", "false_positive_rate"]] + [metric(standard, "roc_auc", False)],
        ["Held-out neptune"] + [metric(heldout, name) for name in ["accuracy", "precision", "recall", "false_positive_rate"]] + [metric(heldout, "roc_auc", False)],
    ], [106] + [(CONTENT_WIDTH - 106) / 5] * 5, numeric=True)
    add("Table 1. Mean ± sample standard deviation (ddof=1). Rates are percentages; their SDs are percentage points. AUC is unitless. These SDs describe seed variation, not a confidence interval.", "SmallText")
    picture("baseline_roc.png", "Figure 1. Continuous-score ROC curves for seeds 42, 43 and 44. Dots show the fixed score-zero operating points. Each panel uses its stated training and test population.")
    heading("Security interpretation")
    add(f"On the standard test, the baseline misses {100 * (1 - standard['metrics']['recall']['mean']):.1f}% of attacks and flags {100 * standard['metrics']['false_positive_rate']['mean']:.1f}% of normal traffic on average. High precision alone therefore does not establish strong attack detection. Recall is TP/(TP+FN); false-positive rate is FP/(FP+TN).")
    add(f"Held-out neptune recall ranges from {100 * min(r['recall'] for r in heldout_runs):.2f}% to {100 * max(r['recall'] for r in heldout_runs):.2f}% across seeds. This variation shows sensitivity to sample order. The restricted held-out evaluation has a different attack mix and class balance, so its higher accuracy is not a measured improvement over the standard scenario.")
    heading("Limits of the current evidence")
    add("All six models reached 100 epochs and still updated their weights in the final epoch; convergence is not demonstrated. No threshold or model setting was selected on either test set. Results cover one dataset split and one held-out attack type. No FlyHash accuracy, speed advantage or general superiority is claimed before encoder integration.")
    add("Per-seed confusion counts and rates are retained in JSON/CSV files. Prediction scores are retained locally for full verification and can be regenerated from the repository; large prediction archives and datasets are excluded from the GitHub package.", "SmallText")

    story.append(PageBreak())
    add("Diagnostics and reproducibility", "ReportTitle")
    picture("baseline_confusion_matrices.png", "Figure 2. Fixed seed 42, selected consistently for illustration. Cell annotations show counts and percentages within each actual class.")
    picture("baseline_training_updates.png", "Figure 3. Online weight updates per epoch for all seeds. These are training update counts, not test accuracy or a batch-error curve.")
    heading("Reproduce the completed work")
    versions = example["package_versions"]
    add(f"Environment: Python {example['python_version']}; NumPy {versions['numpy']}, pandas {versions['pandas']}, scikit-learn {versions['scikit-learn']}, Matplotlib {versions['matplotlib']}. Install requirements.txt, then run <font name='Courier'>python run_pipeline.py --download</font>. Run instructions and test commands are in README.md. File hashes, source-row membership, label exclusion, preprocessing fits and saved metric/score consistency were checked.", "SmallText")
    heading("Next step when the encoder arrives")
    add("Add Siddhesh's real encoder through docs/ENCODER_HANDOFF.md. The prepared runner sweeps expansion ratios 5x, 10x and 20x with WTA sparsities 2%, 5% and 10%, repeating each setting for three seeds. A training-only validation fold selects settings; the final standard and held-out tests then compare the chosen encoder with the matching raw-feature baseline. Encoder execution and the joint comparison remain pending.", "SmallText")
    add("Data source: NSL-KDD, Kaggle mirror by Hassan: <link href='https://www.kaggle.com/datasets/hassan06/nslkdd' color='#245784'>kaggle.com/datasets/hassan06/nslkdd</link>. Original train/test files and source hashes are recorded in data/nsl_kdd_source.json. Submitted screenshots and captured run logs are preserved in report/development_evidence.md. The presentation is deferred.", "SmallText")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(42, 25, "Team sprint | Baseline complete; FlyHash integration pending")
        canvas.drawRightString(WIDTH - 42, 25, f"{document.page} / 3")
        canvas.restoreState()

    REPORT.mkdir(exist_ok=True)
    output = REPORT / "baseline_report.pdf"
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=42, rightMargin=42,
                                 topMargin=37, bottomMargin=43,
                                 title="FlyHash Network Anomaly Detector - Baseline Report",
                                 author="Siddhesh, Hansika and Lakshya")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(output)
    if len(reader.pages) != 3:
        raise ValueError(f"Expected three report pages; got {len(reader.pages)}. Adjust layout before delivery.")
    print("Built three-page report:", output)


if __name__ == "__main__":
    main()
