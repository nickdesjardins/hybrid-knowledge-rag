"""Generate the fictional demo corpus: four quality-system documents as PDFs plus corpus.json.

The documents belong to an invented company ("Aurora Nutraceuticals Inc.") and are written so
that each sample question has one unambiguous answer in one document. They are safe to send to
free model endpoints and safe to publish.

    python scripts/make_demo_corpus.py          # writes demo/*.pdf and demo/corpus.json
"""

from __future__ import annotations

import json
import pathlib

import pymupdf

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"

COMPANY = "Aurora Nutraceuticals Inc."

DOCUMENTS: list[dict] = [
    {
        "file": "SOP-QA-017-cleaning-validation.pdf",
        "title": "SOP-QA-017 Cleaning Validation of Product-Contact Equipment (Rev 4)",
        "questions": [
            "What is the residue acceptance limit for product-contact equipment?",
            "What swab recovery rate is required in cleaning validation?",
            "How many consecutive successful cleaning runs are required to validate a cleaning procedure?",
            "What is the maximum dirty hold time before cleaning must start?",
        ],
        "sections": [
            (
                "1. Purpose",
                [
                    "This Standard Operating Procedure (SOP-QA-017) defines how cleaning procedures for product-contact manufacturing equipment are validated at Aurora Nutraceuticals Inc. Cleaning validation demonstrates, with documented evidence, that a cleaning procedure consistently reduces residues of the previous product, cleaning agents and microorganisms to predetermined acceptance limits.",
                ],
            ),
            (
                "2. Scope",
                [
                    "This procedure applies to all product-contact equipment used in blending, encapsulation, tableting, coating and primary packaging at the Laval manufacturing site, including portable equipment and utensils. It does not apply to laboratory glassware, which is covered by SOP-QC-004.",
                ],
            ),
            (
                "3. Definitions",
                [
                    "Dirty hold time: the period between the end of manufacturing and the start of cleaning. Clean hold time: the period between the end of cleaning and the next use of the equipment. Worst-case product: the product that is hardest to clean because of its solubility, potency or colour, selected per the product matrix in Appendix A.",
                ],
            ),
            (
                "4. Responsibilities",
                [
                    "Production supervisors execute cleaning and complete the cleaning record before releasing equipment. Quality Control performs sampling and analysis. The Validation Lead writes the protocol and the report. The Quality Assurance Manager approves the protocol, the report and any deviation from this procedure.",
                ],
            ),
            (
                "5. Acceptance Criteria",
                [
                    "5.1 Chemical residue: the carry-over of the previous active ingredient must not exceed 10 ppm in the next product, calculated with the maximum allowable carry-over (MACO) formula in Appendix B. Where swab sampling is used, the limit is 4 micrograms per square centimetre (4 ug/cm2) of sampled surface.",
                    "5.2 Visual inspection: all surfaces must be visually clean under normal lighting, with no visible residue, discolouration or standing water.",
                    "5.3 Swab recovery: the analytical method must demonstrate a swab recovery of at least 70 percent from the equipment material of construction before validation sampling starts. Results are corrected for the measured recovery.",
                    "5.4 Microbial limits: the bioburden on sampled surfaces must not exceed 25 colony-forming units per 25 cm2 (25 CFU/25 cm2), and no objectionable organisms may be detected.",
                    "5.5 Number of runs: a cleaning procedure is validated only after three consecutive successful cleaning runs on the worst-case product meet every criterion above.",
                ],
            ),
            (
                "6. Sampling",
                [
                    "Swab samples are taken from the hardest-to-clean locations identified in the protocol, using a 25 cm2 template and the swabbing pattern in Appendix C. Rinse samples are taken from the final rinse water of closed systems. Each sample is labelled with the equipment ID, location, run number and date.",
                ],
            ),
            (
                "7. Hold Times",
                [
                    "The maximum dirty hold time is 72 hours: cleaning must begin within 72 hours of the end of manufacturing. The maximum clean hold time is 7 days: equipment not used within 7 days of cleaning must be cleaned again before use. Both hold times are challenged during validation.",
                ],
            ),
            (
                "8. Revalidation",
                [
                    "Cleaning validation must be repeated when the cleaning agent or its concentration changes, when the cleaning method changes, when a new product with a lower acceptance limit is introduced, after major equipment modification, and in any case every three years. A documented risk assessment may reduce the scope of a revalidation but never its acceptance criteria.",
                ],
            ),
            (
                "9. Records",
                [
                    "Cleaning validation protocols, raw data and reports are quality records. They are retained according to the Records Retention and Data Integrity Policy (POL-DOC-004).",
                ],
            ),
        ],
    },
    {
        "file": "SOP-QA-023-deviation-capa.pdf",
        "title": "SOP-QA-023 Deviation Management and CAPA (Rev 2)",
        "questions": [
            "Within how many hours must a deviation be reported to Quality Assurance?",
            "How many calendar days are allowed for a deviation investigation?",
            "Who approves the closure of a critical deviation?",
            "When is a CAPA effectiveness check performed?",
        ],
        "sections": [
            (
                "1. Purpose",
                [
                    "This procedure (SOP-QA-023) describes how deviations from approved procedures, specifications or validated conditions are reported, classified, investigated and closed, and how corrective and preventive actions (CAPA) are managed at Aurora Nutraceuticals Inc.",
                ],
            ),
            (
                "2. Classification",
                [
                    "Minor deviation: no impact on product quality, safety or efficacy, and no regulatory commitment affected. Major deviation: potential impact on product quality or a departure from a registered process parameter that is contained within the batch. Critical deviation: a confirmed or probable impact on product quality, safety or efficacy, a possible market action, or a breach of data integrity.",
                ],
            ),
            (
                "3. Reporting",
                [
                    "Any employee who discovers a deviation reports it to Quality Assurance within 24 hours of discovery using form F-023-01. A suspected critical deviation is reported to the Quality Assurance Manager by telephone within 4 hours of discovery and the affected material is placed on hold immediately.",
                ],
            ),
            (
                "4. Investigation",
                [
                    "4.1 Quality Assurance assigns a deviation number and an investigation owner within one working day.",
                    "4.2 The investigation, including root cause analysis, impact assessment and proposed CAPA, must be completed within 30 calendar days of the report. One extension of at most 30 additional days may be approved by the Head of Quality with a documented justification.",
                    "4.3 Root cause analysis uses the 5 Whys method for minor deviations and an Ishikawa (fishbone) diagram for major and critical deviations. 'Human error' is not an acceptable root cause without an analysis of the contributing conditions.",
                ],
            ),
            (
                "5. Approval and Closure",
                [
                    "Minor and major deviations are closed by the Quality Assurance Manager. Critical deviations are closed only by the Head of Quality, after review of the investigation report and the CAPA plan. No batch affected by an open deviation may be released; the batch disposition decision is recorded on the deviation form.",
                ],
            ),
            (
                "6. CAPA",
                [
                    "Each CAPA has an owner and a due date. Corrective actions remove the root cause; preventive actions stop recurrence elsewhere. A CAPA effectiveness check is performed 90 days after the CAPA is implemented and must show that the deviation has not recurred. An ineffective CAPA reopens the investigation.",
                ],
            ),
            (
                "7. Trending",
                [
                    "Quality Assurance trends deviations quarterly by classification, department and root cause category and presents the trend at the management review. Three deviations with the same root cause within 12 months trigger a system-level CAPA.",
                ],
            ),
        ],
    },
    {
        "file": "POL-DOC-004-records-retention.pdf",
        "title": "POL-DOC-004 Records Retention and Data Integrity Policy (Rev 3)",
        "questions": [
            "How long must batch records be retained after product expiry?",
            "Which regulation governs electronic records and audit trails?",
            "How often are electronic records backed up, and how often is a restore tested?",
            "How long are training records kept after an employee leaves?",
        ],
        "sections": [
            (
                "1. Purpose",
                [
                    "This policy (POL-DOC-004) sets the retention periods for quality records and the data integrity requirements for paper and electronic records at Aurora Nutraceuticals Inc. It applies to every department that creates or keeps GMP records.",
                ],
            ),
            (
                "2. Data Integrity Principles",
                [
                    "All records must be attributable, legible, contemporaneous, original and accurate, and in addition complete, consistent, enduring and available (ALCOA+). Corrections on paper records are made with a single line through the original entry, initialled and dated, with the reason stated. Electronic records are corrected only through the system's controlled change function.",
                ],
            ),
            (
                "3. Electronic Records and Signatures",
                [
                    "Electronic records and electronic signatures must comply with 21 CFR Part 11. Every GMP computerised system must keep a secure, computer-generated, time-stamped audit trail that records the creation, modification and deletion of records, including the identity of the user and the reason for the change. Audit trails are reviewed as part of batch record review.",
                    "Electronic records are backed up weekly to an off-site location, and a restore from backup is tested quarterly. Backup and restore results are recorded on form F-004-02.",
                ],
            ),
            (
                "4. Retention Periods",
                [
                    "4.1 Batch manufacturing and packaging records: 7 years after the expiry date of the product.",
                    "4.2 Laboratory raw data and certificates of analysis: 7 years after the expiry date of the batch to which they relate.",
                    "4.3 Cleaning validation and process validation reports: 7 years after the validated process is retired.",
                    "4.4 Training records: 10 years after the employee's employment ends.",
                    "4.5 Supplier audit reports and supplier qualification files: 10 years after the supplier is removed from the approved supplier list.",
                    "4.6 Deviation, CAPA and change control records: 10 years after closure.",
                    "4.7 Equipment logbooks and calibration records: the life of the equipment plus 5 years.",
                ],
            ),
            (
                "5. Archiving and Destruction",
                [
                    "Records are archived in a locked, fire-rated room with temperature and humidity monitoring, indexed in the archive register. Records may be destroyed only after their retention period, with the written approval of the Quality Assurance Manager, and a certificate of destruction is filed in the archive register.",
                ],
            ),
        ],
    },
    {
        "file": "SOP-SC-009-supplier-qualification.pdf",
        "title": "SOP-SC-009 Supplier Qualification and Incoming Inspection (Rev 5)",
        "questions": [
            "How often are critical suppliers audited?",
            "What sampling plan applies to incoming excipients?",
            "When is a supplier disqualified?",
            "Is identity testing required on every container of active ingredient?",
        ],
        "sections": [
            (
                "1. Purpose",
                [
                    "This procedure (SOP-SC-009) describes how suppliers of materials and services are qualified, monitored and, when necessary, disqualified, and how incoming materials are inspected and released at Aurora Nutraceuticals Inc.",
                ],
            ),
            (
                "2. Supplier Categories",
                [
                    "Critical suppliers provide active ingredients, primary packaging components in direct contact with the product, and contract laboratories. Major suppliers provide excipients, secondary packaging and calibration services. Minor suppliers provide materials and services with no impact on product quality.",
                ],
            ),
            (
                "3. Qualification",
                [
                    "A critical supplier is approved only after an on-site audit by Quality Assurance with a satisfactory report, a signed quality agreement, and three consecutive lots that pass incoming inspection. A major supplier is approved after a completed supplier questionnaire and a satisfactory review of its certifications. Minor suppliers are approved by Purchasing without a quality assessment.",
                ],
            ),
            (
                "4. Monitoring",
                [
                    "Critical suppliers are audited on site every 2 years. Major suppliers complete a requalification questionnaire every 3 years. The approved supplier list is reviewed annually by Quality Assurance and Purchasing, and the review is recorded on form F-009-03.",
                ],
            ),
            (
                "5. Incoming Inspection",
                [
                    "5.1 Every delivery is placed in quarantine on receipt and released by Quality Control only after inspection and testing.",
                    "5.2 A certificate of analysis (COA) from the supplier is required for every lot of material; a delivery without a COA is rejected.",
                    "5.3 Active ingredients: an identity test is performed on every container of every lot. Full testing is performed on each lot unless a reduced testing plan has been approved after at least 10 consecutive conforming lots.",
                    "5.4 Excipients and packaging components are sampled according to ISO 2859-1, general inspection level II, with an acceptable quality limit of 1.0 (AQL 1.0) for critical defects.",
                ],
            ),
            (
                "6. Disqualification",
                [
                    "A supplier is disqualified when two critical nonconformances are confirmed within any 12-month period, when a regulatory authority issues a warning letter concerning the supplied material, or when an audit identifies a critical finding that is not resolved within 90 days. A disqualified supplier is removed from the approved supplier list and open purchase orders are cancelled.",
                ],
            ),
        ],
    },
]

STYLE = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.4; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; }
h2 { font-size: 12pt; margin: 14pt 0 4pt 0; }
p { margin: 0 0 7pt 0; }
.meta { color: #444444; font-size: 9pt; margin-bottom: 10pt; }
"""


def build_html(doc: dict) -> str:
    parts = [
        f"<h1>{doc['title']}</h1>",
        f"<p class='meta'>{COMPANY} · Controlled document · For training and demonstration only</p>",
    ]
    for heading, paragraphs in doc["sections"]:
        parts.append(f"<h2>{heading}</h2>")
        parts.extend(f"<p>{p}</p>" for p in paragraphs)
    return "\n".join(parts)


def write_pdf(doc: dict, path: pathlib.Path) -> int:
    story = pymupdf.Story(html=build_html(doc), user_css=STYLE)
    writer = pymupdf.DocumentWriter(str(path))
    page_rect = pymupdf.paper_rect("letter")
    where = page_rect + (54, 54, -54, -60)
    pages = 0
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    return pages


def main() -> None:
    DEMO_DIR.mkdir(exist_ok=True)
    manifest = {"company": COMPANY, "documents": []}
    for doc in DOCUMENTS:
        path = DEMO_DIR / doc["file"]
        pages = write_pdf(doc, path)
        manifest["documents"].append(
            {
                "file": doc["file"],
                "title": doc["title"],
                "pages": pages,
                "questions": doc["questions"],
            }
        )
        print(f"wrote {path.name}: {pages} pages, {path.stat().st_size} bytes")
    (DEMO_DIR / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("wrote corpus.json")


if __name__ == "__main__":
    main()
