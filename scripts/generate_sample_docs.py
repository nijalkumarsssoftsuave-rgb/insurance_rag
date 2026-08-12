"""Generates the sample insurance corpus into pdf/.

These are **fictional** documents for a fictional insurer. They exist to exercise
the ingestion and retrieval pipeline against realistic structure, and to seed the
golden evaluation set.

The corpus is deliberately built to hit the features that are hard to get right:

* **Clause hierarchy** - "SECTION 4" / "4.11 Dental" nesting, which is what the
  structure-aware chunker splits on.
* **Benefit tables** - sub-limit grids that must survive chunking intact, and
  that trigger the pipeline's escalation from the fast parser to Docling.
* **Exclusions with carve-outs** - "excluded ... unless necessitated by an
  accident", the clause that must never be split from its condition.
* **Two versions of one policy with different terms** - v2.8 and v3.2 disagree on
  the PED waiting period, the room rent cap and dental cover. That is the test
  for date-of-loss filtering: a 2023 claim must be answered from v2.8, and
  answering it from v3.2 is the correctness failure in ARCHITECTURE 5.3.
* **Rejection reason codes** - the SOP maps codes to clauses, which is what lets
  the claim lane hand off to the document lane to explain a rejection.

    python scripts/generate_sample_docs.py
    python scripts/generate_sample_docs.py --out pdf --clean
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_styles = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=_styles["Heading1"], fontSize=15, spaceAfter=4, leading=19)
SUBTITLE = ParagraphStyle(
    "S",
    parent=_styles["Heading2"],
    fontSize=11.5,
    spaceAfter=10,
    textColor=colors.HexColor("#333333"),
)
H1 = ParagraphStyle(
    "H1",
    parent=_styles["Heading1"],
    fontSize=12.5,
    spaceBefore=12,
    spaceAfter=6,
    textColor=colors.HexColor("#1E4B7A"),
)
H2 = ParagraphStyle("H2", parent=_styles["Heading2"], fontSize=10.5, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle(
    "B", parent=_styles["BodyText"], fontSize=9.3, leading=12.6, spaceAfter=5, alignment=4
)
META = ParagraphStyle(
    "M",
    parent=_styles["BodyText"],
    fontSize=8.6,
    leading=11.5,
    textColor=colors.HexColor("#555555"),
    spaceAfter=3,
)

TABLE_STYLE = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CE")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EFF8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
)


@dataclass
class Doc:
    """One document, declared rather than hand-built."""

    filename: str
    title: str
    subtitle: str
    meta: list[str] = field(default_factory=list)
    body: list = field(default_factory=list)


def h1(text: str):
    return Paragraph(text, H1)


def h2(text: str):
    return Paragraph(text, H2)


def p(text: str):
    return Paragraph(text, BODY)


def table(rows: list[list[str]], widths: list[float] | None = None):
    """Tables are wrapped in KeepTogether so a grid is not split across pages -
    the same instinct the chunker applies later."""
    cols = widths or [170 / len(rows[0]) * mm] * len(rows[0])
    data = [
        [
            Paragraph(str(c), ParagraphStyle("c", parent=BODY, fontSize=8.4, spaceAfter=0))
            for c in row
        ]
        for row in rows
    ]
    return KeepTogether(Table(data, colWidths=cols, style=TABLE_STYLE, repeatRows=1))


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(
        20 * mm,
        12 * mm,
        "SAMPLE DOCUMENT - fictional insurer, generated for testing. Not a real insurance contract.",
    )
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build(doc: Doc, out_dir: Path) -> Path:
    path = out_dir / doc.filename
    story = [Paragraph(doc.title, TITLE), Paragraph(doc.subtitle, SUBTITLE)]
    story += [Paragraph(m, META) for m in doc.meta]
    story.append(Spacer(1, 6 * mm))
    story += doc.body

    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title=doc.title,
        author="Acme General Insurance (sample)",
    ).build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


# ══════════════════════════════════════════════════ shared clause text


DEFINITIONS = [
    (
        "1.1 Accident",
        "An accident means a sudden, unforeseen and involuntary event caused by external, "
        "visible and violent means.",
    ),
    (
        "1.2 Cashless Facility",
        "Cashless facility means a facility under which the Company pays the cost of treatment "
        "directly to the Network Provider, to the extent admissible under the Policy, so that the "
        "Insured Person need not pay at the time of discharge.",
    ),
    (
        "1.3 Hospital",
        "A Hospital means any institution established for in-patient care which has at least ten "
        "in-patient beds in towns with a population below ten lakhs and fifteen in-patient beds "
        "elsewhere, qualified nursing staff round the clock, a fully equipped operation theatre, "
        "and maintains daily records of patients accessible to the Company.",
    ),
    (
        "1.4 Pre-existing Disease",
        "A Pre-existing Disease means any condition, ailment, injury or disease that is diagnosed "
        "by a physician, or for which medical advice or treatment was recommended by or received "
        "from a physician, within forty-eight months prior to the effective date of the first "
        "policy issued by the Company to the Insured Person.",
    ),
    (
        "1.5 Sum Insured",
        "Sum Insured means the pre-defined limit specified in the Policy Schedule, representing "
        "the maximum liability of the Company for any and all claims made during the Policy Year.",
    ),
    (
        "1.6 Network Provider",
        "Network Provider means a hospital or health care provider enrolled by the Company or its "
        "Third Party Administrator to provide medical services on a cashless basis.",
    ),
    (
        "1.7 Day Care Treatment",
        "Day Care Treatment means medical or surgical procedures undertaken under general or local "
        "anaesthesia in a hospital or day care centre in less than twenty-four consecutive hours "
        "because of technological advancement.",
    ),
    (
        "1.8 Co-payment",
        "Co-payment means a cost sharing requirement under which the Insured Person bears a "
        "specified percentage of each admissible claim. A co-payment does not reduce the Sum "
        "Insured.",
    ),
]


def definitions_section() -> list:
    out = [
        h1("SECTION 1 - DEFINITIONS"),
        p(
            "The following words and expressions have the meanings assigned to them below "
            "wherever they appear in this Policy."
        ),
    ]
    for heading, text in DEFINITIONS:
        out += [h2(heading), p(text)]
    return out


CLAIMS_PROCEDURE = [
    (
        "6.1 Intimation of Claim",
        "The Insured Person must notify the Company or the Third Party Administrator of any claim "
        "within twenty-four hours of an emergency hospitalisation, and at least forty-eight hours "
        "prior to a planned hospitalisation.",
    ),
    (
        "6.2 Cashless Facility",
        "For cashless treatment the Insured Person must obtain pre-authorisation from the Third "
        "Party Administrator at least forty-eight hours prior to a planned hospitalisation at a "
        "Network Provider. In case of emergency admission, intimation must be given within "
        "twenty-four hours of admission. Pre-authorisation is granted on the basis of the "
        "information furnished and does not constitute an admission of liability.",
    ),
    (
        "6.3 Reimbursement Claims",
        "Where treatment is taken at a hospital that is not a Network Provider, the Insured Person "
        "must pay the hospital and submit the claim for reimbursement. All original documents, "
        "including the discharge summary, itemised bills, payment receipts, investigation reports "
        "and the treating doctor's prescriptions, must reach the Company within fifteen days of "
        "discharge.",
    ),
    (
        "6.4 Settlement Timelines",
        "The Company shall settle or reject a claim within thirty days of receipt of the last "
        "necessary document. Where an investigation is warranted, it shall be completed within "
        "thirty days and the claim settled within forty-five days of receipt of the last necessary "
        "document.",
    ),
    (
        "6.5 Repudiation",
        "Where a claim is repudiated, the Company shall communicate in writing the grounds for "
        "repudiation with reference to the specific clause of the Policy relied upon, and inform "
        "the Insured Person of the grievance redressal procedure.",
    ),
]


def claims_section() -> list:
    out = [h1("SECTION 6 - CLAIMS PROCEDURE")]
    for heading, text in CLAIMS_PROCEDURE:
        out += [h2(heading), p(text)]
    return out


GENERAL_CONDITIONS = [
    (
        "7.1 Free Look Period",
        "The Insured Person has a period of fifteen days from the date of receipt of the Policy to "
        "review its terms and, if not acceptable, to cancel the Policy and receive a refund of "
        "premium subject to deduction of proportionate risk premium and expenses incurred on "
        "medical examination and stamp duty.",
    ),
    (
        "7.2 Renewal",
        "The Policy shall ordinarily be renewable for life, subject to payment of premium in "
        "advance. The Company shall not deny renewal on the ground of an adverse claims experience "
        "except in the case of established fraud or non-disclosure of material facts.",
    ),
    (
        "7.3 Portability",
        "The Insured Person may port the Policy to another insurer at the time of renewal, and "
        "shall be given credit for the waiting periods already served, subject to an application "
        "made at least forty-five days before the renewal date.",
    ),
    (
        "7.4 Grievance Redressal",
        "Any grievance may be addressed to the Grievance Officer of the Company. If the Insured "
        "Person is not satisfied with the resolution within fifteen days, the matter may be "
        "escalated to the Insurance Ombudsman having jurisdiction.",
    ),
    (
        "7.5 Fraudulent Claims",
        "If any claim is fraudulent, or if any fraudulent means are used to obtain a benefit, all "
        "benefits under this Policy shall be forfeited and the Policy may be cancelled ab initio.",
    ),
]


def conditions_section() -> list:
    out = [h1("SECTION 7 - GENERAL CONDITIONS")]
    for heading, text in GENERAL_CONDITIONS:
        out += [h2(heading), p(text)]
    return out


# ══════════════════════════════════════════════ 1. health wording v3.2


def family_health_v32() -> Doc:
    body = [
        h1("PREAMBLE"),
        p(
            "This Policy is a contract of insurance issued by Acme General Insurance Company "
            "Limited to the Policyholder named in the Schedule, in consideration of the premium "
            "received and on the basis of the statements made in the proposal form. This Policy "
            "records the entire agreement between the parties."
        ),
        *definitions_section(),
        h1("SECTION 2 - SCOPE OF COVER"),
        h2("2.1 In-patient Treatment"),
        p(
            "The Company shall indemnify medical expenses incurred by the Insured Person for "
            "hospitalisation of more than twenty-four consecutive hours during the Policy Period "
            "arising from illness or injury, up to the Sum Insured specified in the Schedule."
        ),
        h2("2.2 Pre-hospitalisation Expenses"),
        p(
            "Medical expenses incurred during the sixty days immediately preceding the date of "
            "admission are payable, provided the claim for in-patient treatment is admissible."
        ),
        h2("2.3 Post-hospitalisation Expenses"),
        p(
            "Medical expenses incurred during the ninety days immediately following the date of "
            "discharge are payable, provided the claim for in-patient treatment is admissible."
        ),
        h2("2.4 Day Care Treatment"),
        p(
            "Expenses for day care treatment taken in a hospital or day care centre are payable "
            "notwithstanding that the hospitalisation is for less than twenty-four hours."
        ),
        h2("2.5 Road Ambulance"),
        p(
            "Expenses incurred on transportation of the Insured Person by road ambulance to a "
            "hospital are payable up to two thousand rupees per hospitalisation."
        ),
        h2("2.6 AYUSH Treatment"),
        p(
            "Expenses for in-patient treatment taken under Ayurveda, Yoga, Naturopathy, Unani, "
            "Siddha and Homeopathy systems of medicine in a recognised hospital are payable up to "
            "the Sum Insured. This benefit was introduced with effect from 1 April 2026."
        ),
        h1("SECTION 3 - BENEFITS AND SUB-LIMITS"),
        p(
            "The following sub-limits apply to the benefits listed. All amounts are per Policy "
            "Year unless stated otherwise. Where the Insured Person occupies accommodation whose "
            "rent exceeds the eligible limit, all associated medical expenses shall be "
            "proportionately reduced in the same ratio, except in respect of intensive care."
        ),
        Spacer(1, 3 * mm),
        table(
            [
                ["Benefit", "Sub-limit", "Applies from"],
                [
                    "Room rent, boarding and nursing",
                    "1% of Sum Insured per day, maximum Rs. 5,000 per day",
                    "Day 31",
                ],
                [
                    "Intensive Care Unit charges",
                    "2% of Sum Insured per day, maximum Rs. 10,000 per day",
                    "Day 31",
                ],
                ["Cataract surgery", "Rs. 40,000 per eye", "After 24 months"],
                ["Maternity expenses - normal delivery", "Rs. 50,000", "After 36 months"],
                ["Maternity expenses - caesarean section", "Rs. 75,000", "After 36 months"],
                ["Modern treatment procedures", "50% of Sum Insured", "After 24 months"],
                ["Road ambulance", "Rs. 2,000 per hospitalisation", "Immediate"],
                ["AYUSH in-patient treatment", "Up to Sum Insured", "Day 31"],
            ],
            widths=[62 * mm, 68 * mm, 30 * mm],
        ),
        Spacer(1, 4 * mm),
        h2("3.1 Co-payment"),
        p(
            "A co-payment of ten per cent shall apply to each and every admissible claim where the "
            "Insured Person is aged sixty-one years or above at the time of first enrolment. No "
            "co-payment applies to Insured Persons enrolled before the age of sixty-one."
        ),
        PageBreak(),
        h1("SECTION 4 - EXCLUSIONS"),
        p(
            "The Company shall not be liable to make any payment under this Policy in respect of "
            "any expenses incurred in connection with or in respect of the following."
        ),
        h2("4.1 Investigation and Evaluation"),
        p(
            "Expenses related to any admission primarily for diagnostics and evaluation purposes, "
            "and any diagnostic expenses not related to a current diagnosis or treatment, are "
            "excluded."
        ),
        h2("4.2 Rest Cure and Rehabilitation"),
        p(
            "Expenses related to any admission primarily for enforced bed rest, custodial care, or "
            "services for people who are terminally ill to address their physical, social, "
            "emotional and spiritual needs, are excluded."
        ),
        h2("4.11 Dental Treatment"),
        p(
            "Dental treatment, dental surgery and orthodontic procedures of any kind are excluded "
            "from the scope of this Policy. This exclusion shall not apply where such treatment is "
            "necessitated by an accident and requires hospitalisation for at least twenty-four "
            "consecutive hours. Routine dental check-ups, scaling, polishing and cosmetic "
            "dentistry are never payable under any circumstances. Where the exception applies, the "
            "Insured Person must submit the treating dentist's report together with evidence of "
            "the accident within thirty days of discharge."
        ),
        h2("4.12 Cosmetic and Aesthetic Treatment"),
        p(
            "Expenses for cosmetic or plastic surgery are excluded unless required as part of "
            "medically necessary treatment to remove a direct consequence of an accident, burn "
            "injury or cancer, and certified as such by the attending medical practitioner."
        ),
        h2("4.13 Obesity and Weight Control"),
        p(
            "Expenses related to the surgical treatment of obesity are excluded, unless the "
            "surgery is advised by the treating medical practitioner, the Insured Person has a "
            "Body Mass Index of forty or above, or thirty-five or above with an associated "
            "co-morbidity, and the condition has not responded to at least three months of "
            "documented conservative management."
        ),
        h2("4.14 Infertility and Sterility"),
        p(
            "Expenses related to sterility and infertility, including assisted reproduction "
            "services, gestational surrogacy and reversal of sterilisation, are excluded."
        ),
        h2("4.15 Hazardous Activities"),
        p(
            "Expenses related to any treatment necessitated due to participation as a professional "
            "in hazardous or adventure sports, including para-jumping, rock climbing, mountaineering, "
            "rafting, motor racing, horse racing and scuba diving, are excluded."
        ),
        h2("4.16 Breach of Law"),
        p(
            "Expenses for treatment directly arising from or consequent upon any Insured Person "
            "committing or attempting to commit a breach of law with criminal intent are excluded."
        ),
        h1("SECTION 5 - WAITING PERIODS"),
        p(
            "The following waiting periods apply from the date of commencement of the first Policy "
            "with the Company, and are not reset on renewal without a break."
        ),
        Spacer(1, 3 * mm),
        table(
            [
                ["Waiting period", "Duration", "Applies to"],
                ["Initial waiting period", "30 days", "All illnesses except accidental injury"],
                ["Pre-existing diseases", "36 months", "Any declared pre-existing condition"],
                [
                    "Specified diseases and procedures",
                    "24 months",
                    "Cataract, hernia, hysterectomy, joint replacement",
                ],
                ["Maternity benefit", "36 months", "Delivery and related expenses"],
                ["Bariatric surgery", "36 months", "Surgical treatment of obesity"],
            ],
            widths=[48 * mm, 26 * mm, 86 * mm],
        ),
        Spacer(1, 4 * mm),
        h2("5.1 Pre-existing Diseases"),
        p(
            "Any pre-existing disease and its direct complications shall be excluded until the "
            "expiry of thirty-six months of continuous coverage from the first policy inception "
            "date with the Company. This waiting period was reduced from forty-eight months to "
            "thirty-six months with effect from 1 April 2026 and applies to all policies incepted "
            "or renewed on or after that date."
        ),
        h2("5.2 Specified Diseases and Procedures"),
        p(
            "A waiting period of twenty-four months applies to cataract, benign prostatic "
            "hypertrophy, hernia of all types, hysterectomy, joint replacement surgery, and "
            "treatment for stones in the urinary and biliary systems."
        ),
        h2("5.3 Accidental Injury"),
        p(
            "No waiting period applies to hospitalisation necessitated by an accident occurring "
            "after the commencement of the Policy."
        ),
        *claims_section(),
        *conditions_section(),
    ]
    return Doc(
        filename="acme_family_health_optima_wording_v3.2.pdf",
        title="ACME GENERAL INSURANCE COMPANY LIMITED",
        subtitle="Family Health Optima Insurance Plan - Policy Wording",
        meta=[
            "UIN: ACMEHLIP26032V032627 &nbsp;|&nbsp; Version 3.2 &nbsp;|&nbsp; Document type: Policy Wording",
            "Effective from 01/04/2026 to 31/03/2027 &nbsp;|&nbsp; Supersedes Version 2.8",
            "Registered Office: 14 Marine Lines, Mumbai 400020 &nbsp;|&nbsp; IRDAI Registration No. 999 (sample)",
        ],
        body=body,
    )


# ═══════════════════════════════════ 2. health wording v2.8 (superseded)


def family_health_v28() -> Doc:
    """The superseded wording.

    Deliberately disagrees with v3.2 on three material terms - PED waiting
    period, room rent cap, and whether accidental dental is covered. A claim with
    a 2023 date of loss must be answered from THIS document.
    """
    body = [
        h1("PREAMBLE"),
        p(
            "This Policy is a contract of insurance issued by Acme General Insurance Company "
            "Limited to the Policyholder named in the Schedule, in consideration of the premium "
            "received and on the basis of the statements made in the proposal form."
        ),
        *definitions_section(),
        h1("SECTION 2 - SCOPE OF COVER"),
        h2("2.1 In-patient Treatment"),
        p(
            "The Company shall indemnify medical expenses incurred by the Insured Person for "
            "hospitalisation of more than twenty-four consecutive hours during the Policy Period "
            "arising from illness or injury, up to the Sum Insured specified in the Schedule."
        ),
        h2("2.2 Pre-hospitalisation Expenses"),
        p(
            "Medical expenses incurred during the thirty days immediately preceding the date of "
            "admission are payable, provided the claim for in-patient treatment is admissible."
        ),
        h2("2.3 Post-hospitalisation Expenses"),
        p(
            "Medical expenses incurred during the sixty days immediately following the date of "
            "discharge are payable, provided the claim for in-patient treatment is admissible."
        ),
        h2("2.4 Day Care Treatment"),
        p(
            "Expenses for day care treatment taken in a hospital or day care centre are payable "
            "notwithstanding that the hospitalisation is for less than twenty-four hours."
        ),
        h2("2.5 Road Ambulance"),
        p(
            "Expenses incurred on transportation of the Insured Person by road ambulance to a "
            "hospital are payable up to one thousand five hundred rupees per hospitalisation."
        ),
        h2("2.6 AYUSH Treatment"),
        p(
            "Treatment taken under Ayurveda, Yoga, Naturopathy, Unani, Siddha and Homeopathy "
            "systems of medicine is not covered under this Policy."
        ),
        h1("SECTION 3 - BENEFITS AND SUB-LIMITS"),
        p(
            "The following sub-limits apply to the benefits listed. All amounts are per Policy "
            "Year unless stated otherwise."
        ),
        Spacer(1, 3 * mm),
        table(
            [
                ["Benefit", "Sub-limit", "Applies from"],
                [
                    "Room rent, boarding and nursing",
                    "1% of Sum Insured per day, maximum Rs. 4,000 per day",
                    "Day 31",
                ],
                [
                    "Intensive Care Unit charges",
                    "2% of Sum Insured per day, maximum Rs. 8,000 per day",
                    "Day 31",
                ],
                ["Cataract surgery", "Rs. 30,000 per eye", "After 24 months"],
                ["Maternity expenses - normal delivery", "Rs. 35,000", "After 48 months"],
                ["Maternity expenses - caesarean section", "Rs. 50,000", "After 48 months"],
                ["Road ambulance", "Rs. 1,500 per hospitalisation", "Immediate"],
            ],
            widths=[62 * mm, 68 * mm, 30 * mm],
        ),
        Spacer(1, 4 * mm),
        h2("3.1 Co-payment"),
        p(
            "A co-payment of twenty per cent shall apply to each and every admissible claim where "
            "the Insured Person is aged sixty-one years or above at the time of first enrolment."
        ),
        PageBreak(),
        h1("SECTION 4 - EXCLUSIONS"),
        p(
            "The Company shall not be liable to make any payment under this Policy in respect of "
            "any expenses incurred in connection with or in respect of the following."
        ),
        h2("4.1 Investigation and Evaluation"),
        p(
            "Expenses related to any admission primarily for diagnostics and evaluation purposes "
            "are excluded."
        ),
        h2("4.11 Dental Treatment"),
        p(
            "Dental treatment, dental surgery and orthodontic procedures of any kind are excluded "
            "from the scope of this Policy in all circumstances, whether or not necessitated by an "
            "accident and whether or not requiring hospitalisation."
        ),
        h2("4.12 Cosmetic and Aesthetic Treatment"),
        p(
            "Expenses for cosmetic or plastic surgery are excluded unless required as part of "
            "medically necessary treatment to remove a direct consequence of an accident or burn "
            "injury."
        ),
        h2("4.13 Obesity and Weight Control"),
        p(
            "Expenses related to the surgical treatment of obesity are excluded in all "
            "circumstances under this Policy."
        ),
        h2("4.15 Hazardous Activities"),
        p(
            "Expenses related to any treatment necessitated due to participation in hazardous or "
            "adventure sports are excluded."
        ),
        h1("SECTION 5 - WAITING PERIODS"),
        Spacer(1, 3 * mm),
        table(
            [
                ["Waiting period", "Duration", "Applies to"],
                ["Initial waiting period", "30 days", "All illnesses except accidental injury"],
                ["Pre-existing diseases", "48 months", "Any declared pre-existing condition"],
                [
                    "Specified diseases and procedures",
                    "24 months",
                    "Cataract, hernia, hysterectomy, joint replacement",
                ],
                ["Maternity benefit", "48 months", "Delivery and related expenses"],
            ],
            widths=[48 * mm, 26 * mm, 86 * mm],
        ),
        Spacer(1, 4 * mm),
        h2("5.1 Pre-existing Diseases"),
        p(
            "Any pre-existing disease and its direct complications shall be excluded until the "
            "expiry of forty-eight months of continuous coverage from the first policy inception "
            "date with the Company."
        ),
        h2("5.2 Specified Diseases and Procedures"),
        p(
            "A waiting period of twenty-four months applies to cataract, benign prostatic "
            "hypertrophy, hernia of all types, hysterectomy and joint replacement surgery."
        ),
        *claims_section(),
        *conditions_section(),
    ]
    return Doc(
        filename="acme_family_health_optima_wording_v2.8_superseded.pdf",
        title="ACME GENERAL INSURANCE COMPANY LIMITED",
        subtitle="Family Health Optima Insurance Plan - Policy Wording (SUPERSEDED)",
        meta=[
            "UIN: ACMEHLIP24028V022426 &nbsp;|&nbsp; Version 2.8 &nbsp;|&nbsp; Document type: Policy Wording",
            "Effective from 01/04/2024 to 31/03/2026 &nbsp;|&nbsp; Superseded by Version 3.2",
            "Retained for claims with a date of loss falling within the above period.",
        ],
        body=body,
    )


# ═════════════════════════════════════════════════════ 3. motor wording


def motor_shield() -> Doc:
    body = [
        h1("PREAMBLE"),
        p(
            "This Policy witnesses that, in consideration of the premium paid, Acme General "
            "Insurance Company Limited will indemnify the Insured against loss or damage to the "
            "vehicle described in the Schedule, and against liability to third parties, during the "
            "Policy Period."
        ),
        h1("SECTION 1 - DEFINITIONS"),
        h2("1.1 Insured Declared Value"),
        p(
            "The Insured Declared Value is the sum arrived at by applying the depreciation "
            "schedule to the manufacturer's listed selling price of the vehicle, and represents "
            "the maximum liability of the Company in the event of total loss or theft."
        ),
        h2("1.2 No Claim Bonus"),
        p(
            "No Claim Bonus means the discount on the own damage premium allowed at renewal in "
            "recognition of a claim-free Policy Period."
        ),
        h2("1.3 Total Loss"),
        p(
            "A vehicle is treated as a total loss where the aggregate cost of retrieval and repair "
            "exceeds seventy-five per cent of the Insured Declared Value."
        ),
        h1("SECTION 2 - OWN DAMAGE"),
        h2("2.1 Scope of Cover"),
        p(
            "The Company shall indemnify the Insured against loss of or damage to the vehicle and "
            "its accessories caused by accidental external means, fire, explosion, self-ignition, "
            "lightning, burglary, housebreaking, theft, riot, strike, malicious act, terrorist "
            "activity, earthquake, flood, storm, and by accident in direct transit by road, rail, "
            "inland waterway, lift, elevator or air."
        ),
        h2("2.2 Depreciation on Parts"),
        p(
            "In the event of a partial loss, depreciation shall be deducted on the cost of "
            "replacement of parts in accordance with the schedule set out in clause 2.3, "
            "irrespective of the age of the vehicle."
        ),
        h2("2.3 Depreciation Schedule"),
        Spacer(1, 2 * mm),
        table(
            [
                ["Part category", "Rate of depreciation"],
                ["Rubber, nylon and plastic parts, tyres, tubes, batteries", "50%"],
                ["Fibre glass components", "30%"],
                ["Parts made of glass", "Nil"],
                ["All other parts - vehicle up to 6 months old", "Nil"],
                ["All other parts - vehicle 6 months to 1 year", "5%"],
                ["All other parts - vehicle 1 to 2 years", "10%"],
                ["All other parts - vehicle 2 to 3 years", "15%"],
                ["All other parts - vehicle exceeding 5 years", "50%"],
            ],
            widths=[110 * mm, 50 * mm],
        ),
        h1("SECTION 3 - LIABILITY TO THIRD PARTIES"),
        h2("3.1 Scope"),
        p(
            "The Company shall indemnify the Insured against all sums which the Insured becomes "
            "legally liable to pay in respect of the death of or bodily injury to any person, and "
            "damage to property belonging to a third party, caused by or arising out of the use of "
            "the vehicle."
        ),
        h2("3.2 Limits"),
        p(
            "Liability in respect of death or bodily injury is unlimited as required by law. "
            "Liability in respect of damage to third party property is limited to seven lakh fifty "
            "thousand rupees."
        ),
        h1("SECTION 4 - NO CLAIM BONUS"),
        Spacer(1, 2 * mm),
        table(
            [
                ["Claim-free Policy Periods", "Discount on own damage premium"],
                ["1 claim-free year", "20%"],
                ["2 consecutive claim-free years", "25%"],
                ["3 consecutive claim-free years", "35%"],
                ["4 consecutive claim-free years", "45%"],
                ["5 consecutive claim-free years", "50%"],
            ],
            widths=[95 * mm, 65 * mm],
        ),
        Spacer(1, 3 * mm),
        p(
            "No Claim Bonus is lost if a claim is made during the Policy Period, and is forfeited "
            "if the Policy is not renewed within ninety days of expiry."
        ),
        h1("SECTION 7 - EXCLUSIONS"),
        h2("7.1 Consequential Loss and Wear"),
        p(
            "Consequential loss, depreciation, wear and tear, mechanical or electrical breakdown, "
            "failures or breakages are excluded. Damage to tyres and tubes is excluded unless the "
            "vehicle is damaged at the same time, in which case liability is limited to fifty per "
            "cent of the cost of replacement."
        ),
        h2("7.2 Driving Without a Valid Licence"),
        p(
            "Loss or damage caused while the vehicle is being driven by any person who is not "
            "holding an effective driving licence, or who is disqualified from holding one, is "
            "excluded."
        ),
        h2("7.3 Driving Under the Influence"),
        p(
            "Loss or damage caused while the vehicle is being driven by a person under the "
            "influence of intoxicating liquor or drugs is excluded in all circumstances."
        ),
        h2("7.4 Use Contrary to Limitations"),
        p(
            "Loss or damage arising while the vehicle is used otherwise than in accordance with "
            "the Limitations as to Use stated in the Schedule, including use for hire or reward, "
            "carriage of goods other than samples, organised racing, pace making, speed testing or "
            "reliability trials, is excluded."
        ),
        h1("SECTION 8 - CLAIMS PROCEDURE"),
        h2("8.1 Intimation"),
        p(
            "Notice of any accident, loss or damage must be given to the Company in writing "
            "immediately and in any event within forty-eight hours of the occurrence. In the event "
            "of theft, a First Information Report must be lodged with the police without delay."
        ),
        h2("8.2 Cashless Repair"),
        p(
            "Where the vehicle is repaired at a garage in the Company's network, the Company shall "
            "settle the admissible amount directly with the garage, and the Insured shall bear the "
            "compulsory deductible, depreciation and any inadmissible items."
        ),
        h2("8.3 Compulsory Deductible"),
        p(
            "A compulsory deductible of one thousand rupees applies to each and every own damage "
            "claim for private cars exceeding 1500 cc, and of two thousand rupees for vehicles "
            "below that engine capacity."
        ),
    ]
    return Doc(
        filename="acme_motor_shield_private_car_wording_v1.4.pdf",
        title="ACME GENERAL INSURANCE COMPANY LIMITED",
        subtitle="Motor Shield Private Car Package Policy - Policy Wording",
        meta=[
            "UIN: ACMEMOTP26014V011627 &nbsp;|&nbsp; Version 1.4 &nbsp;|&nbsp; Document type: Policy Wording",
            "Effective from 01/04/2026 to 31/03/2027",
        ],
        body=body,
    )


# ══════════════════════════════════════════════════════ 4. endorsement


def endorsement() -> Doc:
    body = [
        p(
            "This Endorsement forms part of and is attached to the Family Health Optima Insurance "
            "Plan Policy Wording, Version 3.2. All other terms, conditions, exclusions and "
            "limitations of the Policy remain unaltered."
        ),
        h1("SECTION 1 - AMENDMENTS TO BENEFITS"),
        h2("1.1 Enhancement of Maternity Benefit"),
        p(
            "With effect from 1 October 2026, the sub-limit for maternity expenses set out in "
            "Section 3 of the Policy Wording is enhanced as follows. All other conditions "
            "governing the maternity benefit, including the waiting period of thirty-six months, "
            "remain unchanged."
        ),
        Spacer(1, 2 * mm),
        table(
            [
                ["Benefit", "Existing sub-limit", "Revised sub-limit"],
                ["Maternity - normal delivery", "Rs. 50,000", "Rs. 75,000"],
                ["Maternity - caesarean section", "Rs. 75,000", "Rs. 1,00,000"],
                [
                    "Pre and post natal expenses",
                    "Not covered",
                    "Rs. 10,000 within the maternity limit",
                ],
            ],
            widths=[55 * mm, 50 * mm, 55 * mm],
        ),
        h2("1.2 New Born Baby Cover"),
        p(
            "A new born baby shall be covered from day one of birth up to the maternity sub-limit "
            "applicable, for expenses incurred towards medically necessary treatment of the baby "
            "while the mother is hospitalised. The baby must be added to the Policy within "
            "ninety days of birth for cover to continue beyond the mother's discharge."
        ),
        h1("SECTION 2 - AMENDMENTS TO EXCLUSIONS"),
        h2("2.1 Modification of Clause 4.13"),
        p(
            "Clause 4.13 of the Policy Wording relating to obesity and weight control is amended "
            "by the substitution of a Body Mass Index threshold of thirty-seven for the existing "
            "threshold of forty, where the Insured Person has an associated diagnosis of type 2 "
            "diabetes mellitus."
        ),
        h1("SECTION 3 - PREMIUM"),
        p(
            "An additional premium of one thousand two hundred rupees per insured person per annum "
            "is payable in respect of the enhancements granted by this Endorsement. The additional "
            "premium has been collected and is acknowledged."
        ),
    ]
    return Doc(
        filename="acme_endorsement_2026_1187_maternity_enhancement.pdf",
        title="ACME GENERAL INSURANCE COMPANY LIMITED",
        subtitle="Endorsement No. AGI/END/2026/1187 - Maternity Benefit Enhancement",
        meta=[
            "Attached to: Family Health Optima Policy Wording Version 3.2 (UIN ACMEHLIP26032V032627)",
            "Document type: Endorsement &nbsp;|&nbsp; Effective from 01/10/2026 to 31/03/2027",
        ],
        body=body,
    )


# ══════════════════════════════════════════════════════════ 5. claims SOP


def claims_sop() -> Doc:
    body = [
        p(
            "This Standard Operating Procedure governs the handling of cashless pre-authorisation "
            "requests and reimbursement claims by claims processing staff and the Third Party "
            "Administrator. It is an internal document and does not confer any rights on "
            "policyholders."
        ),
        h1("SECTION 1 - CASHLESS PRE-AUTHORISATION"),
        h2("1.1 Receipt and Acknowledgement"),
        p(
            "Every pre-authorisation request received from a Network Provider must be acknowledged "
            "within thirty minutes of receipt. The acknowledgement must carry a unique "
            "pre-authorisation reference number in the format PA-YYYY-NNNNNN."
        ),
        h2("1.2 Decision Timelines"),
        p(
            "A decision on a planned pre-authorisation request must be communicated within four "
            "hours of receipt of a complete request. For emergency admissions the decision must be "
            "communicated within one hour. Where a decision cannot be taken within these "
            "timelines, an interim response stating the reason and the expected decision time must "
            "be issued."
        ),
        h2("1.3 Query Management"),
        p(
            "Where information is insufficient, a single consolidated query must be raised. "
            "Serial queries on the same request are not permitted. If the response to a query is "
            "not received within forty-eight hours, the request may be closed as incomplete, and "
            "the Insured Person must be informed that a reimbursement claim may still be filed."
        ),
        h2("1.4 Enhancement Requests"),
        p(
            "Requests to enhance an approved pre-authorisation amount must be decided within two "
            "hours. Enhancement is permitted only up to the balance Sum Insured after taking into "
            "account applicable sub-limits and co-payment."
        ),
        h1("SECTION 2 - REJECTION REASON CODES"),
        p(
            "Every rejection or partial approval must carry one of the following codes together "
            "with the specific Policy clause relied upon. Free-text rejection reasons are not "
            "permitted, and a rejection recorded without a clause reference must be returned to "
            "the assessor."
        ),
        Spacer(1, 2 * mm),
        table(
            [
                ["Code", "Description", "Policy clause"],
                [
                    "PED_WAITING_PERIOD",
                    "Condition is pre-existing and the waiting period is not complete",
                    "Clause 5.1",
                ],
                [
                    "SPECIFIED_DISEASE_WAIT",
                    "Specified disease within the 24 month waiting period",
                    "Clause 5.2",
                ],
                [
                    "INITIAL_WAITING_PERIOD",
                    "Claim within the first 30 days and not accidental",
                    "Section 5",
                ],
                ["EXCL_DENTAL", "Dental treatment not necessitated by an accident", "Clause 4.11"],
                ["EXCL_COSMETIC", "Cosmetic or aesthetic treatment", "Clause 4.12"],
                ["EXCL_OBESITY", "Obesity surgery, BMI criteria not met", "Clause 4.13"],
                ["EXCL_INFERTILITY", "Infertility or assisted reproduction", "Clause 4.14"],
                ["EXCL_HAZARDOUS", "Injury from professional hazardous sport", "Clause 4.15"],
                [
                    "ROOM_RENT_LIMIT",
                    "Room rent exceeds eligibility, proportionate deduction applied",
                    "Section 3",
                ],
                [
                    "SUBLIMIT_EXHAUSTED",
                    "Benefit sub-limit already utilised for the Policy Year",
                    "Section 3",
                ],
                ["NON_DISCLOSURE", "Material fact not disclosed in the proposal", "Clause 7.5"],
                [
                    "DOC_INSUFFICIENT",
                    "Required documents not received within the stipulated period",
                    "Clause 6.3",
                ],
            ],
            widths=[48 * mm, 82 * mm, 30 * mm],
        ),
        PageBreak(),
        h1("SECTION 3 - REIMBURSEMENT CLAIMS"),
        h2("3.1 Document Checklist"),
        p(
            "A reimbursement claim is complete only when the claim form duly signed, the original "
            "discharge summary, itemised hospital bill, payment receipts, investigation reports, "
            "prescriptions, and the identity proof of the Insured Person have all been received. "
            "Any deficiency must be communicated in a single consolidated letter within seven days "
            "of receipt."
        ),
        h2("3.2 Assessment"),
        p(
            "The assessor must verify the admissibility of the claim against the Policy Wording in "
            "force on the date of admission, and not the wording current at the time of "
            "assessment. Where the Policy has been renewed with a change in terms, the wording "
            "applicable to the date of loss governs."
        ),
        h2("3.3 Settlement"),
        p(
            "An admissible claim must be settled within thirty days of receipt of the last "
            "necessary document. Where settlement is delayed beyond this period for reasons "
            "attributable to the Company, interest at the bank rate plus two per cent is payable "
            "for the period of delay."
        ),
        h1("SECTION 4 - ESCALATION MATRIX"),
        Spacer(1, 2 * mm),
        table(
            [
                ["Level", "Owner", "Trigger", "Response time"],
                ["L1", "Claims Associate", "Standard claim within limits", "30 days"],
                ["L2", "Claims Manager", "Claim value above Rs. 5,00,000", "15 days"],
                ["L3", "Medical Review Board", "Disputed medical necessity", "7 days"],
                ["L4", "Grievance Officer", "Policyholder complaint", "15 days"],
                ["L5", "Insurance Ombudsman", "Unresolved after L4", "As per statute"],
            ],
            widths=[16 * mm, 42 * mm, 68 * mm, 34 * mm],
        ),
        h1("SECTION 5 - QUALITY CONTROL"),
        h2("5.1 Sampling"),
        p(
            "A minimum of five per cent of all settled claims and one hundred per cent of all "
            "repudiated claims must be reviewed by the quality team each month."
        ),
        h2("5.2 Repudiation Review"),
        p(
            "No claim may be repudiated without a second-level review. The reviewer must confirm "
            "that the rejection code, the clause reference and the communication to the Insured "
            "Person are consistent with one another."
        ),
    ]
    return Doc(
        filename="acme_claims_sop_cashless_and_reimbursement.pdf",
        title="ACME GENERAL INSURANCE COMPANY LIMITED",
        subtitle="Standard Operating Procedure - Health Claims Processing",
        meta=[
            "Document type: SOP &nbsp;|&nbsp; Reference: AGI/SOP/CLAIMS/2026/04 &nbsp;|&nbsp; Version 4.1",
            "Effective from 01/04/2026 &nbsp;|&nbsp; Internal use - claims operations and TPA",
        ],
        body=body,
    )


# ═══════════════════════════════════════════════════════════ 6. circular


def circular() -> Doc:
    body = [
        p(
            "This Circular is issued to all claims processing offices, branch offices and the "
            "empanelled Third Party Administrator. It takes effect from 1 July 2026 and must be "
            "read together with the Standard Operating Procedure AGI/SOP/CLAIMS/2024/04."
        ),
        h1("SECTION 1 - REVISED TURNAROUND TIMES"),
        h2("1.1 Background"),
        p(
            "A review of claims handling during the preceding financial year identified delays "
            "concentrated in the pre-authorisation query cycle and in the collection of "
            "deficiency documents for reimbursement claims. The following turnaround times are "
            "revised with immediate effect."
        ),
        h2("1.2 Revised Timelines"),
        Spacer(1, 2 * mm),
        table(
            [
                ["Activity", "Previous", "Revised"],
                ["Pre-authorisation - planned admission", "6 hours", "4 hours"],
                ["Pre-authorisation - emergency admission", "2 hours", "1 hour"],
                ["Enhancement of approved amount", "4 hours", "2 hours"],
                ["Deficiency letter after document receipt", "15 days", "7 days"],
                ["Settlement after last necessary document", "30 days", "30 days (unchanged)"],
            ],
            widths=[80 * mm, 40 * mm, 40 * mm],
        ),
        h1("SECTION 2 - MANDATORY CLAUSE REFERENCING"),
        h2("2.1 Requirement"),
        p(
            "With effect from the date of this Circular, every repudiation and every partial "
            "settlement must record the applicable rejection reason code and the specific clause "
            "of the Policy Wording relied upon. A repudiation letter that states a reason without "
            "a clause reference must not be despatched."
        ),
        h2("2.2 Rationale"),
        p(
            "Where a policyholder escalates a decision to the Grievance Officer or to the "
            "Insurance Ombudsman, the Company must be able to demonstrate the basis of the "
            "decision by reference to the wording in force on the date of loss. A reason recorded "
            "without a clause reference cannot be defended."
        ),
        h1("SECTION 3 - APPLICABLE POLICY VERSION"),
        h2("3.1 Date of Loss Governs"),
        p(
            "Assessors are reminded that the Policy Wording applicable to a claim is the wording "
            "in force on the date of admission, and not the wording current at the time of "
            "assessment. Following the release of Family Health Optima Version 3.2 on 1 April "
            "2024, claims with a date of loss before that date continue to be governed by Version "
            "2.8, including the forty-eight month pre-existing disease waiting period and the "
            "room rent cap of four thousand rupees per day."
        ),
        h2("3.2 Common Error"),
        p(
            "The most frequently observed assessment error in the preceding quarter was the "
            "application of the revised thirty-six month pre-existing disease waiting period to "
            "claims governed by Version 2.8. All such assessments completed since 1 April 2026 "
            "are to be reviewed and corrected within thirty days."
        ),
        h1("SECTION 4 - ACKNOWLEDGEMENT"),
        p(
            "All claims processing offices must acknowledge receipt of this Circular and confirm "
            "that processing staff have been briefed, within seven days of the effective date."
        ),
    ]
    return Doc(
        filename="acme_internal_circular_2026_09_claim_timelines.pdf",
        title="ACME GENERAL INSURANCE COMPANY LIMITED",
        subtitle="Internal Circular AGI/CL/2026/09 - Revised Claim Handling Timelines",
        meta=[
            "Document type: Circular &nbsp;|&nbsp; Issued by: Head of Health Claims",
            "Effective from 01/07/2026 &nbsp;|&nbsp; Applies to all claims offices and the TPA",
        ],
        body=body,
    )


DOCUMENTS = [
    family_health_v32,
    family_health_v28,
    motor_shield,
    endorsement,
    claims_sop,
    circular,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the sample insurance corpus")
    parser.add_argument("--out", default="pdf", help="output directory (default: pdf)")
    parser.add_argument("--clean", action="store_true", help="remove existing PDFs first")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for old in out_dir.glob("*.pdf"):
            old.unlink()

    print(f"Writing {len(DOCUMENTS)} documents to {out_dir}/\n")
    total = 0
    for factory in DOCUMENTS:
        doc = factory()
        path = build(doc, out_dir)
        size = path.stat().st_size
        total += size
        print(f"  {path.name:58s} {size / 1024:6.1f} KB")

    print(f"\n{len(DOCUMENTS)} documents, {total / 1024:.0f} KB total.")
    print("Upload them from the Documents page, or:")
    print(f"  python scripts/ingest_folder.py --path {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
