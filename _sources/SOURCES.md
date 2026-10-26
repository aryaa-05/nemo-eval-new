# Source documents for the `kyc_india` and `rbi_regulations` datasets

Kept **outside** the `nemo-evaluator` repo (CONTEXT.md §7 step 1). Every question in
`datasets/kyc_india/sample.jsonl` and `datasets/rbi_regulations/sample.jsonl` must be traceable to one of
the text files below (`--source-text` points at them).

All three PDFs were downloaded from rbi.org.in on **20 October 2026** with a normal browser session
(rbidocs.rbi.org.in serves an F5 bot challenge; a plain request returns HTML, not a PDF).

| # | file | document (title / number / date as printed on it) | URL of the PDF | used for |
|---|---|---|---|---|
| 1 | `kyc_master_direction.txt` (107 pp) | Master Direction - Know Your Customer (KYC) Direction, 2016, DBR.AML.BC.No.81/14.01.001/2015-16, February 25, 2016 **(Updated as on August 14, 2025)** | https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.PDF (index page: https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566) | `kyc_india` |
| 2 | `basel_iii_master_circular.txt` (328 pp) | Master Circular - Basel III Capital Regulations, RBI/2025-26/08, DOR.CAP.REC.2/21.06.201/2025-26, April 01, 2025 | http://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT08F65FDFBB966942CAA08E0D766F2F6BD4.PDF (index page: https://www.rbi.org.in/Scripts/BS_ViewMasterCirculars.aspx) | `rbi_regulations` |
| 3 | `irac_advances_master_circular.txt` (77 pp) | Master Circular - Prudential norms on Income Recognition, Asset Classification and Provisioning pertaining to Advances, RBI/2025-26/13, DOR.STR.REC.9/21.04.048/2025-26, April 01, 2025 (covers instructions issued up to March 31, 2025) | http://rbidocs.rbi.org.in/rdocs/notification/PDFs/13MC01042025792E33CF094B46F2B838E6409777438D.PDF (index page: https://www.rbi.org.in/Scripts/BS_ViewMasterCirculars.aspx) | `rbi_regulations` |

## How the `.txt` files were made

`pypdf` text extraction of the whole PDF, page order preserved, UTF-8, no post-editing. Known artefacts of
that extraction, which anyone writing questions has to work around:

- **Intra-word spaces** are sometimes inserted (`Fr equently Asked Questions`) and hyphenated line breaks are
  sometimes lost. The validator's excerpt check ignores case, spacing and punctuation, but a *missing or
  added letter* still fails - always check the excerpt against the `.txt`, do not clean it up by eye.
- Smart quotes and dashes inside quoted material often come out as junk bytes (`?Out of Order?`). Avoid
  excerpts that need them; the punctuation is stripped by the check, but the surrounding letters must match.
- Page furniture appears inside the text: `Withdrawn`, the page number on its own line, and footnote
  markers such as `2The Frequently Asked Questions`. Never copy those into an excerpt.
- Hindi text (the letterhead and some notices) is not extractable and shows as junk. Use the English body
  only.

## Verification commands (from the repo root)

```powershell
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --source-text ..\_sources\kyc_master_direction.txt
python src/tools/validate_dataset.py datasets/rbi_regulations/sample.jsonl --source-text ..\_sources\basel_iii_master_circular.txt --source-text ..\_sources\irac_advances_master_circular.txt
```

`answer_plan.json` holds the pre-decided correct letter for each question number (25 of each letter per
dataset), written before any question existed (CONTEXT.md §7 step 3).
