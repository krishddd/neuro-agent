# Neuro-Oncology Report — Patient P001
*Generated: 2026-04-26 11:51*

---

## Patient Record
- **Diagnosis:** glioblastoma
- **Diagnosis date:** 2023-10-21
- **Age/Sex:** 63 / M

*The enhancing lesion in the right parietal lobe has progressed, showing enlargement, new satellite foci, and increased perilesional oedema with midline shift, consistent with glioblastoma multiforme (GBM). This represents progressive disease (PD) compared to the prior visit.*

## MRI Findings
### Visit v1
**Impression:** Enhancing mass in the right parietal lobe, consistent with Glioblastoma Multiforme (GBM). Significant surrounding edema and mass effect are present.
**Flags:** mass effect, discrepancy: The provided radiology report is incomplete and contains no specific findings for Visit 1 (v1). The image shows a large, enhancing, heterogeneous mass in the right parietal lobe, which is consistent with the diagnosis stated in the report header (GBM, right parietal), but the specific findings and measurements are not documented in the body of the report.

| Location | Size (mm) | Enhancement | Notes |
|---|---|---|---|
| Right parietal lobe | 50.0 | heterogeneous | Large, heterogeneous mass in the right parietal lobe, consistent with Glioblastoma Multiforme (GBM). |

### Visit v2
**Impression:** Enhancement of the right parietal lobe lesion, consistent with Glioblastoma Multiforme (GBM). The lesion shows signs of progression (Progressive Disease, PD) compared to Visit 1, with increased perilesional oedema and midline shift.
**Flags:** mass effect

| Location | Size (mm) | Enhancement | Notes |
|---|---|---|---|
| Right parietal lobe | 39.8 | ring | Enhancing lesion in the right parietal lobe, showing signs of enlargement and new satellite foci of  |

## RECIST 1.1 Assessment
| Field | Value |
|---|---|
| Response | 🔴 **PD** |
| Baseline sum | n/a mm |
| Current sum | 10.8 mm |
| Change | n/a |
| New lesion detected | **YES — automatic PD** |
| Confirmation required | No |

*New lesion detected — automatic Progressive Disease per RECIST 1.1.*

## Urgency Triage
**Level:** 🚨 CRITICAL (score 5/5)
**Drivers:** critical:midline shift

*Critical keywords: ['midline shift']; high keywords: ['mass effect', 'edema', 'enhancement', 'progression']; RECIST PD: True.*

## Treatment Optimization (Phase 4 SMBO v3.0)
**MDT Decision:** ⚠️ **MODIFY**
**Reason:** Neurosurgeon's urgent concern about midline shift requiring decompression takes precedence over systemic therapy, while neuro-oncologist and pharmacist highlight guideline non-concordance and toxicity risks. Neuroradiologist's uncertainty about pseudoprogression adds imaging urgency.
**Proposed Regimen:** `defer`
**Modifications:** Defer systemic therapy until post-surgical imaging confirms true progression; Refer to NCT06558214 trial (if ECOG improves) or consider bevacizumab monotherapy per NCCN guidelines; Implement PJP prophylaxis if bevacizumab resumes
**⚠️ MDT Board Discussion Required**

*The patient requires urgent neurosurgical evaluation for decompression due to critical midline shift, which cannot safely await systemic therapy. Bevacizumab must be paused pre-operatively due to wound-healing risks. Post-surgery, imaging must confirm true progression before resuming therapy. Given ECOG 2 status and toxicity concerns, bevacizumab monotherapy (NCCN-recommended) or trial referral (if ECOG improves) should be prioritized over high-toxicity combos.*

### Survival Prediction (GP + RSF)
| Metric | Value |
|---|---|
| Predicted RECIST Δ | +5.7% (σ=15.343) |
| Predicted PFS | 13.0 weeks |
| 95% CI | 1.6 – 114.9 weeks |
| Optimization triggered | Yes |

### SHAP Explainability — Top 5 Drivers
*Base PFS: 23.7 weeks (population average for this cancer type)*

| Feature | SHAP Impact | Direction |
|---|---|---|
| ecog_ps_score | 🔴 4.3w | - |
| egfr_ml_per_min | 🔴 3.3w | - |
| new_lesion_flag | 🔴 1.9w | - |
| treatment_duration_weeks | 🔴 1.1w | - |
| dose_reduction_flag | 🔴 1.0w | - |

![SHAP Waterfall](C:\Users\aieuser\Documents\neuro_agent\outputs\P001\plots\shap_waterfall.png)

## Medications
### Current
| Drug | Dose | Frequency | Route | Start |
|---|---|---|---|---|
| Bevacizumab | 10 mg/kg | every 2 weeks | IV | 2024-01-31 |
| Lomustine | 90 mg/m² | every 6 weeks | PO | 2024-01-31 |
| Dexamethasone | mg/m² | - | PO | - |
| Omeprazole | mg/m² | - | PO | - |
| Fluconazole | mg/m² | - | PO | - |
| Valproate sodium | mg/m² | - | PO | - |
| Low-molecular-weight heparin (LMWH) | mg/m² | - | IV | - |
### Historical
| Drug | Dose | Stop date |
|---|---|---|
| Temozolomide | 75 mg/m² | 2024-01-31 |
| Temozolomide | 150 mg/m² | 2024-01-31 |

## Drug Interactions
**Highest severity:** 🚨 contraindicated

| Drug A | Drug B | Severity | Mechanism |
|---|---|---|---|
| Dexamethasone | Bevacizumab | 🟠 moderate | Steroid-induced hypertension may compound bevacizumab hypertensive effect |
| bevacizumab | lomustine | 🟠 moderate | Both bevacizumab and lomustine may contribute to myelosuppression, potentially increasing the risk of hematologic toxici |
| bevacizumab | low-molecular-weight heparin (lmwh) | 🔴 major | Bevacizumab increases the risk of thromboembolic events due to anti-angiogenic effects, while LMWH is an anticoagulant.  |
| bevacizumab | omeprazole | 🟡 minor | Omeprazole may increase the risk of gastrointestinal perforation when used with bevacizumab, though the evidence is not  |
| bevacizumab | valproate sodium | 🔴 major | Bevacizumab may inhibit the metabolism of valproate sodium, leading to increased plasma concentrations of valproate and  |
| dexamethasone | fluconazole | 🟠 moderate | Fluconazole inhibits CYP3A4 and CYP2C9, which are involved in the metabolism of dexamethasone, leading to increased dexa |
| dexamethasone | lomustine | 🟡 minor | Dexamethasone may induce hepatic enzymes (e.g., CYP3A4), potentially altering lomustine metabolism. However, lomustine i |
| dexamethasone | low-molecular-weight heparin (lmwh) | 🟡 minor | Pharmacodynamic interaction: Dexamethasone may increase the risk of gastrointestinal bleeding, which could be additive t |
| dexamethasone | omeprazole | 🟡 minor | Omeprazole may inhibit CYP3A4, a metabolic pathway for dexamethasone, potentially increasing dexamethasone plasma concen |
| Dexamethasone | Temozolomide | 🟡 minor | Corticosteroids may reduce temozolomide oral bioavailability |
| dexamethasone | valproate sodium | 🟠 moderate | Dexamethasone may increase the hepatic metabolism of valproate sodium via induction of CYP enzymes (e.g., CYP2C9), poten |
| fluconazole | lomustine | 🟡 minor | Fluconazole is a CYP3A4 inhibitor, but lomustine is primarily metabolized via non-CYP pathways (e.g., hydrolysis). Howev |
| fluconazole | low-molecular-weight heparin (lmwh) | 🟡 minor | Fluconazole may reduce renal clearance of low-molecular-weight heparin (LMWH) by impairing renal function, potentially i |
| fluconazole | omeprazole | 🟡 minor | Fluconazole inhibits CYP3A4, which is involved in the metabolism of omeprazole, potentially increasing omeprazole plasma |
| fluconazole | temozolomide | 🚨 contraindicated | Fluconazole inhibits CYP2C9, the primary metabolic enzyme for temozolomide, leading to increased plasma concentrations o |
| fluconazole | valproate sodium | 🔴 major | Fluconazole inhibits hepatic metabolism of valproate sodium, potentially increasing valproate plasma concentrations and  |
| lomustine | low-molecular-weight heparin (lmwh) | 🟡 minor | Lomustine may cause myelosuppression, including thrombocytopenia, which could increase the risk of bleeding when combine |
| Lomustine | Temozolomide | 🔴 major | Additive myelosuppression; both agents are alkylating |
| lomustine | valproate sodium | 🚨 contraindicated | Lomustine inhibits the hepatic metabolism of valproate sodium, leading to increased plasma concentrations of valproate a |
| low-molecular-weight heparin (lmwh) | omeprazole | 🟡 minor | Omeprazole may increase the risk of gastrointestinal bleeding when used with anticoagulants like LMWH, though there is n |
| low-molecular-weight heparin (lmwh) | temozolomide | 🟠 moderate | Additive risk of bleeding due to the anticoagulant effect of low-molecular-weight heparin (LMWH) and the myelosuppressiv |
| low-molecular-weight heparin (lmwh) | valproate sodium | 🟡 minor | Valproate sodium may cause thrombocytopenia, which, when combined with the anticoagulant effects of low-molecular-weight |
| omeprazole | valproate sodium | 🟠 moderate | Omeprazole may inhibit hepatic metabolism of valproate sodium, potentially increasing valproate plasma concentrations vi |
| temozolomide | valproate sodium | 🔴 major | Valproate sodium inhibits carboxylesterase 1 (CES1), an enzyme involved in the metabolic activation of temozolomide, lea |

## Treatment–Response Correlation
RECIST response: PD | Sum change: n/a | Treatment started: 2023-10-21 | Nearest preceding drug event: Temozolomide stop (2024-01-31) | New lesion: True | Confirmation required: False

### Medication Event Timeline
| Date | Drug | Event | Dose |
|---|---|---|---|
| 2023-10-21 | Temozolomide | start | 75 mg/m² |
| 2023-10-21 | Temozolomide | dose_change | 150 mg/m² |
| 2024-01-31 | Bevacizumab | start | 10 mg/kg |
| 2024-01-31 | Lomustine | start | 90 mg/m² |
| 2024-01-31 | Temozolomide | stop | 75 mg/m² |
| 2024-01-31 | Temozolomide | stop | 150 mg/m² |

## Clinical Timeline
| Date | Type | Event |
|---|---|---|
| 2023-10-21 | visit | Diagnosis: glioblastoma |
| 2023-10-21 | med_start | Start Temozolomide 75 mg/m² |
| 2023-10-21 | med_start | Start Temozolomide 150 mg/m² |
| 2024-01-31 | med_start | Start Bevacizumab 10 mg/kg |
| 2024-01-31 | med_start | Start Lomustine 90 mg/m² |
| 2024-01-31 | med_stop | Stop Temozolomide |
| 2024-01-31 | med_stop | Stop Temozolomide |
| 2026-04-26 | scan | MRI v1: Enhancing mass in the right parietal lobe, consistent with Glioblastoma Multiforme (GBM). Significant surrounding edema  |
| 2026-04-26 | scan | MRI v2: Enhancement of the right parietal lobe lesion, consistent with Glioblastoma Multiforme (GBM). The lesion shows signs of  |
| 2026-04-26 | note | Treatment Optimization: MDT decision=MODIFY — defer |
| 2026-04-26 | note | MDT Board Discussion Required — escalated for multidisciplinary review |
| 2026-04-26 | note | Survival Prediction: PFS=13.0w (CI 1.6–114.9w), RECIST delta predicted=+5.7%, σ=15.343 |

## Patient Summary
### Patient Letter

Dear Patient,

We are writing to update you about your recent scans and treatment. MRI results show that the tumour in your right parietal lobe has grown since your last scan, and a new area of concern has been found. This suggests the condition is progressing. Due to the tumour's size and location, there is pressure on parts of your brain, which requires urgent attention. Your care team is discussing next steps, including possible surgery to relieve this pressure. Currently, you are receiving medications including Bevacizumab, Lomustine, and Dexamethasone to manage symptoms and slow progression. We are also monitoring for potential drug interactions, which your pharmacist will review with you. Your treatment plan may need adjustments based on upcoming tests. Please ensure you attend all appointments and report any new symptoms promptly. This is a research prototype. Please discuss any questions with your care team.

---

### GP Handover

Diagnosis: Glioblastoma multiforme (GBM) with progressive disease (PD) per RECIST 1.1. Imaging: Right parietal lobe enhancing lesion with increased perilesional edema and mass effect; new lesion detected. RECIST: PD confirmed by new lesion (automatic classification). Medications: Bevacizumab, Lomustine, Dexamethasone, Omeprazole, Fluconazole, Valproate sodium. Interactions: Major interactions include bevacizumab+valproate sodium, bevacizumab+LMWH, fluconazole+valproate sodium; contraindications include lomustine+valproate sodium, temozolomide+valproate sodium, fluconazole+temozolomide. Plan: Urgent neurosurgical evaluation required for decompression due to critical midline shift. Defer systemic therapy until post-surgical imaging confirms true progression. Consider bevacizumab monotherapy (NCCN) or trial referral (NCT06558214) if ECOG improves. Initiate PJP prophylaxis if bevacizumab resumes. Monitor for toxicity; MDT discussion required for treatment optimization. Survival prediction: PFS median 13 weeks (CI 1.6-114.9w).


---
*Research prototype — outputs must be verified by a licensed healthcare professional before clinical use.*
