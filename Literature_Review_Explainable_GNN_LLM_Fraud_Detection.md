# Literature Review: Explainable GNN + LLM Financial Fraud Detection

Project PW_25_MAR_02 · Phase-I deliverable (draft v2)

**Evidence level**
- **Read in full (text extracted):** xFraud, SCAFDS, MLED (the first 140 lines plus targeted sections), DGP (HTML).
- **Abstract only:** GNNExplainer, PGExplainer, Sha et al. (HGNN), Pereira & Murai.
- **Third-party summary only:** FLAG (ACM DOI 10.1145/3711896.3737220, not retrieved).
- Items marked **[verify]** still need checking against the source.

---

## 1. Paper-by-paper

### xFraud — Rao et al., PVLDB 15(3), 2022 (arXiv 2011.12193) · *read in full*
- **Detector:** heterogeneous graph transformer ("xFraud detector+") with distributed training. A GraphSAGE-style sampler gives faster inference without a notable AUC loss.
- **Explainer:** a **hybrid** of GNNExplainer edge weights and edge-centrality measures: `A·w(centrality) + B·w(GNNExplainer)`. The coefficients are learned by ridge regression or by maximising hit rate.
- **Data (proprietary eBay):**
  - eBay-small: 289K nodes, 613K edges, 114 features, 4.30% fraud.
  - eBay-large: 8.9M nodes, 13.2M edges, 480 features, 3.57% fraud.
  - eBay-xlarge: used for end-to-end runs.
- **Detection:** AUC on eBay-xlarge with 8 machines was 0.9074 (xFraud) vs 0.8961 (GEM) vs 0.8879 (GAT). With 16 machines xFraud scored 0.8892, below GEM at 0.8938. Authors say its ROC curve is better at low false-positive rates.
  Benign transactions were down-sampled and pre-filtered. In production-like conditions, about 1 of 3 flagged candidates is real fraud, at 0.1 recall.
- **Explainer evaluation:** *Top-k hit rate* against **human-annotated edge importance** on 41 communities.
  - Top-5 hit rate: GNNExplainer 0.445, edge betweenness 0.469, degree 0.464, **random 0.127**.
  - Neither GNNExplainer nor centrality dominates, hence the hybrid.
- **Takeaways for us**
  - GNNExplainer is **not** clearly better than simple centrality on fraud graphs. Include a centrality baseline.
  - Their explanation ground truth is human annotation. We will need an automated proxy (fidelity) and, ideally, a small human check.
  - The gains over baselines are modest and not uniform.
- **Gap:** proprietary data. Explanations are edge weights only, with no language and no LLM.

### GNNExplainer — Ying et al., NeurIPS 2019 (arXiv 1903.03894) · *abstract*
- Per-instance, model-agnostic. Finds a compact subgraph and a small feature subset that maximise mutual information with the prediction.
- Beats baselines by 17.1% on average (synthetic and real).
- Optimised separately per instance, so it is slow and does not generalise.

### PGExplainer — Luo et al., NeurIPS 2020 (arXiv 2011.04573) · *abstract*
- A neural network is trained once to produce edge masks. It can explain many instances and apply inductively.
- Up to 24.7% relative AUC gain on graph classification explanation.
- Benchmarks only, not financial heterogeneous graphs **[verify]**.

### FLAG — Yang et al., KDD 2025 · *third-party summary*
- Problem: graph fraud models ignore text, and LLMs face *neighbourhood camouflage* and input-length limits.
- Method: semantic-similarity neighbour sampling, LLM-based node enhancement, and fine-tuning of the LLM toward label-relevant text.
- Reported: +3.14% F1-macro and +6.97% AUC on public datasets (average). +0.9% KS on an Alipay credit-risk dataset.
- Needs node text. Datasets and baselines are **[verify]**. The LLM improves features and does not explain predictions.

### MLED — Huang et al., ACM MM 2025 (arXiv 2507.11997) · *read in part*
- LLM-enhanced graph fraud detection with a type-level enhancer and a relation-level enhancer. It plugs into existing GNNs such as ConsisGAD.
- Data: Amazon, YelpChi, T-Finance, T-Social. All are users and comments, i.e. **text-rich**. Metrics: AUROC, AUPRC, F1-macro.
- Example gain: ConsisGAD on Amazon improves 1.77 / 1.76 / 2.11 (AUROC / AUPRC / F1) with MLED.
- The authors themselves note that raw node text is hard to get because of privacy.
- Detection only, with no explanations.

### DGP — Li et al., 2025 (arXiv 2507.21653) · *read in full*
- Graph-enhanced LLM (Qwen3-8B with LoRA, Yes/No). Keeps target-node text and compresses neighbours into metapath summaries chosen by diffusion-based trimming.
- Data: YelpReviews, AmazonVideo (public), plus 2 proprietary graphs. About 1,300 labelled training nodes each.
- Yelp AUPRC 48.87 vs 42.11 for the best baseline. Gains are smaller on Amazon.
- Detection only, and text-heavy datasets.

### Sha et al. — HGNN with Graph Attention, 2025 (arXiv 2504.08183) · *abstract*
- User, merchant and transaction nodes, graph attention with temporal decay, SMOTE and cost-sensitive learning. Claims to beat GCN, GAT and GraphSAGE on IEEE-CIS (no numbers on the abstract page).
- No explainability. Split and leakage handling are **[verify]**.

### Pereira & Murai — 2021 (arXiv 2105.14568) · *abstract*
- Message passing can amplify class imbalance around hubs. Imbalance remedies ignore concept drift. Introduces a synthetic data generator.
- Supports using a temporal split. It is a small workshop paper in Portuguese.

### SCAFDS — Uddin, 2026 preprint (arXiv 2605.18913) · *read in full*
Closest to our idea, but the overlap is **smaller than the abstract suggests**.

- **Where each dataset is used**
  - **IEEE-CIS** (590,540 transactions, 432 features, 3.50% fraud) is used **only for tabular and BiLSTM models**, with no GNN. Results: LightGBM AUPRC 0.772 / AUROC 0.962; XGBoost 0.651 / 0.932; Random Forest 0.561 / 0.898; BiLSTM 0.382 / 0.807. Whether the split is temporal is **[verify]**.
  - The **GNN (SCAFDS) is evaluated only on a synthetic interbank network** (8,103 institutions). Reported AUPRC 0.515 vs 0.356 for GraphSAGE-AML. The paper itself says the synthetic edge features and labels share a source, and on real FDIC enforcement labels the gain is +0.002 AUPRC with near-random AUROC (~0.49).
- **Explanations:** SHAP on **LightGBM**, plus edge-feature significance and BiLSTM attention. There is **no GNN explainer** (GNNExplainer or PGExplainer).
- **Narratives:** GPT-4o at temperature 0, constrained by thresholds on the attribution values.
- **Narrative evaluation (their Table III):** "grounding rate" 0.60 overall. By their own explanation, Layer 1 (0.30) is fixed by the top-30% threshold, and Layer 3 (1.00) is "near-mechanically guaranteed". Factual accuracy 0.577; Form-111 compliance 0.587. There is **no comparison with an unconstrained LLM and no human evaluation** (both listed as future work).
- **Takeaways for us**
  - Novelty stands if we (a) run GNN explainers on a **real transaction graph**, and (b) evaluate narratives with a genuine verifier, not threshold-by-construction numbers.
  - **LightGBM is very strong on IEEE-CIS** in their numbers. Our GNN must be positioned honestly against it.
  - Code is public (github.com/nasiruddinstudents-ctrl/SCAFDS-Fraud-Detection), so we can inspect it.

---

## 2. Synthesis

**Established**
- Heterogeneous graphs for transactions work (xFraud).
- GNN explainers exist (GNNExplainer, PGExplainer), but on fraud graphs they are **not clearly better than simple centrality** (xFraud Table 1).
- LLMs help graph fraud detection, as feature enhancers (FLAG, MLED) or as the classifier (DGP). These all target **text-rich** datasets.
- Imbalance and drift matter (Pereira & Murai).
- A pipeline from detection to narrative exists in concept (SCAFDS), but it explains a tabular model and its narrative evaluation is weak.

**Not covered by anything found**
1. A GNN explainer's output (GNNExplainer/PGExplainer) on a **real transaction graph** converted into evidence for an LLM narrative.
2. A **verifier** that checks each claim in the narrative against that evidence, compared with an unconstrained LLM baseline.
3. A fair comparison of explanation methods (GNN explainers vs centrality vs SHAP) on IEEE-CIS.
4. LLM-based work in this area targets **text** data. IEEE-CIS is tabular, so using the LLM for explanation rather than feature extraction fits it better.

**Draft gap statement**
> Prior work either detects fraud with GNNs, explains GNN predictions as edge weights, or uses LLMs to improve detection on text-rich graphs. The one system that narrates fraud evidence explains a tabular model, evaluates narratives with threshold-based metrics, and has no comparison against an unconstrained LLM. We build a pipeline in which a heterogeneous GNN's explainer output on a real transaction graph is converted into structured evidence, narrated by an LLM that may only use that evidence, and verified claim by claim against an unconstrained-LLM baseline.

---

## 3. Design decisions this supports

| Decision | Source |
|---|---|
| Temporal split, PR-AUC headline, recall at fixed precision | Pereira & Murai; SCAFDS and DGP report AUPRC |
| Strong tabular baselines (**LightGBM** as well as XGBoost) | SCAFDS: LightGBM beats XGBoost, RF and BiLSTM on IEEE-CIS |
| Explainer comparison: GNNExplainer, PGExplainer, **edge centrality**, SHAP | xFraud: neither GNNExplainer nor centrality dominates |
| Explanation metrics: top-k hit rate if we annotate a small set, plus fidelity (sufficiency, comprehensiveness) | xFraud's evaluation protocol |
| Narrative verifier plus an **unconstrained-LLM baseline** | SCAFDS lacks both |
| Do not rely on LLM text features (IEEE-CIS has no text) | FLAG, MLED and DGP depend on text |

## 4. Still to do

1. Obtain the FLAG paper from ACM and fill in its datasets and baselines.
2. Check the **[verify]** items: split and leakage in Sha et al., split in SCAFDS Track A, PGExplainer evaluation.
3. Skim SCAFDS's GitHub for its prompt template and verifier.
4. Add an explainability-evaluation reference (fidelity, sufficiency and comprehensiveness) to support our metrics.

## 5. References

- Rao et al., xFraud, PVLDB 2022. https://arxiv.org/abs/2011.12193
- Ying et al., GNNExplainer, NeurIPS 2019. https://arxiv.org/abs/1903.03894
- Luo et al., PGExplainer, NeurIPS 2020. https://arxiv.org/abs/2011.04573
- Yang et al., FLAG, KDD 2025. https://doi.org/10.1145/3711896.3737220
- Huang et al., MLED, ACM MM 2025. https://arxiv.org/abs/2507.11997
- Li et al., DGP, 2025. https://arxiv.org/abs/2507.21653
- Uddin, SCAFDS, 2026 (preprint). https://arxiv.org/abs/2605.18913
- Sha et al., HGNN with Graph Attention, 2025. https://arxiv.org/abs/2504.08183
- Pereira & Murai, How effective are GNNs in Fraud Detection for Network Data?, 2021. https://arxiv.org/abs/2105.14568
