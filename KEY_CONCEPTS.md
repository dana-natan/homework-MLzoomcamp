# ML Zoomcamp: Key Concepts to Remember

A concise reference for ideas worth recalling **without looking at solutions**. Add new concepts only when they come up in the homework or code review.

## Classification: Precision vs. Recall

In binary classification, *positive* means the class we're trying to detect (for example, a customer who signs up).

| Metric | Question it answers | Formula | Memory cue |
| --- | --- | --- | --- |
| **Precision** | Of all cases predicted positive, how many really were positive? | TP / (TP + FP) | **Avoid false alarms** |
| **Recall** (sensitivity) | Of all actual positive cases, how many did we identify? | TP / (TP + FN) | **Avoid missing real cases** |

- **TP (true positive):** Predicted positive, actually positive.
- **FP (false positive):** Predicted positive, actually negative.
- **FN (false negative):** Predicted negative, actually positive.

**Example:** 100 customers actually sign up. The model predicts 80 signups, 60 of which are correct.
- Precision = 60 / 80 = **75%**.
- Recall = 60 / 100 = **60%**.

**Threshold trade-off:** Lowering the classification threshold typically predicts more positives, increasing recall but potentially producing more false positives (and reducing precision). The fitted model does not change, only the decision rule. The precise trade-off depends on the data.

**When to prioritize which:** Prioritize recall when missing a true positive is costly (e.g., initial disease screening with follow-up testing). Prioritize precision when acting on a false positive is costly or capacity is limited (e.g., a sales team with very few calls). In real applications, consider both metrics and the costs of mistakes.

### Retrieval practice

1. Without looking, explain precision and recall in your own words.
2. Which denominator uses *predicted positives*, and which uses *actual positives*?
3. If a model misses many real positives, which metric suffers?
4. Why might a business prefer precision over recall?

---
Last updated: 2026-10-28 (Module 3, classification).
