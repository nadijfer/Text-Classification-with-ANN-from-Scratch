# Comparative Study of Text Classification: Implementing ANN from Scratch with BoW and TF-IDF

This repository presents an experimental study on text classification by implementing an Artificial Neural Network (ANN) from scratch, without relying on high-level deep learning frameworks. The project serves as an application of concepts learned in the Natural Language Processing (NLP) and Artificial Neural Network courses, with the objective of reinforcing both mathematical understanding and low-level implementation details.

The primary focus of this study is to compare two text representation techniques:

1. Bag-of-Words (BoW) as a baseline approach
2. TF-IDF as an improvement over the baseline

Both representations are used as input features for the same ANN architecture, allowing a controlled comparison where the feature extraction method is the main varying factor. References for the implementation are primarily derived from lecture notes and course materials.

## Experimental Setup

* Model: Artificial Neural Network implemented from scratch
* Task: Binary text classification (spam vs. ham)
* Baseline Features: Bag-of-Words (BoW)
* Improved Features: Term Frequency–Inverse Document Frequency (TF-IDF)
* Evaluation Metrics: Accuracy, Precision, Recall, and F1-Score

The baseline and improved models were trained and evaluated under comparable conditions. For the TF-IDF model, a decision threshold adjustment was applied to better balance precision and recall.

## Results
### Baseline Model (Bag-of-Words)

| Metric | Score |
| --- | --- |
| Accuracy | 93.60 |
| Precision | 94.15 |
| Recall | 78.16 |
| F1-Score | 85.41 |

The BoW-based model achieves high precision but relatively low recall, indicating that while spam predictions are often correct, a substantial number of spam emails are misclassified as ham (false negatives).

### Improved Model (TF-IDF)

| Metric | Score |
| --- | --- |
| Accuracy | 95.81 |
| Precision | 90.09 |
| Recall | 92.72 |
| F1-Score | 91.39 |

The TF-IDF-based model demonstrates a more balanced performance across all metrics. In particular, the recall improves significantly, suggesting that TF-IDF enables the model to better capture discriminative terms that are informative for identifying spam emails.

## Conclusion

The experimental results indicate that TF-IDF provides a meaningful improvement over the Bag-of-Words representation when used with an ANN implemented from scratch. While BoW tends to emphasize raw term frequency, TF-IDF mitigates the influence of overly common terms through inverse document frequency weighting, resulting in more informative feature representations.

This improvement is reflected in higher recall and F1-score, demonstrating that TF-IDF allows the model to generalize better without modifying the underlying network architecture. Therefore, the enhancement lies in the feature representation rather than model complexity.

The complete implementation and experimental results can be found in this [notebook](spam-email-using-ann-from-scratch.ipynb).