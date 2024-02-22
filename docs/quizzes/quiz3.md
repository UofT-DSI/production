# Quiz 3 - Working with Training Data

1. Select all random sampling methods from the options below:
Reservoir sampling
Quota sampling
Stratified sampling
Snowball sampling

2. Reservoir sampling is used with:
Batch data
Streaming data

3. Labelling observations by hand is:
Costly
Error-prone
Slow
All of the above

4. Label multiplicity is (select all that apply):
Reduced with clear definitions and task guidance.
A byproduct of weak supervision algorithms.
When multiple labels are applied to the same data instance by different labelers.
Required for active learning.

5. Is similarity a semi-supervised method?
Yes, it assumes that similar instances share similar labels.
No, it is a distance-based method.
Yes, it works by creating new samples that are similar to the original data, but with some added perturbations.

6. Select all True statements from the options below.
Class imbalance occurs when a minority class that is of interest is extremely underrepresented in the total sample.
Class imbalance can result in over-fitted models.
Class imbalance can result in under-fitted models.
Class imbalance is uncommon in most domains.

7. Performance metrics that use class scores or class probabilities are:
Cohen's Kappa
F1-score
ROC AUC
Log loss

8. Pipelines in sklearn are:
A practical way of composing transforms and classifiers.
Only used during training.
Only necessary in complex models.
Used to calculate preprocessing steps with the appropriate training samples during cross-validation.

9. Parameters in Pipline steps:
Cannot be accessed once the Pipeline is created.
Are named following the convention <step>__<parameter>.
Can be modified manually or through code.
Can be accessed, but are immutable.

10. Cross-validation ensures that there is no bias in our models.
True
False
