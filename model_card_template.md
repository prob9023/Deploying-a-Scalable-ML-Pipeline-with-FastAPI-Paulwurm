# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Developed by Paul Aulwurm
Date: May 2026
Model Type: Random Forest Classifier


## Intended Use

This is an educational project to demonstrate MLOps on the publicly available Census Bureau data. Primary intended users are myself and Udacity Evaluators. This model is not intended to make any real world decisions.

## Training Data

75% of the total dataset is used for training. Categorical features are converted via OneHotEncoder and labels are processed via LabelBinarizer

## Evaluation Data

25% of the total dataset is evaluated. Same preprocessing as training data. 

## Metrics
The three metrics used are Precision, Recall, and F1-Score. The scores are:

Precision: 0.7403 | Recall: 0.6233 | F1: 0.6768

## Ethical Considerations

This dataset includes sensitive descriptors such as Sex and Race. There are historical inequalities that could be perpetuated if not taken into consideration. 

## Caveats and Recommendations

Dataset appears to be from 1994, which is likely to be too stale for current economic, educational, or societal trends.