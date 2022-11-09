# HER2-Prediction
Implementation of HER2 type diagnostics
This is the whole implementation step of HER2 type diagnostics.

First, extract radiomics feature.
run ../Her2-Prediction/1_extract_radiomics_feature/pkg_features.py
And you can extract more or less than 1130 features by altering Params_labels.yaml.
reference:https://pyradiomics.readthedocs.io/en/latest/

Second, extract dsfr feature.
In ../Her2_Prediction/2_extract_DSFR_feature, run the code in sequence.

Finally, run the code in ../Her2_Prediction/3_classifier and get the result.
