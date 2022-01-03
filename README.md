# HER2-Prediction
Achieve HER2 classification of breast cancer

You can run the test file directly from the features I extracted.
The extracted features are in ../Her2_Prediction/feature/.
The  test code is in ../Her2_Prediction/3_classifier/.

all_lable.pkl stores the labels of the data.
all_ICC59.xlsx stores the ICC value.

Here is feature extraction code.
First, extract radiomics feature.
run ../Her2_Prediction/1_extract_radiomics_feature/pkg_features.py
And you can extract more or less than 1130 features by altering Params_labels.yaml.
reference:https://pyradiomics.readthedocs.io/en/latest/

Second, extract dsfr feature.
In ../Her2_Prediction/2_extract_DSFR_feature, run the code in sequence.

Finally, run the code in ../Her2_Prediction/3_classifier

Here, 2 refers to HER2-overexpressing, 1 refers to HER2-low-positive, 0 refers to HER2-zero.
