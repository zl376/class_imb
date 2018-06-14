## Comparison of Decision Tree, Random Foreset and Random Forest with undersampled bootstrap for unbalanced data

Note:

This example shows class imbalance of ~200 ( dominant y=0 ). 

+ Precision is always low
+ Using 'class_weight' in Random Forest seems to perform worse with more trees
+ Instead, explicit under-sampling dominant class (y=0) before bootstrap works for Random Forest with more trees
