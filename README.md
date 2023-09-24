# QIN-GBM_TCIA_Classification
In this project, we propose extracting imaging features from pre-radiation MRI along with machine learning have a potential to classify patients with REP or no-REP
Patients with glioblastoma has low survival rates regardless of treatment modality. The standardized approach of treatment for patients with glioblastoma is resection of tumor fol- lowed by co-current radiation therapy with temozolamide (TMZ). However, a subset of glioblastoma patients has shown progression of disease or tumor after the surgery but prior to radiation therapy interval. Recent clinical research indicates these subsets of patients indicates rapid early progression (REP) and has poor survival compared to patients who does not have REP. Therefore, classification of patients who have rapid early progression will assist in treatment planning and strategies. Magnetic Resonance Imaging (MRI) is a diagnostic tool to investigate post-treatment impact in central nervous system (CNS) tumors. In addition, radiomics assist in transforming image into mineable data. The radiomic procedure involves image acquisition, segmentation, and annotation of normal/tumor tissues, extracting quantitative feature (shape, area, texture and geometric) which is followed by statistical analysis and machine learning. 

The raw data is collected from TCIA under the project name QIN-GBM Treatment Response Project:
https://wiki.cancerimagingarchive.net/display/Public/QIN+GBM+Treatment+Response

The features are extracted using R programming RIA tool:
https://cran.r-project.org/web/packages/RIA/index.html

The extracted features are saved in csv file format. There are two models that has been utilized for result generation.

MODEL_1: Imabalabce_XGBoost
MODEL_2: One_Class_SVM

To run the code, all the files need to be one the same directory under same folder.
