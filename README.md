# Introduction
This dataset describes 10 years (1999-2008) of clinical care at 130 US hospitals and combined delivery networks. 
It comes from the UCI machine learning repository ``` https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008 ```
Data were derived from the database for encounters that met the following guidelines.

# Dataset Information
(1) It is an inpatient encounter (a hospital admission).\
(2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.\
(3) The length of stay was at least 1 day and at most 14 days.\
(4) Laboratory tests were performed during the encounter.\
(5) Medications were administered during the encounter.\
I will demonstrate how to design models that can predict whether the patient will readmit or not 
within 30 days using 130 US hospitals' diabetes dataset.

## Install
```
!pip install xgboost
!pip install graphviz
```
## Importing the libraries
```
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import warnings
import graphviz
import pickle
warnings.filterwarnings('ignore')
%matplotlib inline
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from graphviz import Source
from IPython.display import SVG
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.svm import SVC
```

## Data Exploration
Loading the dataset
```
Diabetes = pd.read_csv('diabetic_data.csv', index_col=False)
```

## Check column names
We can see there are some numerical, categorical and unknown information in this dataset as below:
```
Diabetes.head()
```
<img src="https://raw.githubusercontent.com/nith-ch/diabetes/master/pic/head.PNG" height="130" width="585">

## Check data type
Overview of data columns, types and values
```
Diabetes.info()
```
<img src="https://raw.githubusercontent.com/nith-ch/diabetes/master/pic/info.PNG" height="560" width="300">


## Check mean of each columns
We check mean, standard deviation and other of each columns
```
Diabetes.describe()
```
<img src="https://raw.githubusercontent.com/nith-ch/diabetes/master/pic/describe.PNG" height="175" width="540">

## Count readmitted
From the bar chart, we can see that the patient who readmitted within 30 days are the lowest, while "no" readmitted is the highest
```
Diabetes.readmitted.value_counts().plot(kind='barh', rot=0)
plt.title("Count Readmitted")
plt.xlabel("Amount")
plt.ylabel("Readmitted Day")
```
<img src="https://raw.githubusercontent.com/nith-ch/diabetes/master/pic/count_readmitted.PNG" height="276" width="388">

## Count race
This graph shows the patient race, with Asians having the least, but Caucasians the most.
```
Diabetes.race.value_counts().plot(kind='bar', rot=0, color='green', figsize=(8,5), title="Count Race")
plt.xticks(rotation=45, horizontalalignment="center")
plt.title("Count Race")
plt.xlabel("Race")
plt.ylabel("Amount")
```
<img src="https://raw.githubusercontent.com/nith-ch/diabetes/master/pic/count_race.PNG" height="315" width="495">

## Count gender
From the patient data on this graph, it can be seen that women are more than men, and some of the data does not indicate which gender it is.
```
Diabetes.gender.value_counts().plot(kind='bar', rot=0, color='blue')
plt.title("Count Gender")
plt.xlabel("Gender")
plt.ylabel("Amount")
```
<img src="https://raw.githubusercontent.com/nith-ch/diabetes/master/pic/count_gender.PNG" height="260" width="385">

## Count age
This bar graph shows 10 age groups, each divided into 10 years.
```
Diabetes.age.value_counts().plot(kind='barh', rot=0)
plt.title("Count Age")
plt.xlabel("Amount")
plt.ylabel("Age")
```
<img src="https://raw.githubusercontent.com/nith-ch/diabetes/master/pic/count_age.PNG" height="260" width="395">


## Choose columns and removed examide+citoglipton
We will remove examide and citoglipton because these medicines don't have other status types except "No"
```
Diabetes2 = Diabetes[['race','gender','age','weight','admission_type_id','discharge_disposition_id','admission_source_id', 
                      'time_in_hospital', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications',
                      'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
                      'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                      'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                      'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                      'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
                      'readmitted', 'readmitted_in', 'payer_code']]
```

## Remove ID 11,13,14,19,20,21 because it's related to death or hospice
```
Diabetes2 = Diabetes2.loc[~Diabetes2.discharge_disposition_id.isin([11,13,14,19,20,21])]
```

## Remove ? in 4 columns
Remove unnecessary question masks from race, diag_1, diag_2 and diag_3
```
Diabetes2 = Diabetes2[(Diabetes2.race != "?")]
Diabetes2 = Diabetes2[(Diabetes2.diag_1 != "?")]
Diabetes2 = Diabetes2[(Diabetes2.diag_2 != "?")]
Diabetes2 = Diabetes2[(Diabetes2.diag_3 != "?")]
```

## Calculate the prevalence of population that is readmitted with 30 days
```
def calc_prevalence(y_actual):
    return (sum(y_actual)/len(y_actual))
print('Prevalance:%.3f'%calc_prevalence(Diabetes2['readmitted_in'].values)) #Three decimal places
```

## Represent values “0” represents "No readmission or readmission after 30 days whereas 1 represents readmission within 30 days
The most important column is "readmitted", which tell us if the patient was hospitalized in 30 days, 
more than 30 days or not readmitted.
```
Diabetes2.loc[Diabetes2['readmitted'] == "<30", 'readmittedFL'] = '1'
Diabetes2.loc[Diabetes2['readmitted'] == ">30", 'readmittedFL'] = '0'
Diabetes2.loc[Diabetes2['readmitted'] == "NO", 'readmittedFL'] = '0'
```

## Copy table
This step will backup data for another analysis
```
Diabetes3 = Diabetes2.copy()
```

## Create column list of num
Create "Diabetes_num" to analyst in the next process
```
Diabetes_num = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
                 'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
```

## Convert Diabetes_num to string
```
Diabetes3[Diabetes_num] = Diabetes3[Diabetes_num].astype('str')
```

## Create column list of cal
Create "Diabetes_cal" to analyst in the next process
```
Diabetes_cal = ['race', 'gender', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
                'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'payer_code', 'readmitted_in']
```

## Convert Diabetes_cal to string
```
Diabetes3[Diabetes_cal] = Diabetes3[Diabetes_cal].astype('str')
```

## Replace '?'
We will replace '?' with 'Unknown' to make it easier to analyze.
```
Diabetes3['medical_specialty'] = Diabetes3['medical_specialty'].replace('?','Unknown')
Diabetes3['payer_code'] = Diabetes3['payer_code'].fillna('Unknown')
```

## Group medical_specialty column
We will group data according to the symptoms.
```
pediatrics = ['Pediatrics','Pediatrics-CriticalCare','Pediatrics-EmergencyMedicine','Pediatrics-Endocrinology', \
              'Pediatrics-Neurology','Pediatrics-Pulmonology', 'Anesthesiology-Pediatric', 'Cardiology-Pediatric', \
              'Cardiology-Pediatric','Pediatrics-Hematology-Oncology','Pediadocumentation avtrics-InfectiousDiseases', \
              'Pediatrics-AllergyandImmunology','Pediatrics-InfectiousDiseases']
orthopedics = ['Orthopedics','Orthopedics-Reconstructive']
psychic = ['Psychiatry-Addictive','Psychology','Psychiatry','PhysicalMedicineandRehabilitation', \
           'Osteopath','Psychiatry-Child/Adolescent']
obstetrics = ['ObstetricsandGynecology','Obstetrics','Obsterics&Gynecology-GynecologicOnco']
neurology = ['Neurology','Surgery-Neuro','Pediatrics-Neurology','Neurophysiology']
surgery = ['Surgeon','Surgery-Cardiovascular','Surgery-General', \
          'Surgery-Cardiovascular/Thoracic','Surgery-Colon&Rectal','Surgery-Maxillofacial', \
          'Surgery-Plastic','Surgery-PlasticwithinHeadandNeck','Surgery-Thoracic', \
          'Surgery-Vascular','SurgicalSpecialty','Podiatry','Surgery-Pediatric', \
          'Surgery-Colon&Rectal']
others = ['Endocrinology','Gastroenterology','Gynecology','Hematology','Hematology/Oncology','Hospitalist','InfectiousDiseases', \
          'Oncology','Ophthalmology','Otolaryngology','Pulmonology','Radiology','InternalMedicine','Family/GeneralPractice', \
          'Cardiology','Emergency/Trauma','Urology','Nephrology','Radiologist','Proctology','Dermatology','SportsMedicine', \
          'Speech','Perinatology','Resident','Dentistry','DCPTEAM','AllergyandImmunology','Anesthesiology','Pathology','Endocrinology-Metabolism', \
          'PhysicianNotFound','OutreachServices','Rheumatology']
unknown = ['Unknown']

colmed_spec = []

for val in Diabetes3['medical_specialty'] :
    if val in pediatrics :
        colmed_spec.append('pediatrics')
    elif val in psychic :
        colmed_spec.append('psychic')
    elif val in neurology :
        colmed_spec.append('neurology')
    elif val in surgery :
        colmed_spec.append('surgery')
    elif val in orthopedics :
        colmed_spec.append('orthopedics')
    elif val in obstetrics :
        colmed_spec.append('obstetrics')
    elif val in others :
        colmed_spec.append('others')
    elif val in unknown :
        colmed_spec.append('Unknown')
Diabetes3['medical_specialty'] = colmed_spec
```

## Create new column for medical_specialty to colmed_spec
```
Diabetes3['colmed_spec'] = Diabetes3['medical_specialty'].copy()
```

## Convert colmed_spec to string
```
Diabetes3[colmed_spec] = Diabetes3['colmed_spec'].astype('str')
```

## Convert numerical to string data
```
Diabetes_cat_num = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
Diabetes3[Diabetes_cat_num] = Diabetes3[Diabetes_cat_num].astype('str')
```

## Mix Diabetes_num+Diabetes_cal+colmed_spec
Now we merge all the categories together.
```
Diabetes_all = pd.get_dummies(Diabetes3[Diabetes_cal + Diabetes_cat_num + ['colmed_spec']],drop_first = True)
Diabetes_all.head
Diabetes3 = pd.concat([Diabetes3,Diabetes_all], axis=1)
```
## Save column names
```
Diabetes_all_cat = list(Diabetes_all.columns)
```

## Check medical_specialty
Check the total value of each group data
```
Diabetes3.medical_specialty.nunique()
Diabetes3.groupby('medical_specialty').size().sort_values(ascending = False)
```

## Create new column for age_group
Group Age to 3 ranges.
```
age_group = {'[0-10)':1,
             '[10-20)':1,
             '[20-30)':2,
             '[30-40)':2,
             '[40-50)':2,
             '[50-60)':2,
             '[60-70)':3,
             '[70-80)':3,
             '[80-90)':3,
             '[90-100)':3}
Diabetes3['age_mix'] = Diabetes3.age.replace(age_group)

## Keep track for weight
```
Diabetes3['weight'] = Diabetes3.weight.notnull().astype('int')
```

## Keep track age_mix
```
Diabetes3_extra = ['age_mix','weight']
```

## Create new dataframe
Now we created 76 features for the machine learning model.
8 numerical features
67 categorical features
1 extra feature
```
Diabetes3_use =  Diabetes_num + Diabetes_all_cat + Diabetes3_extra
Diabetes3_data = Diabetes3[Diabetes3_use + ['readmitted_in']]
```

## Extract 30% of data
Next step we will build Training/Validation/Test of data samples
```
Diabetes3_data = Diabetes3_data.sample(n = len(Diabetes3_data), random_state = 42)
Diabetes3_data = Diabetes3_data.reset_index(drop = True)
Diabetes3_valid_test = Diabetes3_data.sample(frac=0.30,random_state=42)
print('Split size: %.3f'%(len(Diabetes3_valid_test)/len(Diabetes3_data)))
```
## Training Data of 30%
```
Diabetes3_train_all = Diabetes3_data.drop(Diabetes3_valid_test.index)
```

## Extract 50% of data
```
Diabetes3_test = Diabetes3_valid_test.sample(frac=0.50,random_state=42)
```
## Training Data of 50%
```
Diabetes3_valid = Diabetes3_valid_test.drop(Diabetes3_test.index)
Diabetes3_train_all=Diabetes3_data.drop(Diabetes3_valid_test.index)
```

## Let's verify that we used all the data.
```
print('all samples (n = %d)'%len(Diabetes3_data))
```

## Split the training data into positive and negative
We will create a balanced dataset for training and testing that has 50% each rows
```
rows_pos = Diabetes3_train_all.readmitted_in == '1'
Diabetes3_pos = Diabetes3_train_all.loc[rows_pos]
Diabetes3_neg = Diabetes3_train_all.loc[~rows_pos]
```

## Merge the data
```
Diabetes3_train = pd.concat([Diabetes3_pos, Diabetes3_neg.sample(n = len(Diabetes3_pos), random_state = 42)],axis = 0)
```

## Shuffle the order of training data
```
Diabetes3_train = Diabetes3_train.sample(n = len(Diabetes3_train), random_state = 42).reset_index(drop = True)
```

## Check balanced prevalence
```
Diabetes3_train['readmitted_in'] = Diabetes3_train['readmitted_in'].astype('int')
print('Train balanced prevalence(n= %d):%.3f'%(len(Diabetes3_train), calc_prevalance(Diabetes3_train.readmitted_in.values)))
```

## Create matrix X and vector y
```
X_train = Diabetes3_train[Diabetes3_use].values
X_train_all = Diabetes3_train_all[Diabetes3_use].values
X_valid = Diabetes3_valid[Diabetes3_use].values
## Vector y
y_train = Diabetes3_train['readmitted_in'].values
y_valid = Diabetes3_valid['readmitted_in'].values.astype(int)

print('Training All Shape:',X_train_all.shape)
print('Training Shape:',X_train.shape,y_train.shape)
print('Validation Shape:',X_valid.shape,y_valid.shape)
```

## Fit X_train_all
This step will scale dataset which removes the mean and scales to unit variance.
```
scaler = StandardScaler()
scaler.fit(X_train_all)
```

## Save for the test data
We will scale the test data, so using a package 'pickle'
```
scalerfile = 'scaler.sav'
pickle.dump(scaler,open(scalerfile, 'wb'))
scaler = pickle.load(open(scalerfile, 'rb'))
```

## Transform datasets
```
X_train_t = scaler.transform(X_train)
X_valid_t = scaler.transform(X_valid)
```

# Evaluate the performance of the model
```
def calc_specificity(y_actual, y_pred, thresh):
	return sum((y_pred < thresh) & (y_actual  ==  0))/sum(y_actual == 0)
	
def print_report(y_actual, y_pred, thresh):
	auc = roc_auc_score(y_actual, y_pred)
	accuracy = accuracy_score(y_actual, (y_pred > thresh))
	recall = recall_score(y_actual, (y_pred > thresh))
	precision = precision_score((y_actual, (y_pred > thresh))
	specificity = calc_specificity(y_actual, y_pred, thresh)
	print('AUC:%.3f'%auc)
	print('accuracy:%.3f'%accuracy)
	print('recall:%.3f'%recall)
	print('precision:%.3f'%precision)
	print('specificity:%.3f'%specificity)
	print('prevalence:%.3f'%calc_prevalance(y_actual))
	print(' ')
	return auc, accuracy, recall, precision, specificity
```

We set threshold at 0.5 to label a predicted sample as positive
```
thresh = 0.5
```

## KNN
```
knn = KNeighborsClassifier(n_neighbors = 100)
knn.fit(X_train_t, y_train)

y_train_preds = knn.predict_proba(X_train_t)[:,1]
y_valid_preds = knn.predict_proba(X_valid_t)[:,1]

print('KNN')
print('Training:')
knn_train_auc, knn_train_accuracy, knn_train_recall, \
    knn_train_precision, knn_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
knn_valid_auc, knn_valid_accuracy, knn_valid_recall, \
    knn_valid_precision, knn_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
```

## LogisticRegression
```
lr=LogisticRegression(random_state=42)
lr.fit(X_train_t,y_train)

y_train_preds = lr.predict_proba(X_train_tf)[:,1]
y_valid_preds = lr.predict_proba(X_valid_tf)[:,1]

print('Logistic Regression')
print('Training:')
lr_train_auc, lr_train_accuracy, lr_train_recall, \
    lr_train_precision, lr_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
lr_valid_auc, lr_valid_accuracy, lr_valid_recall, \
    lr_valid_precision, lr_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
```

## Gradient boosting classifier
```
GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=42)
GBC = XGBClassifier()
GBC.fit(X_train_t, y_train)
y_valid = y_valid.astype('int')
```

## Plot Gradient boosting classifier graph
plot_tree(GBC)
plt.show()

## Check the validation score
score = GBC.score(X_valid, y_valid)
print(score)


print("Learning rate: ", 1.0)
print("Accuracy score (training): {0:.3f}".format(GBC.score(X_train_t, y_train)))
print("Accuracy score (validation): {0:.3f}".format(GBC.score(X_valid_t, y_valid)))

print("Confusion Matrix:")
print(confusion_matrix(y_valid, predictions))

print("Classification Report")
print(classification_report(y_valid, predictions))


fig = plt.figure(figsize = (8,5))
sns.countplot(x = 'number_inpatient', data = Diabetes3_data, hue = 'readmitted_in')
plt.legend(bbox_to_anchor=(1, 1))
plt.legend(['Not Admitted', 'Admitted'], title='Readmitted', loc='upper right')


###############################################