# Introduction
This dataset describes 10 years (1999-2008) of clinical care at 130 US hospitals and combined delivery networks. 
It comes from the UCI machine learning repository ``` https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008 ```
Data were derived from the database for encounters that met the following guidelines.

## Data Set Information
(1) It is an inpatient encounter (a hospital admission).\
(2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.\
(3) The length of stay was at least 1 day and at most 14 days.\
(4) Laboratory tests were performed during the encounter.\
(5) Medications were administered during the encounter.

# Install
```
!pip install xgboost
!pip install graphviz
```
# Importing the libraries
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

# Data Exploration
Loading the dataset
```
Diabetes = pd.read_csv('diabetic_data.csv', index_col=False)
```

# Check column names
```
Diabetes.head()
```

# Check data type
```
Diabetes2.info()
```

# Check mean of each columns
```
Diabetes.describe()
```

# Count readmitted
```
Diabetes.readmitted.value_counts().plot(kind='barh', rot=0)
plt.title("Count Readmitted")
plt.xlabel("Amount")
plt.ylabel("Readmitted Day")
```
# Count race
```
Diabetes.race.value_counts().plot(kind='bar', rot=0, color='green')
plt.xticks(rotation=45, horizontalalignment="center")
plt.title("Count Race")
plt.xlabel("Race")
plt.ylabel("Amount")
```
# Count gender
```
Diabetes.gender.value_counts().plot(kind='bar', rot=0, color='blue')
plt.title("Count Gender")
plt.xlabel("Gender")
plt.ylabel("Amount")
```
# Count age
```
Diabetes.age.value_counts().plot(kind='barh', rot=0)
plt.title("Count Age")
plt.xlabel("Amount")
plt.ylabel("Age")
```

# Choose columns and removed examide+citoglipton
We will remove examide and citoglipton because these medicines don't have other status types except "No"
```
Diabetes2 = Diabetes[['race','gender','age','weight','admission_type_id','discharge_disposition_id','admission_source_id', 
                      'time_in_hospital', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications',
                      'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
                      'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                      'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                      'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                      'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
                      'readmitted']]
```

# Remove ? in 4 columns
Remove unnecessary question masks from race, diag_1, diag_2 and diag_3
```
Diabetes2 = Diabetes2[(Diabetes2.race != "?")]
Diabetes2 = Diabetes2[(Diabetes2.diag_1 != "?")]
Diabetes2 = Diabetes2[(Diabetes2.diag_2 != "?")]
Diabetes2 = Diabetes2[(Diabetes2.diag_3 != "?")]
```

# Represent values “0” represents "No readmission or readmission after 30 days whereas 1 represents readmission within 30 days
The most important column is "readmitted", which tell us if the patient was hospitalized in 30 days, 
more than 30 days or not readmitted.
```
Diabetes2.loc[Diabetes2['readmitted'] == "<30", 'readmittedFL'] = '1'
Diabetes2.loc[Diabetes2['readmitted'] == ">30", 'readmittedFL'] = '0'
Diabetes2.loc[Diabetes2['readmitted'] == "NO", 'readmittedFL'] = '0'
```

# Copy table
```
Diabetes3 = Diabetes2.copy()
```

# Create column list of num
Create "Diabetes_num" to analyst in the next process
```
Diabetes_num = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
                 'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
```

# Convert Diabetes_num to string
```
Diabetes3[Diabetes_num] = Diabetes3[Diabetes_num].astype('str')
```

# Create column list of cal
Create "Diabetes_cal" to analyst in the next process
```
Diabetes_cal = ['gender','max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
                'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 
                'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted', 'readmitted_in']
```

# Convert Diabetes_cal to string
```
Diabetes3[Diabetes_cal] = Diabetes3[Diabetes_cal].astype('str')
```

# Replace '?'
We will replace '?' with 'Unknown' to make it easier to analyze.
```
Diabetes3['medical_specialty'] = Diabetes3['medical_specialty'].replace('?','Unknown')
```

# Group medical_specialty column
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

# Create new column for medical_specialty to colmed_spec
```
Diabetes3['colmed_spec'] = Diabetes3['medical_specialty'].copy()
```

# Convert colmed_spec to string
```
Diabetes3[colmed_spec] = Diabetes3['colmed_spec'].astype('str')
```

# Mix Diabetes_num+Diabetes_cal+colmed_spec
Now we merge all the categories together.
```
Diabetes_all = pd.get_dummies(Diabetes3[Diabetes_cal + ['colmed_spec']],drop_first = True)
Diabetes_all.head
Diabetes3 = pd.concat([Diabetes3,Diabetes_all], axis=1)
```
# Save column names
```
Diabetes_all_cat = list(Diabetes_all.columns)
```

# Check medical_specialty
Check the total value of each group data
```
Diabetes3.medical_specialty.nunique()
Diabetes3.groupby('medical_specialty').size().sort_values(ascending = False)
```

# Create new column for age_group
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
#Keep track age_mix
Diabetes3_extra = ['age_mix']
```

# Create new dataframe
Now we created 76 features for the machine learning model.
8 numerical features
67 categorical features
1 extra feature
```
Diabetes3_use =  Diabetes_num + Diabetes_all_cat + Diabetes3_extra
Diabetes3_data = Diabetes3[Diabetes3_use + ['readmitted_in']]
```

# Extract 30% of data
Next step we will build Training/Validation/Test of data samples
```
Diabetes3_data = Diabetes3_data.sample(n = len(Diabetes3_data), random_state = 42)
Diabetes3_data = Diabetes3_data.reset_index(drop = True)
Diabetes3_valid_test = Diabetes3_data.sample(frac=0.30,random_state=42)
print('Split size: %.3f'%(len(Diabetes3_valid_test)/len(Diabetes3_data)))
```
# Training Data of 30%
```
Diabetes3_train_all = Diabetes3_data.drop(Diabetes3_valid_test.index)
```

# Extract 50% of data
```
Diabetes3_test = Diabetes3_valid_test.sample(frac=0.50,random_state=42)
```
# Training Data of 50%
```
Diabetes3_valid = Diabetes3_valid_test.drop(Diabetes3_test.index)
```

# Split the training data into positive and negative
```
rows_pos = Diabetes3_train_all.readmitted_in == '1'
Diabetes3_pos = Diabetes3_train_all.loc[rows_pos]
Diabetes3_neg = Diabetes3_train_all.loc[~rows_pos]
```

# Merge the data
```
Diabetes3_train = pd.concat([Diabetes3_pos, Diabetes3_neg.sample(n = len(Diabetes3_pos), random_state = 42)],axis = 0)
```

# Shuffle the order of training data
```
Diabetes3_train = Diabetes3_train.sample(n = len(Diabetes3_train), random_state = 42).reset_index(drop = True)
```

# Calculate the prevalence of population that is readmitted with 30 days
```
def calc_prevalance(y_actual):
    return (sum(y_actual)/len(y_actual))
print('Prevalance:%.3f'%calc_prevalance(Diabetes2['readmitted_in'].values)) #Three decimal places
```

#Check balanced prevalence
```
Diabetes3_train['readmitted_in'] = Diabetes3_train['readmitted_in'].astype('int')
print('Train balanced prevalence(n= %d):%.3f'%(len(Diabetes3_train), calc_prevalance(Diabetes3_train.readmitted_in.values)))
```

# Create matrix X and vector y
```
X_train = Diabetes3_train[Diabetes3_use].values
X_train_all = Diabetes3_train_all[Diabetes3_use].values
X_valid = Diabetes3_valid[Diabetes3_use].values
#Vector y
y_train = Diabetes3_train['readmitted_in'].values
y_valid = Diabetes3_valid['readmitted_in'].values

print('Training All Shape:',X_train_all.shape)
print('Training Shape:',X_train.shape,y_train.shape)
print('Validation Shape:',X_valid.shape,y_valid.shape)
```

# Fit X_train_all
This step will scale dataset which removes the mean and scales to unit variance.
```
scaler = StandardScaler()
scaler.fit(X_train_all)
```

# Save for the test data
We will scale the test data, so using a package 'pickle'
```
scalerfile = 'scaler.sav'
pickle.dump(scaler,open(scalerfile, 'wb'))
scaler = pickle.load(open(scalerfile, 'rb'))
```

# Transform datasets
```
X_train_t = scaler.transform(X_train)
X_valid_t = scaler.transform(X_valid)
```

# Gradient boosting classifier
```
GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=3, random_state=42)
GBC = XGBClassifier()
GBC.fit(X_train_t, y_train)
y_valid = y_valid.astype('int')
```

# Plot Gradient boosting classifier graph
plot_tree(GBC)
plt.show()

# Check the validation score
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
#Prepare data for SVM
Diabetes_drug = Diabetes2[['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                      'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                      'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                      'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'diabetesMed']]


#Create dummy variables
nominal = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide', 'glipizide', 'glyburide', 
           'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 
           'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
           'metformin-pioglitazone']
Diabetes_drug = pd.get_dummies(Diabetes_drug,columns=nominal)


#Map data for SVM
Diabetes_drug['diabetesMed']=Diabetes_drug['diabetesMed'].map({'Yes': 0,'No': 1})
Diabetes_drug.head()

#Defining features and target variable for SVM
X_DR = Diabetes_drug.drop('diabetesMed', axis=1).values
y_DR = Diabetes_drug['diabetesMed'].values

#Split dataset into training set and test set for SVM
X_train, X_test, y_train, y_test = train_test_split(X_DR, y_DR, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)

#Standardize numeric variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1],
                     'C': [1]},
                    {'kernel': ['linear'], 'C': [1]}]

clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='f1_micro')
clf.fit(X_train, y_train)

##
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_true, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_true, y_pred))


#Fit SVC Class
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#Making Predictions
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#Count Group ReadmittedFL
colors = ['green','blue']
Not_Admitted = mpatches.Patch(color='green', label='Not_Admitted')
Admitted = mpatches.Patch(color='blue', label='Admitted')
ax = Diabetes2.readmittedFL.value_counts().plot(kind='bar', alpha=0.75, legend=False,
                                                rot=0, figsize=(8,5),color=colors)
patches, labels = ax.get_legend_handles_labels()
ax.set_title("Count of Patients who Readmitted", fontsize=12)
ax.set_ylabel("Number of Patients", fontsize=10)
ax.set_xlabel("Readmitted", fontsize=10)
# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.16, i.get_height()+550, \
            str(round((i.get_height()), 2)), fontsize=12, color='black')
ax.legend(handles=[Not_Admitted, Admitted])
 
        
#Count Readmission Days
ax_re = Diabetes2.readmitted.value_counts().plot(kind='bar', alpha=0.75, 
                                                rot=0, figsize=(8,5),)
ax_re.set_title("Count of Patients who Readmitted", fontsize=12)
ax_re.set_ylabel("Number of Patients", fontsize=10)
ax_re.set_xlabel("Readmitted", fontsize=10);
# set individual bar lables using above list
for i in ax_re.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax_re.text(i.get_x()+.12, i.get_height()+520, \
            str(round((i.get_height()), 2)), fontsize=12, color='black')

#Cound readmitted
Diabetes2.readmittedFL.value_counts().plot(kind='bar',rot=0)
Diabetes2.readmitted.value_counts().plot(kind='bar',rot=0)
Diabetes2.race.value_counts()

#Cound readmitted groupby race
Diabetes2.groupby(['readmittedFL']).race.value_counts()

#Cound readmitted groupby gender
Diabetes2.groupby(['readmittedFL']).gender.value_counts()

#Cound readmitted groupby age
Diabetes2.groupby(['readmittedFL']).age.value_counts()

#Split into input and output features
y = Diabetes2["readmittedFL"]
X = Diabetes2[["time_in_hospital","num_lab_procedures","num_procedures","num_medications",
               "number_outpatient","number_emergency","number_inpatient","number_diagnoses"]]
X.head()
y.head()

#Normalize the data attributes
x_normal = preprocessing.normalize(X)
pd.DataFrame(x_normal)
X_imputed= pd.DataFrame(x_normal, columns = X.columns)

#Univariate Histograms
data = pd.DataFrame(X_imputed)
fig = plt.figure(figsize = (10,7))
ax = fig.gca()
data.hist(ax = ax, bins='auto', color='#0504aa', alpha=0.7, rwidth=1)
plt.show()

#Pearson Corellation
X_corr = X_imputed.corr(method='pearson')

#Set label names
names = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
         'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
#Plot correlation matrix
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
cax = ax.matshow(X_corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.tick_params(labelsize=10)
plt.xticks(rotation=90)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.title('Correlation Matrix', fontsize=18)
plt.show()

#Check number of K
SSE = []
for cluster in range(1,15):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(X_imputed)
    SSE.append(kmeans.inertia_)
#Converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,15), 'SSE':SSE})
plt.figure(figsize=(10,5))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

#K-Means
X_array = X_imputed.to_numpy()
y_kmeans = KMeans(n_clusters=5).fit_predict(X_imputed)
plt.scatter(X_array[:, 0], X_array[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.title('Clusters of Diabetes')
#plt.xlabel('Annual Income(k$)')
#plt.ylabel('Spending Score(1-100')
plt.show()

#Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111) # 70% training and 30% test


#Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=None)

#Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
y_score = clf.score(X,y)

#Model performance
print(classification_report(y_test,y_pred))

Diabetes2.groupby('age').size()

#Consolidated from a 10-level factor to 3 and numeric
replaceDict = {'[0-10)' : 1,
'[10-20)' : 1,
'[20-30)' : 1,
'[30-40)' : 1, 
'[40-50)' : 1, 
'[50-60)' : 1,
'[60-70)' : 2, 
'[70-80)' : 2,
'[80-90)' : 3,
'[90-100)' : 3}

Diabetes2['age_adju'] = Diabetes2['age'].apply(lambda x : replaceDict[x])
print(Diabetes2['age_adju'].head())

#Encode field race
le = preprocessing.LabelEncoder()
for cat_var in ['race']:
    X[cat_var] = le.fit_transform(X[cat_var])
#Encode field age
for cat_var1 in ['age']:
    X[cat_var1] = le.fit_transform(X[cat_var1])
#Encode field max_glu_serum
for cat_var2 in ['max_glu_serum']:
    X[cat_var2] = le.fit_transform(X[cat_var2])
#Encode field A1Cresult
for cat_var3 in ['A1Cresult']:
    X[cat_var3] = le.fit_transform(X[cat_var3])
#Encode field A1Cresult
for cat_var4 in ['metformin']:
    X[cat_var4] = le.fit_transform(X[cat_var4])
#Encode field A1Cresult
for cat_var5 in ['insulin']:
    X[cat_var5] = le.fit_transform(X[cat_var5])


#Inverse field race
X_train.race = le.inverse_transform(X_train[cat_var])
#Inverse field age
X_train.age = le.inverse_transform(X_train[cat_var1]) 