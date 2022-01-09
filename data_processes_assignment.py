import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from lifelines import KaplanMeierFitter
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from numpy import sum
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from warnings import simplefilter
import joblib

pd.set_option('display.expand_frame_repr', False)
simplefilter("ignore")

#---------------------------------------------------------------------------------------------------
# AUXILIAR METHODS
#---------------------------------------------------------------------------------------------------

# Returns filtered dataset
def get_wrapper_selection(model):
    # Wrapper FSS: Sequential Forward Selection with 10 folds
    sfs = SFS(model, k_features=9, forward=True, scoring = 'accuracy', cv = 10)
    sfs.fit(X_scaled, y)
    # Find subset of features with maximun average accuracy
    df_SFS_results = pd.DataFrame(sfs.subsets_).transpose()
    # Change type from object to float
    df_SFS_results["avg_score"] = df_SFS_results.avg_score.astype(float)
    # Get the index with the max accuracy
    max_index = df_SFS_results['avg_score'].idxmax()
    # Get the features that maximize the accuracy
    selected_features = list(df_SFS_results['feature_names'][max_index])
    print('Selected features:',selected_features)
    # Filtered dataset depending on the model
    X_fss = X_scaled[selected_features]
    return X_fss

# Returns total confusion matrix, average accuracy 
# and average standard deviation after all the cross-validation runs
def get_metrics(model,X_model):
    X_model = X_model.to_numpy()
    conf_matrix_list_of_arrays = []
    scores = []
    for train_index, test_index in cv.split(X_model):
       X_train, X_test = X_model[train_index], X_model[test_index]
       y_train, y_test = y[train_index], y[test_index]
       score = model.fit(X_train, y_train).score(X_test, y_test)
       conf_matrix = confusion_matrix(y_test, model.predict(X_test))
       scores.append(score)
       conf_matrix_list_of_arrays.append(conf_matrix)
    # Total confusion  is the sum of each cross-validation confusion matrix 
    mean_of_conf_matrix_arrays = sum(conf_matrix_list_of_arrays, axis=0)
    # Average accuracy
    avg_score = mean(scores)
    # Average standard deviation
    std_score = std(scores)
    return avg_score,std_score,mean_of_conf_matrix_arrays

# Print model metrics
def print_model_metrics(model):
    # Get the set of features that maximize accuracy of the model using 10-fold CV
    X_model = get_wrapper_selection(model)
    # Get mean accuracy, standard deviation and total confusion matrix using 10-fold CV  
    mean_model, std_model, total_confusion_matrix_model = get_metrics(model,X_model)
    print('Mean accuracy: %.3f (std deviation: %.3f)' % (mean_model, std_model))
    print('Total confusion matrix:')
    print(total_confusion_matrix_model)
    
#---------------------------------------------------------------------------------------------------
# INITIAL DATA EXPLORATION
#---------------------------------------------------------------------------------------------------

# Load data from CSV
df = pd.read_csv("COVID19_data.csv")

# 10 first rows
print(df.head(10)) # The dataframe has 13 columns

'''
There are many missing values in DESTINATION column
There are many 0's in medical parameters -> anormalities
GLUCOSE variable has so many 0's -> should be deleted
'''

# Type of each column
print(df.dtypes) # 3 string variables (SEX, EXITUS and DESTINATION) and 10 numeric variables 

# Descriptive analysis
print('\nNumber of instances:',len(df.index)) # The dataframe has 2054 instances
print('\nNon missing values per column')
print(df.count())

'''
Apparently, the following variables have missing values:
    AGE -> 4 missing values
    SEX -> 2 missing values
    EXITUS -> 41 missing values
    DESTINATION -> 1383 missing values
'''

print('\nSome statistics for numeric columns:') 
print(df.describe())             

'''
Maximum age is 189 -> outlier
Minimum temperature is 0 -> outlier
Mean temperature is low due to the huge number of 0's
Maximum heart rate is 593 -> outlier
Minimum heart rate is 0 -> outlier
Maximum glucose is 448 -> outlier
Minimum saturation O2 is 0 -> outlier
Mean saturation O2 is low due to the huge number of 0's
Maximum systolic blood pressure is 772 -> outlier
Minimum systolic blood pressure is 0 -> outlier
Maximum diastolic blood pressure is 845 -> outlier
Minimum diastolic blood pressure is 0 -> outlier
''' 

print('\nPlot correlation matrix')
corrMatrix = df.corr()
plt.figure(figsize=(19, 6))
sn.heatmap(corrMatrix, annot=True)
plt.show()

'''
AGE and ID are very correlated (0.97) -> This correlation does not make sense since ID is a unique patient identifier 
                                         and does not give relevant information to the model . 
                                         However, it can be seen that both ID and AGE are in ascending order,
                                         and that is the reason of the high correlation coefficient                
                                                             
TEMP and HEART_RATE are very correlated (0.74) ->It seems that the higher the temperature, 
                                                 the higher the heart rate of the patients.

TEMP and SAT_O2 are very correlated (0.84) ->It seems that the higher the temperature, 
                                                 the higher the O2 saturation of the patients.

HEART_RATE and SAT_O2 are very correlated (0.79)

BLOOD_PRES_SYS and BLOOD_PRES_DIAS are very correlated as expected (0.81)                                                               
                                                             
'''

print('\nRow filtering by EXITUS column')
print('EXITUS equal to NO:')
print(df[df['EXITUS'] == 'NO'])
print('EXITUS equal to YES:')
print(df[df['EXITUS'] == 'YES'])  

print('\nGrouping by EXITUS column:')
grouped_df = df.groupby('EXITUS')
print(grouped_df.mean())  

#---------------------------------------------------------------------------------------------------
# INITIAL DATA VISUALIZATION
#---------------------------------------------------------------------------------------------------

numeric_parameters = ['AGE', 'DAYS_HOSPITAL','DAYS_ICU', 'TEMP', 
                      'HEART_RATE', 'SAT_O2', 'BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS'] 

# Plotting univariate histograms and boxplots to check normality and outliers
print('\nUnivariate plots:')
plt.clf()
for column in numeric_parameters:
    plt.figure(figsize=(19, 6))
    plt.subplot(1,2,1)
    plt.axvline(df[column].mean(), color='k', linestyle='dashed', linewidth=1)
    df[column].plot(kind='hist', title = 'Histogram')
    plt.subplot(1,2,2)
    df[column].plot(kind='box', title = 'Boxplot')
    plt.suptitle(column + ' variable')
    plt.show()

'''
Looking at the histograms it can be seen that not all variables have a normal distribution. 
Therefore, all the data will be scaled.
Regarding outliers:
    AGE ->  1 outlier of almost 200 can be seen looking at the boxplot.
          
    DAYS_HOSPITAL -> Looking at the boxplot, it seems that it is considered outlier 
                     to be more than approximately 20 days in the hospital. 
    DAYS_ICU -> Looking at the boxplot it seems that it is considered outlier 
                to be more than 1 day approximately in the ICU.                    
    TEMP -> Looking at the boxplot we see various temperatures of 0º. 
            Temperatures of more than 39º approximately are also considered outliers.  
    HEART RATE -> Looking at the boxplot we see several heart rates of 0. 
                  Heart rates of more than 150 approximately are also considered outliers. 
                  1 isolated value of almost 600 is also observed.  
    SAT_O2 -> Looking at the boxplot we see that O2 saturations of less 
              than approximately 60 are considered outliers.
    BLOOD_PRES_SYS -> Looking at the boxplot, 1 outlier of almost 800 is observed. 
    BLOOD_PRES_DIAS -> Looking at the boxplot, 2 outliers are observed above 600.  
'''

# Bivariate plots in order to see distributions and relations between variables
print('\nBivariate plots:')
plt.clf()
for a in numeric_parameters:
    for b in numeric_parameters[1:]:
        if(a != b):
           df.plot(kind='scatter', x=a, y=b, title=a +' VS '+ b)

print('\nPlotting dataframe grouped by EXITUS:')
plt.clf()           
grouped_df.mean()['AGE'].plot(kind='bar', title='Mean age of the patients per group')
plt.show()
grouped_df.mean()['SAT_O2'].plot(kind='bar', title='Mean O2 saturation of the patients per group')
plt.show()
grouped_df.mean()['BLOOD_PRES_SYS'].plot(kind='bar', title='Mean systolic blood pressure of the patients per group')
plt.show()
grouped_df.mean()['DAYS_ICU'].plot(kind='bar', title='Mean days in ICU of the patients per group')
plt.show()

#---------------------------------------------------------------------------------------------------
# DATA PREPROCESSING
#---------------------------------------------------------------------------------------------------

# Delete ID column since it does not give relevant information to the model
df = df.drop(['ID'], axis=1)

# Delete DESTINATION column since it has more missing values than real values
# Delete GLUCOSE column since it has more 0's than real values
df = df.drop(['GLUCOSE', 'DESTINATION'], axis=1)
print("\nCheck if there are missing values")
print(df.isnull().any())
print("Done") 

# Delete missing values found in columns AGE, SEX and EXITUS
# Not necessary to replace them since it is a very low number over the total of instances
df = df.dropna(subset=['AGE','SEX','EXITUS'])

# If all medical parameters for a patient are 0, then exclude the corresponding row since 
# those patients do not give much information to the model
df = df[(df['TEMP'] > 0) | (df['HEART_RATE'] > 0) | (df['SAT_O2'] > 0)| (df['BLOOD_PRES_SYS'] > 0) | 
        (df['BLOOD_PRES_DIAS'] > 0)]

# Now replace 0's in medical parameters with the median (not mean due to the considerable number of 0's)
medical_parameters = ['TEMP', 'HEART_RATE', 'SAT_O2', 'BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS']
print("\nMedians:")
for parameter in medical_parameters:
    median = df[parameter].median()
    print(parameter,median)
    df[parameter].replace(0, median, inplace=True)
    
print("\nCheck again if there are missing values")
print(df.isnull().any())
print("Done") 

# Identify and delete outliers  
print("\nLooking for outliers")
print("IQRs: ")
for parameter in numeric_parameters:
    Q1=df[parameter].quantile(0.25)
    Q3=df[parameter].quantile(0.75)
    IQR=Q3-Q1
    lowqe_bound=Q1 - 1.5 * IQR
    upper_bound=Q3 + 1.5 * IQR
    print(parameter,lowqe_bound,upper_bound)
print("Done")    

'''
    AGE -> All the ages between the IQR
    DAYS_HOSPITAL -> There are patients who spent more than 19 days in the hospital, 
                     although this is an atypical situation, it is not impossible. 
                     Therefore, these patients will not be removed from the dataset.
    DAYS_ICU ->  There are patients who spent more than 0 days in ICU, 
                 this is not impossible. 
                 Therefore, these patients will not be removed from the dataset.                  
    TEMP -> There are patients with less than 34.85 degrees and with more than 38.45 degrees, 
            But, the min and max values are consistent and possible.
            Therefore, these patients will not be removed from the dataset.  
    HEART RATE -> There are two patients with values of 21 and 593. These are very extreme values.
                  Therefore, these patients are going to be removed from the dataset. 
    SAT_O2 -> There is a patient with a value of 10 (very extreme value). 
              Normal values are between 95 and 100. Therefore, this patient will be removed
    BLOOD_PRES_SYS -> Deleted patients with values between 10 and 26. 
                     Also deleted a patient with a value of 772.
    BLOOD_PRES_DIAS -> Deleted a patients with value 11. 
                     Also deleted patients with values of 741 and 845.              
''' 
                
print("\nFilter selected outliers:")
df = df[(df['HEART_RATE'] != 21) & (df['HEART_RATE'] != 593) & 
        (df['SAT_O2'] != 10) & (df['BLOOD_PRES_SYS'] > 26)
        & (df['BLOOD_PRES_SYS'] != 772) &
        (df['BLOOD_PRES_DIAS'] < 741) & (df['BLOOD_PRES_DIAS'] != 11)]
print("Done")

df = df.reset_index(drop=True)

# Encoding SEX and EXITUS variables
print("\nEncoding nominal variables:")
le = LabelEncoder()
sex_encoded  = le.fit_transform(df['SEX'])
# FEMALE -> 0 ; MALE -> 1
df['SEX'] = sex_encoded
exitus_encoded = le.fit_transform(df['EXITUS'])
# NO -> 0 ; YES -> 1
df['EXITUS'] = exitus_encoded
print(df.head(10))
print("Done")

# NOTE: Scaled dataframe later (in survival prediction)

#---------------------------------------------------------------------------------------------------
# DATA EXPLORATION AFTER PREPROCESSING
#---------------------------------------------------------------------------------------------------

# Check changed types
print("\nChecking data types:")
print(df.dtypes) # All the columns are numeric now

# Descriptive analysis
print('\nNumber of instances:',len(df.index)) # Now the dataframe has 1631 instances
print('\nNon missing values per column')
print(df.count()) # No missing values now

print('\nSome statistics for numeric columns:') 
print(df.describe())

print('\nEXITUS values:') 
print(df['EXITUS'].value_counts())

print('\nPlot correlation matrix')
plt.clf()
corrMatrix = df.corr()
plt.figure(figsize=(19, 6))
sn.heatmap(corrMatrix, annot=True)
plt.show()

'''
After preprocessing, it can be seen that the correlations between the variables have decreased. 
This is possibly due to the removal of outliers and the replacement of missing values
with the corresponding medians.

The most notable correlations are:
    AGE with EXITUS (0.39) -> the older the patient, the greater the possibility of not surviving COVID-19
    DAYS_HOSPITAL with DAYS_ICU (0.36) -> the more days a patient is in the hospital, the greater the possibility 
                                          of going to the ICU
    EXITUS with SAT_O2 (-0.35) -> the lower the oxygen saturation level, the greater the possibility 
                                  of not surviving COVID-19
    BLOOD_PRES_SYS with BLOOD_PRES_DIAS (0.55) -> they are correlated as expected
'''

print('\nGrouping by EXITUS column:')
grouped_df = df.groupby('EXITUS')
print(grouped_df.mean()) 

'''
It seems that patients who survive COVID-19 present on average: 
younger age, higher O2 saturation level and 
spend fewer days in the ICU than those patients who do not survive. 
'''              

#---------------------------------------------------------------------------------------------------
# DATA VISUALIZATION AFTER PREPROCESSING
#---------------------------------------------------------------------------------------------------

# Adding encoded SEX variable in order to visualize it together with other numeric variables
numeric_parameters.append('SEX')

# Checking distributions again after deleting some outliers
print('\nHistograms:')
plt.clf()
for column in numeric_parameters:
    plt.axvline(df[column].mean(), color='k', linestyle='dashed', linewidth=1)
    df[column].plot(kind='hist', title = column)
    plt.show()
    
# Bivariate plots
print('\nEXITUS vs rest of numeric variables:')
plt.clf()
for a in numeric_parameters:
    df.plot(kind='scatter', x='EXITUS', y=a, title='EXITUS VS '+ a)
 
'''
It can be seen that patients who do not survive tend to be older. In another plot, it can be seen 
that patients who do not survive tend to spend more days in the ICU. In addition, 
it is observed that these patients tend to have a lower level of O2 saturation.
'''

# Bivariate plots colored by EXITUS variable
print('\nBivariate plots colored by EXITUS variable:')
plt.clf()
for a in numeric_parameters:
    for b in numeric_parameters[1:]:
        if(a != b):
           df.plot(kind='scatter', x=a, y=b, title=a +' VS '+ b, c='EXITUS', cmap="RdYlGn")
           
'''
It can be seen how those patients who spend more days in the ICU 
and have high age do not usually survive COVID-19. 

In another plot it can be observed that those patients with high age and 
low O2 saturation level do not usually survive COVID-19 either.
'''
    
print('\nPlotting dataframe grouped by EXITUS:')
plt.clf()           
grouped_df.mean()['AGE'].plot(kind='bar', title='Mean age of the patients per group')
plt.show()
grouped_df.mean()['SAT_O2'].plot(kind='bar', title='Mean O2 saturation of the patients per group')
plt.show()
grouped_df.mean()['DAYS_ICU'].plot(kind='bar', title='Mean days in ICU of the patients per group')
plt.show()

print('\nPlotting survival curves:')
plt.clf()  
# Survival curves (event = EXITUS ; duration = DAYS_HOSPITAL)
kmf = KaplanMeierFitter() 
kmf.fit(df['DAYS_HOSPITAL'], df['EXITUS'])
plt.figure(figsize=(8,4))
kmf.plot()
plt.xlabel("Survival time (days in hospital)")
plt.ylabel("Survival probability")
plt.title("Kaplan-Meier curve");
plt.show()

print('\nSurvival probability by days in hospital:')
print(kmf.survival_function_)

# Survival curves (event = EXITUS ; duration = DAYS_ICU)
kmf.fit(df['DAYS_ICU'], df['EXITUS'])
plt.figure(figsize=(8,4))
kmf.plot()
plt.xlabel("Survival time (days in ICU)")
plt.ylabel("Survival probability")
plt.title("Kaplan-Meier curve");
plt.show()

print('\nSurvival probability by days in ICU:')
print(kmf.survival_function_)

# Survival curves by AGE -> need to discretize AGE 
aux_df = df.copy()	
aux_df['AGE_BINS']=pd.cut(x=aux_df['AGE'], bins=3,labels=["Small", "Middle", "Older"])

kmf_small = KaplanMeierFitter()
kmf_middle = KaplanMeierFitter()
kmf_older = KaplanMeierFitter()

small_df = aux_df[aux_df['AGE_BINS'] == 'Small']
middle_df = aux_df[aux_df['AGE_BINS'] == 'Middle']
older_df = aux_df[aux_df['AGE_BINS'] == 'Older']

kmf_small.fit(small_df['DAYS_ICU'], small_df['EXITUS'],label = 'Small age')
kmf_middle.fit(middle_df['DAYS_ICU'], middle_df['EXITUS'],label = 'Middle age')
kmf_older.fit(older_df['DAYS_ICU'], older_df['EXITUS'],label = 'Older age')

plt.figure(figsize=(8,4))
kmf_small.plot()
kmf_middle.plot()
kmf_older.plot()
plt.xlabel("Survival time (days in ICU)")
plt.ylabel("Survival probability")
plt.title("Kaplan-Meier curves by AGE");
plt.show()

# Survival curves by SAT_O2 -> need to discretize SAT_O2 
aux_df['SAT_O2_BINS']=pd.cut(x=aux_df['SAT_O2'], bins=2,labels=["Low", "High"])

kmf_low = KaplanMeierFitter()
kmf_high = KaplanMeierFitter()

low_df = aux_df[aux_df['SAT_O2_BINS'] == 'Low']
high_df = aux_df[aux_df['SAT_O2_BINS'] == 'High']

kmf_low.fit(low_df['DAYS_ICU'], low_df['EXITUS'],label = 'Low O2 saturation level')
kmf_high.fit(high_df['DAYS_ICU'], high_df['EXITUS'],label = 'High O2 saturation level')

plt.figure(figsize=(8,4))
kmf_low.plot()
kmf_high.plot()
plt.xlabel("Survival time (days in ICU)")
plt.ylabel("Survival probability")
plt.title("Kaplan-Meier curves by SAT_O2");
plt.show()

'''
Looking at the survival curves, it is observed that the more days the patients are 
in the hospital or ICU, the lower the probability of surviving COVID-19.

It is also observed that, being the same number of days in the ICU, 
younger patients with a higher level of O2 saturation are more 
likely to survive the disease.
'''

#---------------------------------------------------------------------------------------------------
# CONCLUSIONS OF SURVIVAL ANALYSIS
#---------------------------------------------------------------------------------------------------

'''
After the analysis carried out, it seems that what most influences the survival of a patient is his age. 
In addition, the level of O2 saturation, and the days in ICU are other factors that also seem to influence survival.

Thus, it seems that those older patients who have a low level of O2 saturation 
or spend many days in the ICU (or both together) have a very high probability of not surviving COVID-19.
'''

#---------------------------------------------------------------------------------------------------
# SURVIVAL PREDICTION ~ Training and evaluation of different supervised classification models 
#---------------------------------------------------------------------------------------------------

'''
Next, different supervised classification models will be trained and evaluated by cross-validation 
(10-fold) to try to predict survival.
Before training, a wrapper filtering will be applied in each of the models to eliminate those variables 
that are not influential or important for survival (in each corresponding model) and, thus, obtain better results.
'''

# Separate predictor variables and target variable
X = df.drop('EXITUS', axis=1)
y = df['EXITUS']

# Scale data
print("\nScaling data:")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,index=X.index, columns=X.columns)
print(X_scaled.head(10))
print("Done")

# Prepare the cross-validation procedure
cv = KFold(n_splits=10,shuffle=True, random_state=1)

print('\nLogistic Regression model')
print('------------------------------')

# Create a Logistic Regression model
logistic = LogisticRegression(random_state=1)
print_model_metrics(logistic)

'''
Selected features: ['AGE', 'DAYS_ICU', 'HEART_RATE', 'SAT_O2']
Mean accuracy: 0.863 (std deviation: 0.026)
Total confusion matrix:
[[1355   11]
 [ 212   53]]
'''

print('\nDecision Tree model')
print('------------------------------')

# Create a Decision Tree model
tree_m = tree.DecisionTreeClassifier(random_state=1)
print_model_metrics(tree_m)

'''
Selected features: ['DAYS_HOSPITAL', 'SAT_O2']
Mean accuracy: 0.850 (std deviation: 0.017)
Total confusion matrix:
[[1310   56]
 [ 188   77]]
'''

print('\nSupport Vector Machine model')
print('------------------------------')

# Create a Support Vector Machine model
svc= svm.SVC(gamma='auto',random_state=1)
print_model_metrics(svc)

'''
Selected features: ['AGE', 'DAYS_ICU', 'SAT_O2']
Mean accuracy: 0.848 (std deviation: 0.024)
Total confusion matrix:
[[1361    5]
 [ 243   22]]
'''

print('\nGaussian Naive Bayes model')
print('------------------------------')

# Create a Gaussian Naive Bayes model
gnb = GaussianNB()
print_model_metrics(gnb)

'''
Selected features: ['SEX', 'HEART_RATE', 'SAT_O2', 'BLOOD_PRES_DIAS']
Mean accuracy: 0.858 (std deviation: 0.024)
Total confusion matrix:
[[1330   36]
 [ 195   70]]
'''

print('\nRandom Forest model')
print('------------------------------')

# Create a Random Forest model
rft = RandomForestClassifier(n_estimators=10,random_state=1)
print_model_metrics(rft)

'''
Selected features: ['DAYS_HOSPITAL', 'DAYS_ICU', 'SAT_O2']
Mean accuracy: 0.853 (std deviation: 0.021)
Total confusion matrix:
[[1309   57]
 [ 183   82]]
'''

print('\nMulti-layer Perceptron model')
print('------------------------------')

# Create a Multi-layer Perceptron model
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),random_state=1)
print_model_metrics(mlp)

'''
Selected features: ['AGE', 'DAYS_HOSPITAL', 'DAYS_ICU', 'TEMP', 'HEART_RATE', 'SAT_O2', 'BLOOD_PRES_SYS']
Mean accuracy: 0.867 (std deviation: 0.021)
Total confusion matrix:
[[1323   43]
 [ 174   91]]
'''

print('\nLinear Discriminant Analysis model')
print('------------------------------')

# Create a Linear Discriminant Analysis model
lda = LinearDiscriminantAnalysis()
print_model_metrics(lda)

'''
Selected features: ['DAYS_HOSPITAL', 'DAYS_ICU', 'TEMP', 'HEART_RATE', 'SAT_O2', 'BLOOD_PRES_SYS']
Mean accuracy: 0.864 (std deviation: 0.028)
Total confusion matrix:
[[1349   17]
 [ 205   60]]
'''

#---------------------------------------------------------------------------------------------------
# CONCLUSIONS OF SURVIVAL PREDICTION
#---------------------------------------------------------------------------------------------------

'''
After training several supervised classification models, it could be thought that the level of O2 saturation 
is more important than the age of a patient in predicting survival 
(contrary to what was previously observed in the survival analysis) since it is 
selected in the 7 trained models, while age is only selected in 3 of them. 
It can also be seen that the heart rate of a patient is of certain relevance as it is selected in 4 models.

Regarding the performance of the models, they all have a fairly similar accuracy (between 84% and 87%). 
However, in this case it is important to take into account the number of false negatives when choosing a model, 
that is, the number of patients that are classified as survivors but do not actually survive the disease.
In most of the models, it is observed that the number of false negatives is much higher than 
the number of false positives. Possibly this is due to the fact that the dataset is unbalanced, 
that is, in the preprocessed dataset there are only 265 non-surviving patients 
compared to 1366 patients who have survived the disease. 
Hence, it seems more difficult for the models to predict when a patient will not survive.  

The Multilayer Perceptron model (neural network) seems to be the most feasible to choose 
since it has the least number of false negatives (174) and also the best accuracy (0.867). 
However, neural networks consume many resources and are little or no interpretable (black-box models).
The second model with the least false negatives (183) is the Random Forest. 
Random Forests are scalable and their most significant variables can be easily identified. 
In addition, this model also has good accuracy (0.853).

Therefore, the model that seems the most optimal for this problem 
(and the one that would not be bad to choose) is the Random Forest.
Finally, in future studies it would be interesting to balance the dataset 
as much as possible in order to retrain the models and obtain better results.
'''   

#---------------------------------------------------------------------------------------------------
# APP PROTOTYPE
#---------------------------------------------------------------------------------------------------

'''
A simple prototype has been developed with the selected model (Random Forest). 
This prototype allows receiving data about a patient and predicting whether
the patient will survive COVID-19 or not.

How to run:
    1.- Go to the root folder of the prototype -> cd prototype
    2.- Run the script 'main.sh' in a shell -> ./main.sh
    3.- Follow the steps of the script (enter parameters of a patient, etc)
    4.- The diagnosis can be found under the line 'DIAGNOSIS:'
'''

pkl_dir = 'prototype/resources/'

# Save scaler for app prototype
scaler_filename = "scaler.pkl"
joblib.dump(scaler, pkl_dir + scaler_filename) 

# Save model for app prototype
model_filename = "random_forest.pkl"
joblib.dump(rft, pkl_dir + model_filename) 
