
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix , f1_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("database.csv")
df = df.loc[:,["Gender","Married", "Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount",
               "Loan_Amount_Term","Credit_History","Property_Area","Loan_Status"]]


# Basic stats
print("Number of rows : {}".format(df.shape[0]))
print()
print("Display of dataset: ")
print(df.head())
print()

print("Basics statistics: ")
data_desc = df.describe(include='all')
print(data_desc)
print()

print("Percentage of missing values: ")
a =df.isnull().sum()/len(df)*100
a.sort_values(ascending=False)


df = df.dropna()

df['Dependents'] = df['Dependents'].replace('3+', '3').astype('int')


matrix = df.corr()
print("Correlation matrix is : ")
print(matrix)

sns.heatmap(matrix, annot=True) #show correlation matrix
plt.show()

sns.countplot( x = 'Loan_Status',data=df)
print ('There is {} % of loan accepted'.format(np.round(len(df[df['Loan_Status'] == 'Y'])/len(df['Loan_Status'])*100, 2)))
print ('There is {} % of loan rejected'.format(np.round(len(df[df['Loan_Status'] == 'N'])/len(df['Loan_Status'])*100 , 2)))


pie_chart_data = df.groupby('Loan_Status')['Gender'].value_counts()
pie_chart_data

explode = (0.1,0.1,0.1,0.1)
plt.figure()
plt.pie(pie_chart_data.values, labels=pie_chart_data.index,  
       autopct='%1.1f%%',
       shadow=True, 
       startangle=90,
        explode=explode,
       radius=1.5
       )
plt.title('Loan status per "gender"', loc ='left', color ='black')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.show()


df['Gender'] = df['Gender'].astype('str')
df['Married'] = df['Married'].astype('str')
df['Education'] = df['Education'].astype('str')
df['Self_Employed'] = df['Self_Employed'].astype('str')
df['Credit_History'] = df['Credit_History'].astype('str')
df['Property_Area'] = df['Property_Area'].astype('str')


# Separate target variable Y from features X
print("Separating target variable from features...")

## Choose the columns you want to have as your features
features_list = df.columns[:-1]

X = df.loc[:,features_list] 
y = df.loc[:,"Loan_Status"]

#train test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=y)





numeric_features = [2, 5, 6, 7, 8] 

numeric_transformer = StandardScaler()


categorical_features = [0, 1, 3, 4, 9, 10]

categorical_transformer = OneHotEncoder()



feature_encoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),    
        ('num', numeric_transformer, numeric_features)])
    
X_train = feature_encoder.fit_transform(X_train)

#our model:
classifier = LogisticRegression()

classifier.fit(X_train, y_train) 

y_train_pred = classifier.predict(X_train)


X_test = feature_encoder.transform(X_test)

# Predictions on test set
y_test_pred = classifier.predict(X_test)

### Assessment of performances ###
print("--- Assessment of performances ---")

# Plot confusion matrix
cm = plot_confusion_matrix(classifier, X_train, y_train)
cm.ax_.set_title("Confusion matrix on train set ") 
plt.show()
print("accuracy-score on train set : ", classifier.score(X_train, y_train))


cm = plot_confusion_matrix(classifier, X_test, y_test)
cm.ax_.set_title("Confusion matrix on test set ")
plt.show()
print("accuracy-score on test set : ", classifier.score(X_test, y_test))

print("f1-score on training set: {} ".format(np.round(f1_score(y_train, y_train_pred, pos_label='Y') , 2)))
print()
print("f1-score on test set : {} ".format(np.round(f1_score(y_test, y_test_pred, pos_label='Y'), 2)))


# Check coefficients 

print("All transformers are: ", feature_encoder.transformers_)
print()

print("One Hot Encoder transformer is: ", feature_encoder.transformers_[0][1])
print()

# Print categories
categorical_column_names = feature_encoder.transformers_[0][1].categories_


numerical_column_names = X.iloc[:, numeric_features].columns 
print("numerical columns are: ", numerical_column_names)
print('categorical variables are: ', categorical_column_names)

categorical_column_names = np.concatenate(categorical_column_names)
all_column_names = np.append(categorical_column_names, numerical_column_names)
print("All column names are: ",all_column_names)
print()


feature_importance = pd.DataFrame({
    "feature_names": all_column_names,
    "coefficients": classifier.coef_.squeeze()
})
feature_importance.sort_values(by="coefficients", ascending=False)

# Set coefficient to absolute values to rank features
feature_importance["coefficients"] = feature_importance["coefficients"].abs()

# Visualize ranked features using seaborn
sns.catplot(x="feature_names", 
            y="coefficients", 
            data=feature_importance.sort_values(by="coefficients", ascending=False), 
            kind="bar",
            aspect=17/4)


# Build a Decision Tree
# Train model

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train) 

# Predictions on training set
y_train_pred = classifier.predict(X_train)


# Predictions on test set
y_test_pred = classifier.predict(X_test)

### Assessment of performances ###
print("--- Assessment of performances ---")

# Plot confusion matrix
cm = plot_confusion_matrix(classifier, X_train, y_train)
cm.ax_.set_title("Confusion matrix on train set ")
plt.show()
print("accuracy-score on train set : ", classifier.score(X_train, y_train))


cm = plot_confusion_matrix(classifier, X_test, y_test)
cm.ax_.set_title("Confusion matrix on test set ")
plt.show()
print("accuracy-score on test set : ", classifier.score(X_test, y_test))

print("f1-score on training set: {} ".format(np.round(f1_score(y_train, y_train_pred, pos_label='Y') , 2)))
print()
print("f1-score on test set : {} ".format(np.round(f1_score(y_test, y_test_pred, pos_label='Y'), 2)))

# Feature importance 
feature_importance = pd.DataFrame({
    "feature_names": all_column_names,
    "coefficients": classifier.feature_importances_
                                        
})

feature_importance

sns.catplot(x="feature_names", 
            y="coefficients", 
            data=feature_importance.sort_values(by="coefficients", ascending=False), 
            kind="bar",
            aspect=17/4)


#Random Forest:
classifier = RandomForestClassifier(n_estimators = 30)
classifier.fit(X_train, y_train)


# Predictions on training set
y_train_pred = classifier.predict(X_train)

# Predictions on test set
y_test_pred = classifier.predict(X_test)

#Model evaluation:
print("--- Assessment of performances ---")


# Plot confusion matrix
cm = plot_confusion_matrix(classifier, X_train, y_train)
cm.ax_.set_title("Confusion matrix on train set ")
plt.show()
print("accuracy-score on train set : ", classifier.score(X_train, y_train))


cm = plot_confusion_matrix(classifier, X_test, y_test)
cm.ax_.set_title("Confusion matrix on test set ")
plt.show()
print("accuracy-score on test set : ", classifier.score(X_test, y_test))

print("f1-score on training set: {} ".format(np.round(f1_score(y_train, y_train_pred, pos_label='Y') , 2)))
print()
print("f1-score on test set : {} ".format(np.round(f1_score(y_test, y_test_pred, pos_label='Y'), 2)))

# Feature importance 
feature_importance = pd.DataFrame({
    "feature_names": all_column_names,
    "coefficients": classifier.feature_importances_
                                        
})

feature_importance

# Visualize ranked features using seaborn
sns.catplot(x="feature_names", 
            y="coefficients", 
            data=feature_importance.sort_values(by="coefficients", ascending=False), 
            kind="bar",
            aspect=17/4)