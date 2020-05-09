import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/ziontaber/Downloads/seabornProjectCumulative/kiva_data.csv')
#print(df.info())

categorical_col = []
for column in df.columns:
    if df[column].dtype == object:
        categorical_col.append(column)


for i, column in enumerate(categorical_col, 1):
    plt.subplot(3, 3, i)
    g = sns.barplot(x='{}'.format(column), y='loan_amount', data=df)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.ylabel('loan_amount')
    plt.xlabel('{}'.format(column))
#plt.show()

label = LabelEncoder()
for column in categorical_col:
    df[column] = label.fit_transform(df[column])
x = df.drop('loan_amount', axis=1)
y = df.loan_amount
for column in x.columns:
    print(f"{column} : {x[column].unique()}")
    print("====================================")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

pk = df[df.country =='Pakistan']
ax = plt.subplots(figsize=(15, 10))
sns.barplot(data=pk, x="activity", y="loan_amount").set_title('Pakistan')

ky = df[df.country =='Kenya']
ax = plt.subplots(figsize=(15, 10))
sns.barplot(data=ky, x="activity", y="loan_amount").set_title('Kenya')

es = df[df.country =='El Salvador']
ax = plt.subplots(figsize=(15, 10))
sns.barplot(data=es, x="activity", y="loan_amount").set_title('El Salvador')

ph = df[df.country =='Philippines']
ax = plt.subplots(figsize=(15, 10))
sns.barplot(data=ph, x="activity", y="loan_amount").set_title('Philippines')

ca = df[df.country =='Cambodia']
ax = plt.subplots(figsize=(15, 10))
sns.barplot(data=ca, x="activity", y="loan_amount").set_title('Cambodia')
