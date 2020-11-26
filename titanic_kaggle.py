# Kaggle username = smalltowncelery

# Linear algebra
import numpy as np

# Data processing
import pandas as pd
# Set_option so that all columns are displayed in output
pd.set_option('display.width', None)
pd.set_option('display.max_columns',None)

# Data visualization
import seaborn as sns
from matplotlib import pyplot as plt

# Data transformation and splitting tools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ML models
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

# Model Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

# String Manipulation
import string

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Read the data into dataframes
# Both csv files are located in project file on pycharm
df_test = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")

# Concatenate training set and test set to make data manipulation faster
def concat_df(train_data, test_data):
    # Returns concatenated dataframe including both training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # DataFrame.loc accesses a specific cell in df using index & column labels
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
    # Returns training and test df separately

# Create concatenated df of df_train and df_test
df_all = concat_df(df_train, df_test)
# Create list of both dfs
dfs = [df_train, df_test]

# Preview df_train info, sample, and description
print("df_train info, sample, and description:")
print(df_train.info())
print('-'*25)
print(df_train.sample(5))
print('-'*25)
print(df_train.describe(), '\n'*2)

# Preview df_train info, sample, and description
print("df_test info, sample, and description:")
print(df_test.info())
print('-'*25)
print(df_test.sample(10))
print('-'*25)
print(df_test.describe(), '\n')

# Feature Descriptions:
# PassengerId = unique id of row & doesn't affect target (survival)
# Survived is the target variable we are trying to predict (0 or 1):
#   1 = Survived
#   0 = Not Survived
# Pclass = socioeconomic status = categorical ordinal feature:
#   1 = Upper Class
#   2 = Middle Class
#   3 = Lower Class
# SibSp = total number of the passengers' siblings and spouse
# Parch = total number of the passengers' parents and children
# Ticket = ticket number of the passenger
# Fare = passenger fare
# Cabin = cabin number of the passenger
# Embarked = port of embarkation = categorical feature (C, Q or S):
#   C = Cherbourg
#   Q = Queenstown
#   S = Southampton

# Plot how many people survived in df_train
f,ax=plt.subplots(1,1,figsize=(18,8)) # 1 subplots (1 row 1 column)
# Show data in seaborn countplot
sns.countplot('Survived',data=df_train,ax=ax)
ax.set_title('Count of Survived Passengers in df_train')

# Find how many null values there are in the training and final test df
print('\n', '-'*25, '\n')
print('\nNull Values in Training DF:\n', df_train.isnull().sum())
print('\n', '-'*25, '\n')
print('Null Values In Final Test DF:\n', df_test.isnull().sum())
print('\n', '-'*25, '\n')

# Training set have missing values in Age, Cabin and Embarked columns
# Test set have missing values in Age, Cabin and Fare columns

# Explore data visually
# Compare distribution of survival across Pclass and Age
f,ax=plt.subplots(figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax)
ax.set_title('Pclass and Age vs Survived')

# Explore correlation between pclass or age and other numeric features
print('\n', '-'*25, '\n')
print('Correlation of pclass or age with other numeric features')
for feature in (['Pclass', 'Age']):
    df_corr = df_all.corr().abs().unstack().\
        sort_values(kind="quicksort", ascending=False).reset_index()
    df_corr.rename(columns={"level_0": "Feature 1", "level_1":
        "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    print(df_corr[df_corr['Feature 1'] == '{}'.format(feature)], '\n')
print('\n', '-'*25, '\n')

# Because age and pclass correlate --> assign missing age based on pclass
# Compare distribution of survival across Sex and Age
f,ax=plt.subplots(figsize=(18,8))
sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax)
ax.set_title('Sex and Age vs Survived')

# Fill in missing age values w/ median of Sex and Pclass bc high correlation
# Use groupby to group passengers by Sex and Pclass to find median age
age = df_all.groupby(['Sex', 'Pclass']).median()['Age']
# Use lambda function to fill null age values with sex and pclass median
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x:
                                                               x.fillna
                                                               (x.median()))

# Plot survival count by port of embarkation in a barplot
fig, axes = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
sns.barplot(x = 'Embarked', y = 'Survived', data=df_train,
            ax = axes, errcolor='k')

# Preview the 2 passengers for which embarked is null
print('\n', '-'*25, '\n')
print("Two Passengers Info for Which Embarked is Null:")
print(df_all[df_all['Embarked'].isnull()])
print('\n', '-'*25, '\n')

# Googling Mrs. George Stone reveals she embarked from port S w/ Amelie Icard
df_all['Embarked']=df_all['Embarked'].fillna('S')

# Compare how port embarked + pclass + sex varies with survival
# Use facetgrid tool from seaborn
fg = sns.FacetGrid(df_train, col = 'Embarked')
fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
fg.add_legend()
# Embarked seems to correlate w/ survival, depending on sex and pclass

# Find passenger with missing fare value
print('\n', '-'*25, '\n')
print('Passenger Info for which Fare is Null:')
print(df_all[df_all['Fare'].isnull()])
# Assume that fare is related to family size and pclass
# Fill in with median fare of a male w/ a 3rd class ticket and no family
fare = df_all.groupby(['Pclass', 'Parch', 'SibSp',
                       'Sex'])['Fare'].median()[3][0][0]['male']
# Where [3][0][0]['male'] specify a male with 3=Pclass,0=Parch, and 0=SibSp
df_all['Fare']=df_all['Fare'].fillna(fare)
print('\n', '-'*25, '\n')

# Find how many null values there are in the training and final test df now
print('\n', '-'*25, '\n')
print('Null Values in Training DF:\n', df_train.isnull().sum())
print('\n', '-' * 25, '\n')
print('Null Values In Final Test DF:\n', df_test.isnull().sum())
print('\n', '-'*25, '\n')

# Compare pclass and survival between sexes in barplot
fig, axes = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df_train, ax=axes)
# Pclass clearly contributes to a persons chance of survival

# Compare pclass with survival
fg = sns.FacetGrid(df_train, col = 'Pclass', row='Survived')
# Plot age histogram in bins to compare age and survival between Pclass
fg.map(plt.hist, 'Age', alpha=0.5, bins=10)
fg.add_legend()

# Combine SibSp and Parch to create 1 feature = FamilySize in a shallow df copy
data1=df_train.copy()
data1['Fam_Size'] = data1['SibSp'] + data1['Parch'] +1
# Add the 1 because just 1 is considered 'Alone'
print('\n', '-'*25, '\n')
print('Family Size Counts:')
print(data1['Fam_Size'].value_counts().sort_values(ascending=False))
print('\n', '-'*25, '\n')

# Compare family size against survival
axes = sns.barplot(x = 'Fam_Size', y = 'Survived', data=data1)
plt.title('Fam_Size and Survival Comparison')
# Survival increases with family size until family size => 5

# How to deal with missing cabin values
# Can't be ignored completely because some cabins might have higher survival
# First letter of Cabin value = deck where cabin was located
# Boat deck = T cabin
# A, B, and C decks = 1st class passengers
# D and E decks = all classes
# F and G decks = 2nd and 3rd class passengers
# use M to indicate the cabin is missing from this passenger

# Create Deck column --> extract 1st letter of the Cabin
df_all['Deck'] = df_all['Cabin'].apply(lambda x:
                                       x[0] if pd.notnull(x) else 'M')
dr_op = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
         'Cabin', 'PassengerId','Ticket']
# Drop columns from df_all when grouping 'deck' with 'pclass' and 'Name'
df_decks = df_all.groupby(['Deck', 'Pclass']).count().drop(columns=dr_op)
#transpose df_decks to make more easy to work with
df_decks=df_decks.transpose()

# Find pclass distribution across decks
def get_pclass_dist(df):
    # Create dictionary for the number of passengers per class in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {},
                   'G': {}, 'M': {}, 'T': {}}
    # Deck column values are extracted from df_decks
    # Must call with columns.levels[0] because df_decks was transposed
    decks = df.columns.levels[0]
    # Create a new df = copy of df_decks with 0 in Pclass if empty
    for deck in decks:
        for pclass in range(1, 4):
            try:
                # Determine the count of passengers within the deck & pclass
                count = df[deck][pclass][0]
                # Add the count of that deck + pclass to deck_counts dictionary
                deck_counts[deck][pclass] = count
            # Use try and except in case the pclass called isn't on that deck
            except KeyError:
                # If no passengers from that Pclass on that deck, count = 0
                deck_counts[deck][pclass] = 0
    # Create df where columns are decks and Pclass are row indexes
    df_decks2 = pd.DataFrame(deck_counts)
    # Initalize empty dictionary of deck percentages to assess survival
    deck_percent = {}

    # Create dictionary of the percentage of every passenger class on each deck
    for col in df_decks2.columns: # Where columns are decks (A,B,C,etc.)
        # Find count of passengers per class on deck then divide by deck sum
        # Add to deck_percent dictionary under the right deck
        deck_percent[col] = [(round(count / df_decks2[col].sum(),4) * 100) for
                             count in df_decks2[col]]

    return deck_counts, deck_percent, df_decks2


all_deck_count, all_deck_per, df_decks_return = get_pclass_dist(df_decks)

# Display distribution of pclass across decks
def display_pclass_dist(percents):
    # Convert dict to a df then transpose to have Pclass as columns
    df_percentages = pd.DataFrame(percents).transpose()
    # Create list of deck names
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))
    # Label the columns as the different Pclasses
    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]

    plt.figure(figsize=(10, 8))
    plt.bar(bar_count, pclass1, label='Class 1')
    plt.bar(bar_count, pclass2, bottom=pclass1, label='Class 2')
    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, label='Class 3')

    plt.xlabel('Deck')
    plt.ylabel('Passenger Class Percentage')
    plt.xticks(bar_count, deck_names)
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)

    plt.legend(loc='best')
    plt.title('Distribution of Passenger Class in Decks')


display_pclass_dist(all_deck_per)

# Leave M like it is a deck because can't reasonably derive this feature
# Only 1 person on the T deck that is similar to A deck so move them there
move = df_all[df_all['Deck'] == 'T'].index
# Move represents row label of the passenger on the T deck
df_all.loc[move, 'Deck'] = 'A'

# Same Method is applied as above but deck is grouped with 'Survived' Feature
#instead of the 'Pclass' feature

drop2 = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked',
         'Pclass', 'Cabin', 'PassengerId', 'Ticket']
# Group together the count of passengers on same deck with same target label
df_decks_survived = df_all.groupby(['Deck', 'Survived']).\
    count().drop(columns=drop2).transpose()


def get_survived_dist(df):
    # Create a dict for the amount of people that survived or not on each deck
    surv_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {},
                   'G': {}, 'M': {}}
    # Deck column values are extracted again from df_decks_survived
    # Must call with columns.levels[0] because df_decks was transposed
    decks = df.columns.levels[0]

    for deck in decks:
        for target in range(0, 2): # Because survive can be yes (1) or no (0)
            surv_counts[deck][target] = df[deck][target][0]
            # Add amount of people that survive or not to the dict surv_counts

    # Create a df by deck (columns) of survival or not (row indexes)
    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}

    for col in df_surv.columns: # Iterate through decks
        # Divide number of people that survive on that deck by deck sum
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count
                                 in df_surv[col]]
    # Returns surv_counts (dictionary of survival counts by deck)
    # Returns surv_percentages (dictionary of survival percentage by deck)
    return surv_counts, surv_percentages

all_surv_count, all_surv_per = get_survived_dist(df_decks_survived)

# Display the percentage of passengers that survived by deck
def display_surv_dist(percentages):
    df_survived_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    bar_count = np.arange(len(deck_names))

    not_survived = df_survived_percentages[0]
    survived = df_survived_percentages[1]

    plt.figure(figsize=(10, 8))
    plt.bar(bar_count, not_survived, label="Not Survived")
    plt.bar(bar_count, survived, bottom=not_survived, label="Survived")

    plt.xlabel('Deck')
    plt.ylabel('Survival Percentage')
    plt.xticks(bar_count, deck_names)
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)

    plt.legend(loc='best')
    plt.title('The Percentage of Passengers that Survived on every Deck')


display_surv_dist(all_surv_per)

# Decks have different survival rates so this feature can't be dropped
# M w/ lowest survival rate bc hard to find cabin number of victims?
# A, B, and C decks = ABC bc all only 1st class passengers
# D and E decks = DE bc similar passenger class distribution + survival rate
# F and G decks = FG bc similar passenger class distribution + survival rate
# M deck not grouped bc it is very different + has lowest survival rate
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')
# Verify grouping worked
print('\n', '-'*25)
print('\n','Value Count of Every Deck Count:')
print(df_all['Deck'].value_counts())
print('\n', '-'*25, '\n')

# Drop cabin feature because now using deck feature instead
df_all.drop(['Cabin'], inplace=True, axis=1)
# Split df_all back into the df_train and df_test dataframes
df_train, df_test = divide_df(df_all)
dfs = [df_train, df_test]
# Check to see how many null values in both df
# Find how many null values there are in the training and final test df now
print('\n', '-'*25, '\n')
print('\nNull Values in Training DF:\n', df_train.isnull().sum())
print('\n', '-' * 25, '\n')
print('Null Values In Final Test DF:\n', df_test.isnull().sum())
print('\n', '-'*25, '\n')


# Break down age and fare features to allow model to generalize better
continuous = ['Age', 'Fare']
surv = df_train['Survived'] == 1

# Plot survival against various features in distplots
fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 8)) # 2 subplots
# Enumerate the continuous features to facilitate plotting
for i, feature in enumerate(continuous):
    # Display the distribution of survival in the continuous features
    sns.distplot(df_train[~surv][feature], label='Not Survived', ax=axs[i])
    # [~surv] indicates where ['Survived'] != 1, ~ is Bitwise NOT
    sns.distplot(df_train[surv][feature], label='Survived', ax=axs[i])
    axs[i].legend(loc='best')
    axs[i].set_xlabel('{}'.format(feature))
    axs[i].set_title('Survival Distribution in {}'.format(feature))


# Next to observe the categorical features
# Plot survival rate for all categorical features
categorical = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(14,9))
# Again, enumerate features to facilitate plotting
# Must specify 1 because index of the subplot starts at 1
for i, feature in enumerate(categorical, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=df_train)

    plt.xlabel('{}'.format(feature))
    plt.legend(['Not Survived', 'Survived'], loc='best')


# End of exploratory data analysis, print df.head() to see changes
print('\n', '-'*25, '\n')
print('Preview Dataframe At End of Exploratory Data Analysis:')
for df in dfs:
    print('\n', df.head())
    print("-" * 25)
print('\n', '-'*25, '\n')
# Visualize correlation between the features
# Only correlates numeric features bc strings cannot be correlated
plt.figure(figsize=(9,8))
colours = sns.color_palette("Blues", as_cmap=True)
# Use seaborn heat map tool and corr() to compute pairwaise column correlation
sns.heatmap(df_all.corr(),annot=True,cmap=colours)
# Get figure with plt.gcf() (get current figure = gcf)
# Features are not too correlated (no multicolinearity) so none are redundant

# Break fare into bins
# Use pd.qcut to bin continuous variables into quantile bins (= #people/bin)
# Create 13 bins for the fare category
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)

# Plot fare bins against survival
plt.figure(figsize=(22,9))
sns.countplot(x='Fare', hue='Survived', data=df_all)
plt.xlabel('Fare')
plt.ylabel('Passenger Count')
plt.tick_params(axis='x', labelsize=7)
plt.tick_params(axis='y', labelsize=10)
plt.legend(['Not Survived', 'Survived'], loc='best')
plt.title('Count of Survival in Fare Feature')

# Break age into bins
# Use pd.qcut bins continuous variables into quantile bins (= #people/bin)
# Create 10 bins for the age category
df_all['Age'] = pd.qcut(df_all['Age'], 10)
# Plot age bins against survival
plt.figure(figsize=(22, 9))
sns.countplot(x='Age', hue='Survived', data=df_all)
plt.xlabel('Age')
plt.ylabel('Passenger Count')
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)
plt.legend(['Not Survived', 'Survived'], loc='best')
plt.title('Survival Counts in Age Feature')

# Create family_size feature (in actual df not df copy like previously)
# To create family size, add sibsp, parch, and 1 (1 for the current passenger)
# Family size of 1 = alone; 2,3,4 = small; 5,6 = medium; and 7,8,11 = large
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1

# Plot distribution of family size and survival before + after grouping
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))

# Subplot 1 = Value Counts of Each Family Size
sns.barplot(x=df_all['Family_Size'].value_counts().index,
            y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])
axs[0][0].set_title('Value Counts of Each Family Size')

# Subplot 2 = Survival Counts in Family Size
sns.countplot(x='Family_Size', hue='Survived', data=df_all, ax=axs[0][1])
axs[0][1].set_title('Survival Counts in Family Size ')
# Map family size to replace with alone, small, medium, and large
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium',
              6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
# Map function used to substitute each value in a series according to a {}
df_all['Family_Size_Group'] = df_all['Family_Size'].map(family_map)

# Subplot 3 = Value Counts of each Grouped Family Size
sns.barplot(x=df_all['Family_Size_Group'].value_counts().index,
            y=df_all['Family_Size_Group'].value_counts().values, ax=axs[1][0])
axs[1][0].set_title('Value Counts of each Grouped Family Size')

# Subplot 4 = Grouped Survival Counts in Family Size
sns.countplot(x='Family_Size_Group', hue='Survived', data=df_all, ax=axs[1][1])
axs[1][1].set_title('Grouped Survival Counts in Family Size')

# Adjust size of tick labels + add legend on the survival subplots
for i in range(2):
    axs[i][1].legend(['Not Survived', 'Survived'], loc='best')
    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=12)
        axs[i][j].tick_params(axis='y', labelsize=12)
        axs[i][j].set_xlabel('')
        axs[i][j].set_ylabel('')


# Feature engineering with ticket feature now
# People travelling in groups that weren't family will have same ticket number
# Group tickets by the frequency of tickets
df_all['Ticket_Freq'] = df_all.groupby('Ticket')['Ticket'].transform('count')
# Plot the frequency of tickets against survival
fig, axs = plt.subplots(figsize=(12, 9))
sns.countplot(x='Ticket_Freq', hue='Survived', data=df_all)
# Configure countplot
plt.xlabel('Ticket Frequency')
plt.ylabel('Passenger Count')
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(['Not Survived', 'Survived'], loc='best')
plt.title('Count of Survival in Ticket Freq Feature')
# Survival rate: alone = low; 2,3,4 members = high; >4 members = low

# Create Title & IsMarried features
# Replace Miss, Mrs, Ms, Mlle, Lady, Mme, the Countess, Dona w/ Miss/Mrs/Ms
# Replace Dr, Col, Major, Jonkheer, Capt, Sir, Don, Rev w/ Dr/Army/Noble/Clergy
# Master = males less than 26 years old
# Split name into first and last then split title off first using str.split
df_all['Title'] = df_all['Name'].str.split(
    ', ', expand=True)[1].str.split('.', expand=True)[0]
# Is_Married = binary feature based on Mrs title = high survival rate
# Initialize 'Is_Married' to 0 (where 0 = not married)
df_all['Is_Married'] = 0
# Change from 0 to 1 if the individual's title is 'Mrs'
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1

# Group titles together that have similar characteristics to reduce overfitting
rep1 = ['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona']
rep2 = ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev']
df_all['Title'] = df_all['Title'].replace(rep1, 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(rep2, 'Dr/Military/Noble/Clergy')

# Divide df_all back into 2 df with updates using divide_df function
df_train,df_test= divide_df(df_all)
dfs=[df_train,df_test]

# Group last names to see if certain families have higher survival rate
# First need to extract last names from individual names
def extract_lastname(data):
    # Initialize empty families list
    families = []

    for i in range(len(data)): # Iterate through rows of column in df
        name = data.iloc[i] # Use iloc to access entry in df by index number
        # Bracket indicates marriage with female's maiden name in brackets!
        if '(' in name:
            name_wo_brack = name.split('(')[0]
        else:
            name_wo_brack = name
        # First split family name off then split off title
        family = name_wo_brack.split(',')[0]
        title = name_wo_brack.split(',')[1].strip().split(' ')[0]
        # Remove punctuation by replacing what string.punctuation returns
        for c in string.punctuation:
            family = family.replace(c, '').strip()
        # Add family name to all of the family names
        families.append(family)
    return families
    # Returns a list of family names

# Assign surname to family feature
df_all['Family'] = extract_lastname(df_all['Name'])
# Divide back into df_train and df_test bc family names may be unique to 1 df
df_train = df_all.loc[:890]
df_test = df_all.loc[891:]
dfs = [df_train, df_test]

# Create list of families & tickets in both df_train and df_test
non_unique_fams = [x for x in df_train['Family'].unique()
                       if x in df_test['Family'].unique()]
non_unique_ticks = [x for x in df_train['Ticket'].unique()
                      if x in df_test['Ticket'].unique()]

# Find family survival rate by grouping to find median survival rate + size
df_fam_surv = df_train.groupby('Family')['Survived','Family_Size'].median()
# Find ticket survival rate by grouping to find median survival rate + freq
df_tick_surv = df_train.groupby('Ticket')['Survived','Ticket_Freq'].median()

# Initialize empty dictionaries for the family and ticket survival rates
family_rates = {}
ticket_rates = {}

for i in range(len(df_fam_surv)): # Iterates thru family indexes in df_fam_surv
    # Check family exists in df_train and df_test + >1 member
    if df_fam_surv.index[i] in non_unique_fams and df_fam_surv.iloc[i, 1] > 1:
        # Add median survival rate by that family to family_rates dictionary
        family_rates[df_fam_surv.index[i]] = df_fam_surv.iloc[i, 0]

for i in range(len(df_tick_surv)):
    #check ticket exists in df_train and df_test + >1 member
    if df_tick_surv.index[i] in non_unique_ticks and df_tick_surv.iloc[i, 1]>1:
        ticket_rates[df_tick_surv.index[i]] = df_tick_surv.iloc[i, 0]

# Calculate mean survival rate of all individuals in df_train
mean_survival_rate = np.mean(df_train['Survived'])

# Initialize lists
train_fam_surv_rate = []
train_fam_surv_rate_NA = []
test_fam_surv_rate = []
test_fam_surv_rate_NA = []

# Create binary feature Family_Survival_Rate_NA for families unique to df_test
# The survival rate for these families cannot be known so = mean_survival_rate

# Iterate through rows in df_train df
for i in range(len(df_train)):
    if df_train['Family'].iloc[i] in family_rates:
        # Append family survival rate to end of survival_rate list
        train_fam_surv_rate.append(family_rates[df_train['Family'].iloc[i]])
        # Append 1 to survival_rate_NA list
        train_fam_surv_rate_NA.append(1)
    else:
        # Append mean survival rate to end of survival_rate list
        train_fam_surv_rate.append(mean_survival_rate)
        # Append 0 to survival_rate_NA list
        train_fam_surv_rate_NA.append(0)

# Repeat same process as above for test set
for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        # Append family survival rate to end of survival_rate list
        test_fam_surv_rate.append(
            family_rates[df_test['Family'].iloc[i]])
        # Append 1 to survival_rate_NA list
        test_fam_surv_rate_NA.append(1)
    else:
        test_fam_surv_rate.append(mean_survival_rate)
        # Append mean survival rate to end of survival_rate list
        test_fam_surv_rate_NA.append(0)
        # Append 0 to survival_rate_NA list

# Add the family survival rates as features to the df_train and df_test dfs
df_train['Family_Survival'] = train_fam_surv_rate
df_train['Family_Survival_NA'] = train_fam_surv_rate_NA
df_test['Family_Survival'] = test_fam_surv_rate
df_test['Family_Survival_NA'] = test_fam_surv_rate_NA

# Initialize lists
train_tick_surv_rate = []
train_tick_surv_rate_NA = []
test_tick_surv_rate = []
test_tick_surv_rate_NA = []

# Same process that was previously done for families, repeat for tickets
# Iterate through rows
for i in range(len(df_train)):
    if df_train['Ticket'].iloc[i] in ticket_rates:
        # Append ticket survival rate to end of survival_rate list
        train_tick_surv_rate.append(ticket_rates[df_train['Ticket'].iloc[i]])
        train_tick_surv_rate_NA.append(1)
    else:
        train_tick_surv_rate.append(mean_survival_rate)
        # Append ticket survival rate to end of survival_rate list
        train_tick_surv_rate_NA.append(0)

for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        # Append ticket survival rate to end of survival_rate list
        test_tick_surv_rate.append(
            ticket_rates[df_test['Ticket'].iloc[i]])
        test_tick_surv_rate_NA.append(1)
    else:
        test_tick_surv_rate.append(mean_survival_rate)
        # Append mean survival rate to end of survival_rate list
        test_tick_surv_rate_NA.append(0)

df_train['Ticket_Survival'] = train_tick_surv_rate
df_train['Ticket_Survival_NA'] = train_tick_surv_rate_NA
df_test['Ticket_Survival'] = test_tick_surv_rate
df_test['Ticket_Survival_NA'] = test_tick_surv_rate_NA

dfs = [df_train, df_test]
# Average Ticket_Survival_Rate + Family_Survival_Rate = Survival_Rate
# Average Ticket_Survival_Rate_NA + Family_Survival_Rate_NA = Survival_Rate_NA
for df in dfs:
    df['Survival_Rate'] = (df['Ticket_Survival'] + df['Family_Survival']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_NA']
                              + df['Family_Survival_NA']) / 2

# Time for feature transformation
# Convert categorical data to dummy variables using pandas + sklearn functions
# Embarked, sex, deck, title, + family_size_grouped = object type
# Age and fare features are category type

# Transform age and fare with labelencoder() (labels n objects from 0 to n)
labels = ['Age', 'Fare']
for df in dfs:
    for feature in labels:
        df[feature] = LabelEncoder().fit_transform(df[feature])

# Transform pclass,sex,deck,embarked,title,family_size_grouped w/ get_dummies
# Get_dummies creates additional features based on the number of unique values
#in the categorical feature = better for features that shouldn't be ranked

onehot = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Group']
df_train = pd.get_dummies(data=df_train,columns=onehot)
df_test = pd.get_dummies(data=df_test,columns=onehot)

# Concatenate df_train and df_test
df_all = concat_df(df_train, df_test)

# List columns that can be dropped because = irrelevant or have been replaced
drop_cols = ['Family', 'Family_Size', 'Survived','Name', 'Parch',
             'PassengerId','SibSp', 'Ticket','Ticket_Survival',
             'Family_Survival', 'Ticket_Survival_NA',
             'Family_Survival_NA']
# Drop the columns from both datasets
df_all.drop(columns=drop_cols, inplace=True)
print('\n', '-'*25, '\n')
print('Preview Dataframe after Feature Engineering and Transformation:')
print(df_all.head())
print('\n', '-'*25, '\n')

# Use sklearn train_test_split to split data df into train and test sets
# Train_test_split default split is 75/25 for train/test (will use this)
tts = train_test_split
X = df_train.drop(columns=drop_cols)
train_x, valid_x, train_y, valid_y = tts(X, df_train['Survived'],
                                         random_state = 0)

# Train several ML models + compare results
# Use standard scaler to normalize mean and sd in train and validate dfs
X_train = StandardScaler().fit_transform(train_x)
Y_train = train_y
X_valid = StandardScaler().fit_transform(valid_x)
Y_valid = valid_y
X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))
print('\n', '-'*25, '\n')
print('Shape of Dataframes after Splitting:')
print('X_train shape: {}'.format(X_train.shape))
print('Y_train shape: {}'.format(Y_train.shape))
print('X_valid shape: {}'.format(X_valid.shape))
print('Y_valid shape: {}'.format(Y_valid.shape))
print('X_test shape: {}'.format(X_test.shape))
print('\n', '-'*25, '\n')

# Stochastic Gradient Descent (SGD):
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
# Train the model on the training data
sgd.fit(X_train, Y_train)
# Test performance of SGD on validation set
acc_sgd = round(sgd.score(X_valid, Y_valid) * 100, 2)
# Repeat with other ML models

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
acc_random_forest = round(rf.score(X_valid, Y_valid) * 100, 2)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_valid, Y_valid) * 100, 2)

# K Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_valid, Y_valid) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_valid, Y_valid) * 100, 2)

# Percepton
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)
acc_perceptron = round(perceptron.score(X_valid, Y_valid) * 100, 2)

# Linear Suppport Vector Machine
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
acc_linear_svc = round(linear_svc.score(X_valid, Y_valid) * 100, 2)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_valid, Y_valid) * 100, 2)

# Compare accuracy of models on validation set to find the best model
# Create dataframe of models with accuracy scores on validation set
model_results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent',
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_decision_tree]})
result_df = model_results.sort_values(by='Score', ascending=False)
results_df = result_df.set_index('Score')
print('\n','Model Accuracy Scores on Validation Set')
print(result_df)
print('\n', '-'*25, '\n')
# Random forest performs best on the validation model


# Check random forest with K-fold cross validation on the training set
# Set random forest parameters (number of trees in forest = 100)
rf = RandomForestClassifier(n_estimators=100,oob_score=True)
# Run 10 folds for the K-folds cross validation score
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
# Returns array of scores of the estimator for each run of the cross validation
# Calculate mean and sd of the array of scores
print('K-Fold Cross Validation Metrics:')
print("Cross Validation Scores Array:", scores)
print("Mean of Cross Validation Scores:", scores.mean())
print("Standard Deviation of Cross Validation Scores:", scores.std())
print('\n','-'*25)
# Model has an average accuracy of approx 82.5% with 6% SD

# Train random forest on training set again for feature selection
rf.fit(X_train, Y_train)
# Evaluate accuracy score on validation set again
acc_random_forest = round(rf.score(X_valid, Y_valid) * 100, 2)

# Determine the importance of features with featureimportances function
f_i = pd.DataFrame({'feature':X.columns,
                    'importance':np.round(rf.feature_importances_,3)})
f_i = f_i.sort_values('importance',ascending=False)
print('\n', 'Feature Importance Dataframe:')
print(f_i.head(26))
print('-'*25)

# Find oob score to estimate the generalization accuracy of the training set
print('\n', '-'*25)
print('Random Forest oob score:')
print("oob score:", round(rf.oob_score_, 4)*100,"%")
print("-"*25, '\n')

# Hyperparameter tuning with random forest
# Param_grid is list of hyperparameters as keys and lists of parameter settings
#to try as values
# Changed to comments because takes approx 6-7 minutes to run
#param_grid = { "criterion" : ["gini", "entropy"],
               #"min_samples_leaf" : [1, 5, 10,],
               #"min_samples_split" : [2, 4, 10,],
               #"n_estimators": [100,500,1100,1500]}
#gs=GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                #param_grid=param_grid,verbose=True)
#gs.fit(X_train,Y_train)
#print best score: mean cross-validated score of the best estimator
#print('Best Score:', gs.best_score_)
#print best estimator: estimator that was chosen by the search
#print('Best Estimator:', gs.best_estimator_)
#print best hyperparameters: hyperparameter settings w/ best results
#print('Best Hyperparameters:', gs.best_params_)

# GridSearchCV Results:
# Best Score: 0.8413533834586466
# Best Estimator: RandomForestClassifier(min_samples_leaf=10, n_estimators=500, random_state=0)
# Best Hyperparameters: {'criterion': 'gini',
#   'min_samples_leaf': 10,
#   'min_samples_split': 2,
#   'n_estimators': 500}


# Run RandomForestClassifer with new hyperparameters from GridSearchCV
random_forest = RandomForestClassifier(criterion='gini',
                                           n_estimators=500,
                                           max_depth=7,
                                           min_samples_split=2,
                                           min_samples_leaf=10,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=0,
                                           n_jobs=-1,
                                           verbose=1)


# Train model with new hyperparameters on training dataset
random_forest.fit(X_train, Y_train)
# Use model to make predictions about test set (will be used for submission)
Y_prediction = (random_forest.predict(X_test)).astype(int)


# Score model with new hyperparameters on validation dataset
print('\n','-'*25)
print('Random Forest Accuracy Score after Hyperparameter Tuning:')
print('Accuracy Score:', random_forest.score(X_valid, Y_valid))
print('\n')


# Determine oob score of the model with new hyperparameters
print('\n','-'*25)
print('Random Forest oob Score after Hyperparameter Tuning:')
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
print('\n')

# Repeat K-Fold cross validation with new hyperparameters
scores = cross_val_score(random_forest, X_train, Y_train, cv=10,
                         scoring = "accuracy")
print('\n','-'*25)
print('Random Forest Cross_Val Score after Hyperparameter Tuning:')
print("Cross Validation Scores Array:", scores)
print("Mean of Cross Validation Scores:", scores.mean())
print("Standard Deviation of Cross Validation Scores:", scores.std())
print('\n')

# Use confusion matrix to determine number of incorrect and correct predictions
#on validation model
y_predict = cross_val_predict(random_forest, X_valid, Y_valid, cv=3)
print('\n','-'*25)
print('Confusion Matrix from Random Forest Model:')
print(confusion_matrix(Y_valid, y_predict))
print('\n')

# Confusion Matrix Results:
# 118 passengers in training set are correctly predicted as not survived (tn)
# 21 passengers in training set are incorrectly predicted as survived (fp)
# 30 passengers in training set are incorrectly predicted as not survived (fn)
# 54 passengers in training set are correctly predicted as survived (tp)

# Find F-1 score on validation set
# First must find Precision and Recall Scores on validation set

# Precision = tp / (tp + fp) where tp = true positives and fp = false positives
print('\n','-'*25)
print('Precision and Recall Scores from Random Forest Model:')
print("Precision:", precision_score(Y_valid, y_predict))
# Recall = tp / (tp + fn) where tp = true positives and fn = false negatives
print("Recall:",recall_score(Y_valid, y_predict))
print('\n')


#F1 score can be interpreted as a weighted average of the precision and recall
#F1 score reaches its best value at 1 and worst score at 0
print('\n','-'*25)
print('F1 from Random Forest Model:')
print("f1 score:", f1_score(Y_valid, y_predict))
print('\n')

# Find ROC AUC curve for validation set
# Find probabilities of our predictions
y_probs = random_forest.predict_proba(X_valid)
y_probs = y_probs[:,1]

# Compute tp (true positive) rate and fp (false positive) rate for valid set
fp_rate, tp_rate, thresholds = roc_curve(Y_valid, y_probs)
# Plot fp_rate against tp_rate w/ a benchmark random classifier
plt.figure(figsize=(9,8))
plt.plot(fp_rate, tp_rate)
plt.plot([0, 1], [0, 1], 'r', linewidth=3)
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Calculate ROC_AUC score on validation set
rocauc_score = roc_auc_score(Y_valid, y_probs)
print('\n','-'*25)
print('ROC AUC Score from Random Forest Model:')
print("ROC AUC Score:", rocauc_score)
print('\n')

# Show all plots
plt.show()

# Create dataframe for submission to convert to CSV
kaggle_submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_prediction
    })

kaggle_submission.to_csv('kaggle_submission.csv', index=False)