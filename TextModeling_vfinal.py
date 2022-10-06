import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import nltk
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold
from sklearn import tree, metrics
import tensorflow_hub as hub

'''
This data contains student responses to graph interpretation and construction questions from
the GraphSmarts assessment tool within SimBio. It also contains hand-coded labels of the subject
matter and content of student responses, such as whether a response contains content about
statistical features, or an aesthetic choice.

The goal of this project is to construct machine-learned models that can accurately identify 
the content of a student's response, given a student's response text and a set of hand-coded labels.
'''

input_data = r"C:\Users\Stefan\Documents\GraphSmart\GraphSmarts Class Summaries Scrubbed_csv version.csv"
tf.keras.utils.set_random_seed(20220919)

df_input = pd.read_csv(input_data,
                       encoding='WINDOWS-1252')
'''
The dataset contains two different types of questions: 
Explanation questions, where students explain the information that is presented in a graph.
Justification questions, where students explain the choices that they made in constructing a graph.

Students are asked two Explanation questions, and three Justification questions.
The hand-coded labels are consistent within Explanation and Justification questions.
There are two labels that are consistent across both Explanation and Justification questions.

In the next section of the code, I construct separate dataframes for each question, noting them as
Explanation or Justification. I was worried about potential typos in the source data -- there are
some weird formatting choices and inconsistencies -- so these column labels are copied and pasted
directly from the .csv file. You can see for yourself where these inconsistencies are, such as 
"GraphSelection1_Explain" versus "GraphSelect2_Explain".
'''

df_explain_1 = df_input[['studentID',
                         'GraphSelection1_Explain',
                         'GraphSelect1:Trend/Analysis',
                         'GraphSelect1:Personal Preference',
                         'GraphSelect1:StateWhatIsPlotted',
                         'GraphSelect1:Comparing/Contrasting',
                         'GraphSelect1:Aesthetics',
                         'GraphSelect1:Data Form/Statistics',
                         'GraphSelect1:Data Summarization/variability']].copy()
df_explain_2 = df_input[['studentID',
                         'GraphSelect2_Explain',
                         'GraphSelect2:Trend/Analysis',
                         'GraphSelect2:Personal Preference',
                         'GraphSelect2:StateWhatIsPlotted',
                         'GraphSelect2:Comparing/Contrasting',
                         'GraphSelect2:Aesthetics',
                         'GraphSelect2:Data Form/Statistics',
                         'GraphSelect2:Data Summarization/variability']].copy()
df_justify_1 = df_input[['studentID',
                         'GraphType:P1:Justification',
                         'GraphType:P1:Statistical features',
                         'GraphType:P1:Data Exploration',
                         'GraphType:P1:Trend/analysis',
                         'GraphType:P1:Comparing and contrasting',
                         'GraphType:P1:Variable type',
                         'GraphType:P1:Unclear',
                         'GraphType:P1:Personal preference',
                         'GraphType:P1:Number of given data points',
                         'GraphType:P1:Stating what is plotted']].copy()
df_justify_2 = df_input[['studentID',
                         'GraphType:P2:Justification',
                         'GraphType:P2:Statistical features',
                         'GraphType:P2:Data Exploration',
                         'GraphType:P2:Trend/analysis',
                         'GraphType:P2:Comparing and contrasting',
                         'GraphType:P2:Variable type',
                         'GraphType:P2:Unclear',
                         'GraphType:P2:Personal preference',
                         'GraphType:P2:Number of given data points',
                         'GraphType:P2:Stating what is plotted']].copy()
df_justify_3 = df_input[['studentID',
                         'GraphType:P3:Justification',
                         'GraphType:P3:Statistical features',
                         'GraphType:P3:Data Exploration',
                         'GraphType:P3:Trend/analysis',
                         'GraphType:P3:Comparing and contrasting',
                         'GraphType:P3:Variable type',
                         'GraphType:P3:Unclear',
                         'GraphType:P3:Personal preference',
                         'GraphType:P3:Number of given data points',
                         'GraphType:P3:Stating what is plotted']].copy()

'''
Now I rename the columns in each dataframe, generating better consistency across problems and 
problem types.
'''

df_explain_1.rename(columns={'studentID': 'studentID',
                             'GraphSelection1_Explain': 'Text Response',
                             'GraphSelect1:Trend/Analysis': 'Trend/Analysis',
                             'GraphSelect1:Personal Preference': 'Personal Preference',
                             'GraphSelect1:StateWhatIsPlotted': 'Stating What Is Plotted',
                             'GraphSelect1:Comparing/Contrasting': 'Comparing/Contrasting',
                             'GraphSelect1:Aesthetics': 'Aesthetics',
                             'GraphSelect1:Data Form/Statistics': 'Data Form/Statistics',
                             'GraphSelect1:Data Summarization/variability': 'Data Summarization/Variability'},
                    inplace=True)
df_explain_2.rename(columns={'GraphSelect2_Explain': 'Text Response',
                             'GraphSelect2:Trend/Analysis': 'Trend/Analysis',
                             'GraphSelect2:Personal Preference': 'Personal Preference',
                             'GraphSelect2:StateWhatIsPlotted': 'Stating What Is Plotted',
                             'GraphSelect2:Comparing/Contrasting': 'Comparing/Contrasting',
                             'GraphSelect2:Aesthetics': 'Aesthetics',
                             'GraphSelect2:Data Form/Statistics': 'Data Form/Statistics',
                             'GraphSelect2:Data Summarization/variability': 'Data Summarization/Variability'},
                    inplace=True)
df_justify_1.rename(columns={'GraphType:P1:Justification': 'Text Response',
                             'GraphType:P1:Statistical features': 'Statistical Features',
                             'GraphType:P1:Data Exploration': 'Data Exploration',
                             'GraphType:P1:Trend/analysis': 'Trend/Analysis',
                             'GraphType:P1:Comparing and contrasting': 'Comparing/Contrasting',
                             'GraphType:P1:Variable type': 'Variable Type',
                             'GraphType:P1:Unclear': 'Unclear',
                             'GraphType:P1:Personal preference': 'Personal Preference',
                             'GraphType:P1:Number of given data points': 'Number of Data Points',
                             'GraphType:P1:Stating what is plotted': 'Stating What Is Plotted'},
                    inplace=True)
df_justify_2.rename(columns={'GraphType:P2:Justification': 'Text Response',
                             'GraphType:P2:Statistical features': 'Statistical Features',
                             'GraphType:P2:Data Exploration': 'Data Exploration',
                             'GraphType:P2:Trend/analysis': 'Trend/Analysis',
                             'GraphType:P2:Comparing and contrasting': 'Comparing/Contrasting',
                             'GraphType:P2:Variable type': 'Variable Type',
                             'GraphType:P2:Unclear': 'Unclear',
                             'GraphType:P2:Personal preference': 'Personal Preference',
                             'GraphType:P2:Number of given data points': 'Number of Data Points',
                             'GraphType:P2:Stating what is plotted': 'Stating What Is Plotted'},
                    inplace=True)
df_justify_3.rename(columns={'GraphType:P3:Justification': 'Text Response',
                             'GraphType:P3:Statistical features': 'Statistical Features',
                             'GraphType:P3:Data Exploration': 'Data Exploration',
                             'GraphType:P3:Trend/analysis': 'Trend/Analysis',
                             'GraphType:P3:Comparing and contrasting': 'Comparing/Contrasting',
                             'GraphType:P3:Variable type': 'Variable Type',
                             'GraphType:P3:Unclear': 'Unclear',
                             'GraphType:P3:Personal preference': 'Personal Preference',
                             'GraphType:P3:Number of given data points': 'Number of Data Points',
                             'GraphType:P3:Stating what is plotted': 'Stating What Is Plotted'},
                    inplace=True)

'''
Now that we have our explanation and justification dataframes, we can join them using studentID as our key.
'''

df_explain = pd.concat([df_explain_1, df_explain_2],
                       axis=0,
                       ignore_index=True)
df_justify = pd.concat([df_justify_1, df_justify_2, df_justify_3],
                       axis=0,
                       ignore_index=True)

'''
We also do a quick validation that the dataframes are the right shape, and that the columns are correct.
'''

print(f'Dimensions of Explanation responses: {df_explain.shape}.')
print(f'Dimensions of Justification responses: {df_justify.shape}.')

print(f'Column names for Explanation responses: {df_explain.columns}.')
print(f'Column names for Justification responses: {df_justify.columns}.')

# assertion: .csv file data and dataframe size match
'''
Now that we have the two dataframes we'll be working with, we start to clean up the contents of the data.
First, we correct some weirdness that exists in the source data. Explanation questions use a binary label of 
[x,NaN] for presence/absence of a label, while Justification questions use a binary label of [1,0].
Our lives get easier if we make these consistent internally, so we reformat Explanation questions to match.
'''

columns = ['Trend/Analysis',
           'Personal Preference',
           'Stating What Is Plotted',
           'Comparing/Contrasting',
           'Aesthetics',
           'Data Form/Statistics',
           'Data Summarization/Variability']

df_explain[columns] = df_explain[columns].replace('x', '1')
df_explain[columns] = df_explain[columns].fillna('0')

'''
We remove any responses where a student did not provide an answer, as we shouldn't be counting these.
'''

df_explain.dropna(how='any',
                  subset=['Text Response'],
                  inplace=True)
df_justify.dropna(how='any',
                  subset=['Text Response'],
                  inplace=True)

df_explain.reset_index(inplace=True,
                       drop=True)
df_justify.reset_index(inplace=True,
                       drop=True)

'''
Finally, we perform a stemming step. This isn't necessary for all of our modeling approaches, but it will
be useful for bag of words. We store the stemmed responses in a new column that we can use later.

We force lowercase in all of our text responses, remove stopwords and punctuation, and then apply
the Porter Stemmer from NLTK on the tokenized result.
'''


def stem_token(sentence):
    tokens = sentence.split()
    stemmed = [stemmer.stem(token) for token in tokens]
    return [" ".join(stemmed)]


def stem_column(df_col):
    _case = df_col.apply(lambda x: x.lower())
    _punc = _case.str.replace('[.,!?]', '')
    _stop = _punc.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    return _stop.apply(stem_token)


nltk.download('stopwords')
stopwords = stopwords.words('english')
stemmer = PorterStemmer()

df_explain['Stemmed Response'] = stem_column(df_explain['Text Response'])
df_justify['Stemmed Response'] = stem_column(df_justify['Text Response'])

'''
Our preprocessing is done, and now we start modeling. For this paper, we wanted to compare the performance
of four different types of models: 

(1) bag of words
(2) lexical sophistication
(3) semantic tagging
(4) sentence encoders

First, we set up bag of words. We convert our tokenized column to a vector space that represents the frequency
with which each word appears in each sentence. We include both unigrams (single words) and bigrams (pairs of words)
to capture additional information.
'''


def vectorize(tokens):
    tokens = [item for sublist in tokens for item in sublist]
    _cvr = CountVectorizer(input=tokens,
                           ngram_range=(1, 2))
    bow_vector = _cvr.fit_transform(tokens)
    return pd.DataFrame(bow_vector.toarray(),
                        columns=_cvr.get_feature_names_out())


explain_bow = vectorize(df_explain['Stemmed Response'].values.tolist())
justify_bow = vectorize(df_justify['Stemmed Response'].values.tolist())

'''
This is all we need to do bag of words. Now we can do a train/test split and start fitting a model.
Group Shuffle is what we want to use to ensure that our train/test split is at the student level. We do not
want the same student's responses in both the training and testing sets, as this could lead to overfitting.
'''


def kfold_crossval(X, y, students, n_split):
    k_fold = GroupKFold(n_splits=n_split)
    scores = []

    for train_index, test_index in k_fold.split(X,
                                                y,
                                                groups=students):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        scores.append(metrics.cohen_kappa_score(pred, y_test))
    return scores


# fit to entire dataset; report kappa from validation step; report coefficients from entire dataset


def fit_report_tree(X, y):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(X, y)
    pred = clf.predict(X)
    scores = metrics.cohen_kappa_score(pred, y)
    pred_y_data = clf.predict(X)
    tree_text = tree.export_text(clf, feature_names=list(X.columns))

    return pred_y_data, scores, tree_text


label_columns_explain = ['Trend/Analysis',
                         'Personal Preference',
                         'Comparing/Contrasting',
                         'Data Form/Statistics',
                         'Data Summarization/Variability']
# Stating What Is Plotted and Aesthetics don't have enough positive labeled cases to model.

label_columns_justify = ['Statistical Features',
                         'Data Exploration',
                         'Trend/Analysis',
                         'Comparing/Contrasting',
                         'Variable Type',
                         'Personal Preference']
# Unclear, Number of Data Points, and Stating What Is Plotted don't have enough positive labeled cases to model.

students_explain = df_explain['studentID'].values.tolist()
students_justify = df_justify['studentID'].values.tolist()

print("Decision Tree models for 'explain' questions.")
for i in label_columns_explain:
    kappa_scores = kfold_crossval(explain_bow, df_explain[i], students_explain, n_split=10)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
    bow_prediction, r_kappa, r_tree = fit_report_tree(explain_bow, df_explain[i])
    print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
    print(f'Decision tree classifier for {i} = {r_tree}.')
print("***")
print("Decision Tree models for 'justify' questions.")
for i in label_columns_justify:
    kappa_scores = kfold_crossval(justify_bow, df_justify[i], students_justify, n_split=10)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
    bow_prediction, r_kappa, r_tree = fit_report_tree(justify_bow, df_justify[i])
    print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
    print(f'Decision tree classifier for {i} = {r_tree}.')
print("***")
quit()

'''
Next we're going to generate features using TAALES and LIWC. We want to pipe out the text data that we currently
have, as neither of these software platforms have Python integration that I know of.
'''

e_out = 'GraphSmart_Text_Output_Explain.csv'
j_out = 'GraphSmart_Text_Output_Justify.csv'
df_explain.to_csv(e_out,
                  columns=['studentID', 'Text Response'])
df_justify.to_csv(j_out,
                  columns=['studentID', 'Text Response'])

# we run LIWC on our output data here

'''
Now we read that data back in, making a set of LIWC features. We also save these
features to our primary dataframes, so that we can build a model using all features later on.
'''

e_in = r"C:\Users\Stefan\Documents\GraphSmart\Output Data\GraphSmart_Text_Output_Explain - LIWC Analysis.csv"
j_in = r"C:\Users\Stefan\Documents\GraphSmart\Output Data\GraphSmart_Text_Output_Justify - LIWC Analysis.csv"

explain_liwc = pd.read_csv(e_in,
                           encoding='WINDOWS-1252')
justify_liwc = pd.read_csv(j_in,
                           encoding='WINDOWS-1252')

explain_liwc.index = explain_liwc['index']
justify_liwc.index = justify_liwc['index']

explain_liwc_features = explain_liwc.drop(['studentID', 'ColumnID', 'Text', 'Segment'],
                                          axis=1)
justify_liwc_features = justify_liwc.drop(['studentID', 'ColumnID', 'Text', 'Segment'],
                                          axis=1)

'''
With our new LIWC features, we fit these models separately.
'''

# print("Decision Tree models for 'explain' questions using LIWC features.")
# for i in label_columns_explain:
#     kappa_scores = kfold_crossval(explain_liwc_features, df_explain[i], students_explain, n_split=10)
#     print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
#     bow_prediction, r_kappa, r_tree = fit_report_tree(explain_liwc_features, df_explain[i])
#     print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
#     # print(f'Decision tree classifier for {i} = {r_tree}.')
# print("***")
# print("Decision Tree models for 'justify' questions using LIWC features.")
# for i in label_columns_justify:
#     kappa_scores = kfold_crossval(justify_liwc_features, df_justify[i], students_justify, n_split=10)
#     print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
#     bow_prediction, r_kappa, r_tree = fit_report_tree(justify_liwc_features, df_justify[i])
#     print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
#     # print(f'Decision tree classifier for {i} = {r_tree}.')
# print("***")

'''
Now we do the same thing for TAALES features. TAALES wants to read text data in one at a time, so we have to write
all of our individual rows out to separate files, process them, and recombine them.
'''
os.chdir("TAALES Input Explain")

for i, row in df_explain.iterrows():
    with open(str(i) + "_"+ str(row['studentID']) + "_explain.txt", 'w') as f:
        f.write(row['Text Response'])

os.chdir("..")
os.chdir("TAALES Input Justify")

for i, row in df_justify.iterrows():
    with open(str(i) + "_"+ str(row['studentID']) + "_justify.txt", 'w') as f:
        f.write(row['Text Response'])

os.chdir("..")

# we run TAALES on the two output folders here

e_in = r"C:\Users\Stefan\Documents\GraphSmart\Output Data\TAALES_Explain_Output.csv"
j_in = r"C:\Users\Stefan\Documents\GraphSmart\Output Data\TAALES_Justify_Output.csv"
explain_taales = pd.read_csv(e_in,
                             encoding="WINDOWS-1252")
justify_taales = pd.read_csv(j_in,
                             encoding="WINDOWS-1252")

'''
TAALES screws the order of the questions up, so we need to re-sort on the portion of the filename that contains
the index position, then set that to the new index, so that we can join TAALES features to existing dataframes.
'''

explain_taales['ProxyIndex'] = explain_taales['Filename'].str.split('_').str[0]
justify_taales['ProxyIndex'] = justify_taales['Filename'].str.split('_').str[0]

explain_taales.ProxyIndex = explain_taales.ProxyIndex.astype(int)
justify_taales.ProxyIndex = justify_taales.ProxyIndex.astype(int)

explain_taales.sort_values(by='ProxyIndex',
                           inplace=True)
justify_taales.sort_values(by='ProxyIndex',
                           inplace=True)

explain_taales.set_index('ProxyIndex',
                         inplace=True)
justify_taales.set_index('ProxyIndex',
                         inplace=True)

explain_taales.drop('Filename',
                    inplace=True,
                    axis=1)
justify_taales.drop('Filename',
                    inplace=True,
                    axis=1)

'''
Now we can fit the TAALES features.
'''

print("Decision Tree models for 'explain' questions using TAALES features.")
for i in label_columns_explain:
    kappa_scores = kfold_crossval(explain_taales, df_explain[i], students_explain, n_split=10)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
    bow_prediction, r_kappa, r_tree = fit_report_tree(explain_taales, df_explain[i])
    print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
    # print(f'Decision tree classifier for {i} = {r_tree}.')
print("***")
print("Decision Tree models for 'justify' questions using TAALES features.")
for i in label_columns_justify:
    kappa_scores = kfold_crossval(justify_taales, df_justify[i], students_justify, n_split=10)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
    bow_prediction, r_kappa, r_tree = fit_report_tree(justify_taales, df_justify[i])
    print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
    # print(f'Decision tree classifier for {i} = {r_tree}.')
print("***")
quit()

'''
Now that we have constructed all of our features, we combine them to run a model that includes all three sources
of features. We drop redundant and/or unnecessary columns, then join the feature dataframes and repeat our 
previous analyses.
'''

combined_explain = pd.concat([explain_bow, explain_liwc_features, explain_taales],
                             axis=1)
combined_justify = pd.concat([justify_bow, justify_liwc_features, justify_taales],
                             axis=1)

print("Decision Tree models for 'explain' questions using ALL features.")
for i in label_columns_explain:
    kappa_scores = kfold_crossval(combined_explain, df_explain[i], students_explain, n_split=10)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
    bow_prediction, r_kappa, r_tree = fit_report_tree(combined_explain, df_explain[i])
    print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
    # print(f'Decision tree classifier for {i} = {r_tree}.')
print("***")
print("Decision Tree models for 'justify' questions using ALL features.")
for i in label_columns_justify:
    kappa_scores = kfold_crossval(combined_justify, df_justify[i], students_justify, n_split=10)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
    bow_prediction, r_kappa, r_tree = fit_report_tree(combined_justify, df_justify[i])
    print(f'Resultant Cohen\'s Kappa for {i} = {r_kappa}.')
    # print(f'Decision tree classifier for {i} = {r_tree}.')
print("***")

'''
Now with all the other feature sets done, we work on the neural network and sentence encoder features.
'''

embed = hub.load(r"C:\Users\Stefan\Documents\GraphSmart\UniversalSentenceEncoder")
explain_embeddings = embed(df_explain['Text Response'])
explain_embeddings = pd.DataFrame(explain_embeddings)
justify_embeddings = embed(df_justify['Text Response'])
justify_embeddings = pd.DataFrame(justify_embeddings)


def nn_crossval(X, y, group):
    score = []
    _k = GroupKFold(n_splits=10)
    for train_index, test_index in _k.split(X, y, groups=group):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = Sequential()
        model.add(Dense(12, input_shape=(512,), activation='relu')) # layers = features/3 (roughly)
        model.add(Dense(8, activation='relu')) # try different activations
        model.add(Dense(1, activation='sigmoid')) # try different #s of layers
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      optimizer='adam',
                      metrics=[tfa.metrics.CohenKappa(num_classes=2)])
        model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
        _, performance = model.evaluate(X_test, y_test)
        score.append(performance)
    return score


print("Decision Tree models for 'explain' questions using sentence encoder features.")
for i in label_columns_explain:
    kappa_scores = nn_crossval(explain_embeddings, df_explain[i].astype(int), students_explain)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
print("***")

print("Decision Tree models for 'justify' questions using sentence encoder features.")
for i in label_columns_justify:
    kappa_scores = nn_crossval(justify_embeddings, df_justify[i].astype(int), students_justify)
    print(f'Cross-validated Cohen\'s Kappa for {i} = {np.mean(kappa_scores), np.std(kappa_scores)}.')
print("***")