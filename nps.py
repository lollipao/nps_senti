#DataFrame manipulation
import pandas as pd
#Text cleaning
import neattext.functions as nfx
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk import pos_tag
#Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Split dataset
from sklearn.model_selection import train_test_split
#Build Model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, f1_score


def cleanText(text):
    text = text.apply(nfx.remove_street_address)
    text = text.apply(nfx.remove_shortwords)
    text = text.apply(nfx.remove_userhandles)
    text = text.apply(nfx.remove_puncts)
    text = text.apply(nfx.remove_stopwords)
    return text

def returnDtype(x):
    for s in x:
        # print(type(s))
        if(isinstance(s, float)):
            print("FLOATTTT")
            s = str(s)
            print(type(s))
            # print(s)
            # break

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 

def tokenize(text):
    return [lemmatizer.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(word_tokenize(text))]


#Load Datset
df = pd.read_csv("dataset.csv")
#Delete first and last columns (survey id, CSAT)
del df[df.columns[0]]
del df[df.columns[-1]]

df['Reason for Rating'] = df['Reason for Rating'].astype(str)
df['Clean_Text'] = df['Reason for Rating'].str.strip()

#Call text cleaning
df['Clean_Text'] = cleanText(df['Clean_Text'])

#Call Tokenization/Lemmatization
df['Tokenized'] = df['Clean_Text']
# df.at[0, 'Tokenized'] = 'new value'
for i in range (0, len(df['Clean_Text'])):
    delimiter = ' '
    df.at[i, 'Tokenized'] = delimiter.join(tokenize(df['Clean_Text'][i]))

#Transforming NPS values into sentiments
#0-6 negative, 7-8 neutral, 9-10 positive
def senti(x):
    if x>=0 and x<=6:
        return 'negative'
    elif x>=7 and x<=8:
        return 'neutral'
    elif x>=9 and x<=10:
        return 'positive'

# df['NPSsenti'] = df['NPS']
# for entry in range(0, len(df['NPS'])):
#     df['NPSsenti'][entry] = senti(df['NPS'][entry])

def npsToSenti(entry):
    entry = entry.apply(senti)
    return entry

df['NPS'] = npsToSenti(df['NPS'])

#Feature Extraction
# Xfeatures = df['Clean_Text']
Xfeatures = df['Tokenized']
ylabels = df['NPS']

#Vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(Xfeatures)

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=5)

#Build Model

#===================  MULTINOMIAL NB ===================

# mnb = MultinomialNB()
# mnb.fit(X_train, y_train)

# y_predict = mnb.predict(X_test)

# print(classification_report(y_test, y_predict))
##Accuracy
## print(f'Accuracy: {mnb.score(X_test, y_test)}')

#===================  LOGISTIC REGRESSION ===================

# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# print(f'Accuracy: {lr.score(X_test, y_test)}')

#===================  SVM  ===================

# svmClassifier = svm.SVC(kernel='linear') #79% - neg/pos
# svmClassifier = svm.SVC(kernel='linear', gamma='auto') #80% - neg/pos
# svmClassifier = svm.SVC() #79% - neg/pos


svmClassifier = svm.SVC().fit(X_train, y_train) #72.6190476%
# svmClassifier = svm.SVC(kernel='linear').fit(X_train, y_train) #72.00854700%

y_predict = svmClassifier.predict(X_test)

print(classification_report(y_test, y_predict))
print("Accuracy Score: ", accuracy_score(y_test, y_predict))
print("F1-score: ", f1_score(y_test, y_predict, average='weighted'))