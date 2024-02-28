#DataFrame manipulation
import pandas as pd
#Text cleaning
import neattext.functions as nfx
#Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Split dataset
from sklearn.model_selection import train_test_split
#Build Model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def cleanText(text):
    text = text.apply(nfx.remove_street_address)
    text = text.apply(nfx.remove_shortwords)
    text = text.apply(nfx.remove_userhandles)
    text = text.apply(nfx.remove_puncts)
    text = text.apply(nfx.remove_stopwords)
    return text

#Load Datset
df = pd.read_csv("dataset.csv")
#Delete first and last columns (survey id, CSAT)
del df[df.columns[0]]
del df[df.columns[0]]

df['Reason for Rating'] = df['Reason for Rating'].astype(str)
df['Clean_Text'] = df['Reason for Rating'].str.strip()
df['Clean_Text'] = cleanText(df['Clean_Text'])

# print(df['Clean_Text'])

#Transforming CSAT values into sentiments
#0-6 negative, 7-8 neutral, 9-10 positive
def senti(x):
    if x>=0 and x<=6:
        return 'negative'
    elif x>=7 and x<=8:
        return 'neutral'
    elif x>=9 and x<=10:
        return 'positive'

def csatToSenti(entry):
    entry = entry.apply(senti)
    return entry

df['CSAT'] = csatToSenti(df['CSAT'])
# print(df['CSAT'])

#Feature Extraction
Xfeatures = df['Clean_Text']
ylabels = df['CSAT']

#Vectorizer
# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(Xfeatures)
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

#Build Model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# print(f'Accuracy: {lr.score(X_test, y_test)}')

#Accuracy
print(f'Accuracy: {mnb.score(X_test, y_test)}')