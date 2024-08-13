

import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def nltk_pos_tag_to_wordnet_pos(nltk_tag):
    """
    Convert nltk POS tag to the format recognized by WordNetLemmatizer
    """
    if nltk_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None

def lemmatize_text(text):
    """
    Function to lemmatize the input text.
    """
    tokens = word_tokenize(text)
    nltk_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for token, tag in nltk_tags:
        wordnet_pos = nltk_pos_tag_to_wordnet_pos(tag)  # Convert the tag to wordnet format
        if wordnet_pos is None:
            lemmatized_tokens.append(token)
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(token, wordnet_pos))
    return ' '.join(lemmatized_tokens)

def clean_text(text):
    """
    Clean and lemmatize text.
    """
    if not isinstance(text, str):
        text = str(text)
        text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    specific_words = ['ai assistant', "i'm sorry to hear that"]
    for word in specific_words:
        text = text.replace(word, '')
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    # Lemmatization added here
    lemmatized_text = lemmatize_text(' '.join(filtered_text))
    return lemmatized_text.strip()

def pos_proportion(text):
    '''
    Function to calculate the proportion of specified POS tags in text
    :param text:
    :return:
    '''
    # Check if text is not a string
    if not isinstance(text, str):
        # Handle non-string input (e.g., NaN values)
        return {"NOUN": 0, "VERB": 0, "DET": 0, "ADJ": 0, "AUX": 0, "CCONJ": 0, "PART": 0}
    # Tokenize the text
    tokens = word_tokenize(text)
    # POS tagging
    tagged = nltk.pos_tag(tokens)
    # Count the occurrences of the specified POS tags
    pos_counts = {"NOUN": 0, "VERB": 0, "DET": 0, "ADJ": 0, "AUX": 0, "CCONJ": 0, "PART": 0}
    for word, tag in tagged:
        if tag.startswith('NN'):  # Nouns
            pos_counts["NOUN"] += 1
        elif tag.startswith('VB'):  # Verbs
            pos_counts["VERB"] += 1
        elif tag == 'DT':  # Determiners
            pos_counts["DET"] += 1
        elif tag.startswith('JJ'):  # Adjectives
            pos_counts["ADJ"] += 1
        elif tag.startswith('MD'):  # Auxiliary verbs
            pos_counts["AUX"] += 1
        elif tag == 'CC':  # Coordinating conjunctions
            pos_counts["CCONJ"] += 1
        elif tag == 'RP':  # Particles
            pos_counts["PART"] += 1
    # Calculate proportions
    total = sum(pos_counts.values())
    if total > 0:
        for key in pos_counts:
            pos_counts[key] /= total
    return pos_counts


def calculate_noun_proportion(pos_counts):
    '''
    Function to calculate the proportion of nouns
    :param pos_counts:
    :return:
    '''
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    noun_count = sum(pos_counts.get(tag, 0) for tag in noun_tags)
    total_count = sum(pos_counts.values())
    return noun_count / total_count if total_count > 0 else 0


def get_pos_counts(text):
    '''
    Function to perform POS tagging and return POS counts
    :param text:
    :return:
    '''
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    pos_counts = {}
    for word, tag in tagged_tokens:
        if tag in pos_counts:
            pos_counts[tag] += 1
        else:
            pos_counts[tag] = 1
    return pos_counts


def transform_to_pos(text):
    '''
    POS Tagging and Transformation
    :param text:
    :return:
    '''
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    # Return a string where each word is replaced by its POS tag
    return ' '.join([tag for _, tag in tagged])


df = pd.read_excel("df.xlsx", sheet_name='in', usecols='A, B')
print(df.head(5))
# =============================================================================
# preprocessing
# =============================================================================
# Apply the clean_text function to the 'text' column of the DataFrame
df['cleaned_text'] = df['text'].apply(clean_text)
print(df[['text', 'cleaned_text']].head())


# =============================================================================
# EDA
# 1. Tokenize the cleaned_text using NLTK's word_tokenize.
# 2. Use NLTK's pos_tag to tag each token with its part of speech.
# 3. Count the occurrences of the specified POS tags.
# 4. Calculate the proportions of these tags relative to the total number of tokens in the text.
# =============================================================================
df['POS_proportions'] = df['cleaned_text'].apply(pos_proportion)
print(df[['text', 'POS_proportions']])


# Apply the function to 'cleaned_text' column to create a new 'POS_counts' column
df['POS_counts'] = df['cleaned_text'].apply(get_pos_counts)

# TF-IDF
df['POS_text'] = df['cleaned_text'].apply(transform_to_pos)
# Apply TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['POS_text'])

# Convert the TF-IDF matrix to a DataFrame for easier analysis
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# tfidf_df contains the TF-IDF scores for each POS tag in each document
print(tfidf_df.head())
# After applying TfidfVectorizer
print(tfidf_df.columns)

# Define a mapping of target categories to their corresponding POS tags
pos_tag_mapping = {
    "NOUN": ['nn', 'nns', 'nnp', 'nnps'],
    "VERB": ['vb', 'vbd', 'vbg', 'vbn', 'vbp', 'vbz'],
    "DET": ['dt'],
    "ADJ": ['jj', 'jjr', 'jjs'],
    "AUX": ['md'],
    "CCONJ": ['cc'],
    "PART": ['rp']
}

# Initialize a column for the TF-IDF sum of each category in the DataFrame to 0
for category in pos_tag_mapping:
    df[category + "_TFIDF"] = 0.0

# Calculate the sum of TF-IDF scores for the tags in each category
for category, tags in pos_tag_mapping.items():
    df[category + "_TFIDF"] = tfidf_df[tags].sum(axis=1)


#df.to_excel('/Users/user/Desktop/INSY 669/group project/df_adding_clean_texts_POS_proportions_tfidf2.xlsx', index=False)

# Split the DataFrame based on the label
df_label_0 = df[df['label'] == 0]
df_label_1 = df[df['label'] == 1]

# List of POS tag categories
categories = ["NOUN", "VERB", "DET", "ADJ", "AUX", "CCONJ", "PART"]

# Perform t-tests and print results
for category in categories:
    t_stat, p_value = ttest_ind(df_label_0[f'{category}_TFIDF'], df_label_1[f'{category}_TFIDF'])
    print(f"{category} - T-statistic: {t_stat}, P-value: {p_value}")
    
 '''
Final result interpretations

The results from the t-tests provide insights into the differences in the usage of various parts of speech (POS) tags between texts with labels 0 and 1, based on their TF-IDF scores. Here's how to interpret the results:
NOUN
T-statistic: 9.114744374802482
P-value: 9.401746959990088e-20


The positive t-statistic indicates that texts with label 0 have a higher mean TF-IDF score for nouns compared to texts with label 1. The extremely small p-value suggests that this difference is statistically significant, meaning it's highly unlikely to have occurred by chance.
VERB
T-statistic: 11.318473878497102
P-value: 1.6074425378666074e-29


Similar to nouns, verbs also show a higher mean TF-IDF score in texts with label 0 compared to those with label 1, with the difference being statistically significant.
DET (Determiners)
T-statistic: 14.373591608675191
P-value: 2.1957782670126075e-46


Determiners follow the same pattern as nouns and verbs, with a statistically significant higher mean TF-IDF score in texts with label 0.
ADJ (Adjectives)
T-statistic: -27.341147894179855
P-value: 8.701420680417221e-159


The negative t-statistic here indicates that texts with label 1 have a higher mean TF-IDF score for adjectives compared to texts with label 0. The difference is statistically significant, suggesting a distinct usage pattern of adjectives between the two groups.
AUX (Auxiliary Verbs)
T-statistic: 22.089852060200528
P-value: 1.2993057110410845e-105


Auxiliary verbs show a higher mean TF-IDF score in texts with label 0, with the difference being statistically significant.
CCONJ (Coordinating Conjunctions)
T-statistic: 3.3772931743930887
P-value: 0.0007348232267293067


Although the difference is smaller compared to other categories, coordinating conjunctions still show a statistically significant higher mean TF-IDF score in texts with label 0.
PART (Particles)
T-statistic: 8.333728721020956
P-value: 8.86384390832682e-17


Particles, like most other POS tags, have a higher mean TF-IDF score in texts with label 0, and the difference is statistically significant.

Overall Interpretation

The results suggest that there are statistically significant differences in the usage of various POS tags between texts with labels 0 and 1. Specifically, nouns, verbs, determiners, auxiliary verbs, coordinating conjunctions, and particles are used more prominently (or are more important based on TF-IDF scores) in texts with label 0. In contrast, adjectives are more prominent in texts with label 1. These differences could reflect varying writing styles, topics, or other factors that distinguish the two groups of texts. The statistical significance of these differences indicates that they are unlikely to be due to random chance.
'''

   

#########################  Modelling ###################
##############  naive bayes with lemmatization and without lemmatization ###################################
data = pd.read_excel("df_adding_clean_texts_POS_proportions_tfidf.xlsx")
data.columns
data.shape
data = data[data['label'].isin([0, 1])].reset_index(drop=True)
data['label'].value_counts()

#missing rows
data.isnull().sum()
data = data.dropna()
data.info()

# Function to lemmatize text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text.lower()) 
    return ' '.join([lemmatizer.lemmatize(w) for w in word_tokens])

# Apply lemmatization to your dataset
data['lemmatized_text'] = data['cleaned_text'].apply(lemmatize_text)

# Creating TF-IDF features for both non-lemmatized and lemmatized text
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_non_lem = tfidf_vectorizer.fit_transform(data['cleaned_text']).toarray()
X_lem = tfidf_vectorizer.fit_transform(data['lemmatized_text']).toarray()

# Assuming 'other_variables' are the names of the other columns you want to include
# Replace 'other_variable1', 'other_variable2', etc., with your actual column names
other_variables = data[['NOUN_TFIDF', 'VERB_TFIDF', 'DET_TFIDF', 'ADJ_TFIDF', 'AUX_TFIDF', 'CCONJ_TFIDF', 'PART_TFIDF']]  # Add more variables as needed
other_variables_scaled = StandardScaler().fit_transform(other_variables)  # Standardize the other variables

# Combine TF-IDF features with other variables
X_non_lem_combined = np.hstack((X_non_lem, other_variables_scaled))
X_lem_combined = np.hstack((X_lem, other_variables_scaled))

# Labels

y = data['label'].astype(int)
y.shape
# Define the  model
clf = MultinomialNB()

# Example with train-test split (you could also use cross-validation as shown previously)
X_train_non_lem, X_test_non_lem, y_train, y_test = train_test_split(X_non_lem_combined, y, test_size=0.2, random_state=42)
X_train_lem, X_test_lem, y_train_lem, y_test_lem = train_test_split(X_lem_combined, y, test_size=0.2, random_state=42)

X_train_non_lem[X_train_non_lem < 0] = 0 #navie bayes does not accept negative values

X_train_lem[X_train_lem < 0] = 0 #navie bayes does not accept negative values

# Train and evaluate the  model without lemmatization
clf.fit(X_train_non_lem, y_train)
y_pred_non_lem = clf.predict(X_test_non_lem)
print("Non-lemmatized Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_non_lem))
#Accuracy: 0.9089544772386193
print("recall:", recall_score(y_test, y_pred_non_lem))
#recall: 0.9144008056394763
print("precision:", precision_score(y_test, y_pred_non_lem))
#precision: 0.903482587064676
print("f1:", f1_score(y_test, y_pred_non_lem))
#f1: 0.9089089089089091
auc = roc_auc_score(y_test, y_pred_non_lem)
print(f"AUC: {auc}")
#AUC: 0.908989667233257
confusion_matrix(y_test, y_pred_non_lem)
#array([[909,  97],
#       [ 85, 908]]


# Train and evaluate the model with lemmatization
clf.fit(X_train_lem, y_train_lem)
y_pred_lem = clf.predict(X_test_lem)
print("\nLemmatized Model Performance:")
print("Accuracy:", accuracy_score(y_test_lem, y_pred_lem))
#Accuracy: 0.9084542271135568
print("recall:", recall_score(y_test_lem, y_pred_lem))
#recall: 0.9204431017119838
print("precision:", precision_score(y_test_lem, y_pred_lem))
#precision: 0.8978388998035364
print("f1:", f1_score(y_test_lem, y_pred_lem))
#f1: 0.9090004972650423
auc = roc_auc_score(y_test_lem, y_pred_lem)
print(f"AUC: {auc}")
#AUC: 0.9085316900210019
confusion_matrix(y_test_lem, y_pred_lem)
#array([[902, 104],
#       [ 79, 914]]



###############################################################################
# =============================================================================
##############KNN with lemmatization and non-lemmatization#####################



# Load your data
data = pd.read_excel("df_adding_clean_texts_POS_proportions_tfidf.xlsx")
data.columns
data.shape
data = data[data['label'].isin([0, 1])].reset_index(drop=True)
data['label'].value_counts()

#missing rows
data.isnull().sum()
data = data.dropna()
data.info()

# Function to lemmatize text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text.lower()) 
    return ' '.join([lemmatizer.lemmatize(w) for w in word_tokens])

# Apply lemmatization to your dataset
data['lemmatized_text'] = data['cleaned_text'].apply(lemmatize_text)

# Creating TF-IDF features for both non-lemmatized and lemmatized text
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_non_lem = tfidf_vectorizer.fit_transform(data['cleaned_text']).toarray()
X_lem = tfidf_vectorizer.fit_transform(data['lemmatized_text']).toarray()

# Assuming 'other_variables' are the names of the other columns you want to include
# Replace 'other_variable1', 'other_variable2', etc., with your actual column names
other_variables = data[['NOUN_TFIDF', 'VERB_TFIDF', 'DET_TFIDF', 'ADJ_TFIDF', 'AUX_TFIDF', 'CCONJ_TFIDF', 'PART_TFIDF']]  # Add more variables as needed
other_variables_scaled = StandardScaler().fit_transform(other_variables)  # Standardize the other variables

# Combine TF-IDF features with other variables
X_non_lem_combined = np.hstack((X_non_lem, other_variables_scaled))
X_lem_combined = np.hstack((X_lem, other_variables_scaled))

# Labels

y = data['label'].astype(int)
y.shape
# Define the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Example with train-test split 
X_train_non_lem, X_test_non_lem, y_train, y_test = train_test_split(X_non_lem_combined, y, test_size=0.2, random_state=42)
X_train_lem, X_test_lem, y_train_lem, y_test_lem = train_test_split(X_lem_combined, y, test_size=0.2, random_state=42)

# Train and evaluate the KNN model without lemmatization
knn_model.fit(X_train_non_lem, y_train)
y_pred_non_lem = knn_model.predict(X_test_non_lem)
print("Non-lemmatized Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_non_lem))
#Accuracy: 0.8189094547273637
print("recall:", recall_score(y_test, y_pred_non_lem))
#recall: 0.8076535750251762
print("precision:", precision_score(y_test, y_pred_non_lem))
#precision: 0.8242548818088387
print("f1:", f1_score(y_test, y_pred_non_lem))
#f1: 0.8158697863682605
auc = roc_auc_score(y_test, y_pred_non_lem)
print(f"AUC: {auc}")
#AUC: 0.8188367278704409
confusion_matrix(y_test, y_pred_non_lem)
#array([[835, 171],
#       [191, 802]]


# Train and evaluate the KNN model with lemmatization
knn_model.fit(X_train_lem, y_train_lem)
y_pred_lem = knn_model.predict(X_test_lem)
print("\nLemmatized Model Performance:")
print("Accuracy:", accuracy_score(y_test_lem, y_pred_lem))
#Accuracy: 0.8214107053526764
print("recall:", recall_score(y_test_lem, y_pred_lem))
#recall: 0.796576032225579
print("precision:", precision_score(y_test_lem, y_pred_lem))
#precision: 0.8361522198731501
print("f1:", f1_score(y_test_lem, y_pred_lem))
#f1: 0.815884476534296
auc = roc_auc_score(y_test_lem, y_pred_lem)
print(f"AUC: {auc}")
#AUC: 0.8212502427529486
confusion_matrix(y_test_lem, y_pred_lem)
#array([[851, 155],
 #      [202, 791]]
 
# Using Naive Bayes, the model with lemmatization performed slightly better than the model without lemmatization and other models 
##### Pipleline ####
### Create Pipeline ####
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import nltk
#import pipeline
from sklearn.pipeline import Pipeline
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')




class TextCleanerLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for text in X:
            text = text.lower()
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            word_tokens = word_tokenize(text)
            filtered_text = [word for word in word_tokens if word not in self.stop_words]
            lemmatized_text = ' '.join([self.lemmatizer.lemmatize(w) for w in filtered_text])
            X_transformed.append(lemmatized_text)
        return X_transformed


df = pd.read_excel("df.xlsx", sheet_name='in', usecols='A, B')
df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
df = df.dropna(subset=['text'])


pipeline = Pipeline([
    ('text_preprocessor', TextCleanerLemmatizer()),
    ('tfidf_vectorizer', TfidfVectorizer(max_features=1000)),
    ('Naive Baise', MultinomialNB()) 
])

X = df['text']  # Adjust column name as needed
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
#Accuracy: 0.9025918541726003
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
#Precision: 0.9194630872483222
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)



# Save the pipeline
dump(pipeline, 'text_classification_pipeline.joblib')




# Load the pipeline
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import nltk
#import pipeline
from sklearn.pipeline import Pipeline
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextCleanerLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for text in X:
            text = text.lower()
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)
            word_tokens = word_tokenize(text)
            filtered_text = [word for word in word_tokens if word not in self.stop_words]
            lemmatized_text = ' '.join([self.lemmatizer.lemmatize(w) for w in filtered_text])
            X_transformed.append(lemmatized_text)
        return X_transformed

# Load the model
from joblib import dump, load
loaded_pipeline = load('text_classification_pipeline.joblib')


new_text_gpt = [
'''The Dual Impact of Social Media on Society

Introduction:
Social media platforms have transformed communication and information access, impacting society profoundly. While they offer numerous benefits, they also present significant challenges.

Positive Impacts:

Communication and Connectivity: Social media removes geographical barriers, fostering global connections and cultural exchanges.

Information Access: These platforms are vital for education and real-time news dissemination, enhancing knowledge and awareness.

Empowerment: Marginalized voices find a platform for advocacy, exemplified by movements like #MeToo and #BlackLivesMatter, highlighting social media's potential for positive societal change.

Negative Impacts:

Mental Health: The correlation between social media use and mental health issues, including anxiety and depression, is concerning, especially among youth.

Misinformation: The rapid spread of false information on these platforms can have dire consequences for public discourse and societal trust.

Privacy Risks: The collection and misuse of personal data by social media companies pose severe privacy concerns, risking user security and autonomy.

Conclusion:
Social media's impact on society is complex, blending benefits with significant drawbacks.
To harness its positive aspects while mitigating negatives, a collaborative effort from platforms, 
users, and policymakers is essential. Addressing privacy, misinformation, 
and mental health concerns can make social media a more positive force in our global community'''
]




# gpt4 generated text  The Dual Impact of Social Media on Society
predictions = loaded_pipeline.predict(new_text_gpt)
print(predictions)
#predicted 1 which is correct

new_text_real = ['''
                 
"Cars. Cars have been around since they became famous in the 1900s, when Henry Ford created and built the first ModelT. Cars have played a major role in our every day lives since then. But now, people are starting to question if limiting car usage would be a good thing. To me, limiting the use of cars might be a good thing to do.

In like matter of this, article, ""In German Suburb, Life Goes On Without Cars,"" by Elizabeth Rosenthal states, how automobiles are the linchpin of suburbs, where middle class families from either Shanghai or Chicago tend to make their homes. Experts say how this is a huge impediment to current efforts to reduce greenhouse gas emissions from tailpipe. Passenger cars are responsible for 12 percent of greenhouse gas emissions in Europe...and up to 50 percent in some carintensive areas in the United States. Cars are the main reason for the greenhouse gas emissions because of a lot of people driving them around all the time getting where they need to go. Article, ""Paris bans driving due to smog,"" by Robert Duffer says, how Paris, after days of nearrecord pollution, enforced a partial driving ban to clear the air of the global city. It also says, how on Monday, motorist with evennumbered license plates were ordered to leave their cars at home or be fined a 22euro fine 31. The same order would be applied to oddnumbered plates the following day. Cars are the reason for polluting entire cities like Paris. This shows how bad cars can be because, of all the pollution that they can cause to an entire city.

Likewise, in the article, ""Carfree day is spinning into a big hit in Bogota,"" by Andrew Selsky says, how programs that's set to spread to other countries, millions of Columbians hiked, biked, skated, or took the bus to work during a carfree day, leaving streets of this capital city eerily devoid of traffic jams. It was the third straight year cars have been banned with only buses and taxis permitted for the Day Without Cars in the capital city of 7 million. People like the idea of having carfree days because, it allows them to lesson the pollution that cars put out of their exhaust from people driving all the time. The article also tells how parks and sports centers have bustled throughout the city uneven, pitted sidewalks have been replaced by broad, smooth sidewalks rushhour restrictions have dramatically cut traffic and new restaurants and upscale shopping districts have cropped up. Having no cars has been good for the country of Columbia because, it has aloud them to repair things that have needed repairs for a long time, traffic jams have gone down, and restaurants and shopping districts have popped up, all due to the fact of having less cars around.

In conclusion, the use of less cars and having carfree days, have had a big impact on the environment of cities because, it is cutting down the air pollution that the cars have majorly polluted, it has aloud countries like Columbia to repair sidewalks, and cut down traffic jams. Limiting the use of cars would be a good thing for America. So we should limit the use of cars by maybe riding a bike, or maybe walking somewhere that isn't that far from you and doesn't need the use of a car to get you there. To me, limiting the use of cars might be a good thing to do."
'''
]
#human written essay

predictions = loaded_pipeline.predict(new_text_real)
print(predictions)
probabilities = loaded_pipeline.predict_proba(new_text_real)
print(np.round(probabilities, 3))

#predicted 0 which is correct