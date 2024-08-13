# Detecting AI Generating Essays

[Streamlit App Link](https://gpt-generated-text-detector.streamlit.app/)

# Objective

Our group project aims to create a model capable of distinguishing between Argumentative Essays generated by AI, such as GPT, and text written by humans. Our goal is to bolster the digital information landscape's transparency and trustworthiness. This endeavor seeks to provide decision-makers with a tool to navigate the challenges presented by AI technology, thus ensuring the integrity and authenticity of digital content.

# Data Collection and Description

The data collection process involved sourcing from two Kaggle datasets to build a comprehensive corpus for analysis:

[DAIGT Proper Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset): This dataset contains AI-generated text from ChatGPT and LIama, along with human-written essays from established training materials and the Persuade corpus, providing a blend of synthetic and organic text samples.

[ArguGPT](https://www.kaggle.com/datasets/alejopaullier/argugpt?select=argugpt.csv): Encompassing approximately 4,000 essays, this dataset adds depth with AI-generated content from seven different GPT models, offering a spectrum of linguistic patterns for examination.

# Data Preprocessing Methodology

The initial phase of methodology focused on data preprocessing, involving the conversion of all text to lowercase to ensure uniformity, and the removal of new lines, special characters, and symbols like '\n' that could skew analysis. Any indicative words or phrases, such as "AI assistant" or "I'm sorry to hear that," which could directly reveal the source of the text, were either excised or marked for feature creation. Spaces at the beginning and end of texts, along with extraneous punctuation like commas and full stops, were also eliminated. Furthermore, common stopwords were removed to distill the text down to its most informative elements. The preprocessed data was then subjected to t-tests to elicit the distinct usage patterns of various parts of speech (POS) tags between the AI-generated and human-written texts, analyzing the differences through their Term Frequency-Inverse Document Frequency (TF-IDF) scores


# Feature Selection Approach

For the feature selection in modeling, two main categories were employed: TF-IDF features and POS-specific variables. The TF-IDF features were calculated to reflect the significance of words within a document relative to the corpus, highlighting the importance of words in context. The POS variables, representing the TF-IDF scores for grammatical categories such as nouns, verbs, determiners, adjectives, auxiliary verbs, coordinating conjunctions, and particles, were extracted to complement the traditional TF-IDF. This approach not only focused on the frequency of word occurrence but also on capturing the nuanced linguistic and syntactic characteristics inherent in the text. The creation of these features involved tokenizing the text and assigning POS tags, followed by the computation and standardization of the TF-IDF scores for each grammatical category, thereby equipping the model with a multifaceted representation of the text data

# Results and Insights

The Naïve Bayes model with lemmatization was chosen primarily for its high recall rate, critical in accurately identifying AI-generated content. Although the model has shown a tendency to overfit, especially on unseen datasets, and performs best on texts similar to those it was trained on, it presents a reliable first step in detecting AI-authored essays. Future improvements could include diversifying the training dataset to reduce overfitting and expanding the dataset size for more robust learning. Additionally, incorporating new features like sentiment distribution could refine the model's predictive capabilities. This foundational model paves the way for more sophisticated versions that could play a pivotal role in educational integrity, legal document authentication, and shaping the public's understanding of AI in the information age























