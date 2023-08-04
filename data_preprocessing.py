import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

def encode_and_add_padding(sentences, seq_length, word_index):
  sent_encoded = []
  for sent in sentences:
    temp_encoded = [word_index[word] if word in word_index else word_index['[UNKNOWN]'] for word in sent]
    if len(temp_encoded) < seq_length:
      temp_encoded += [word_index['[PAD]']] * (seq_length - len(temp_encoded))
    if len(temp_encoded) > seq_length:
      temp_encoded = temp_encoded[:seq_length]
    sent_encoded.append(temp_encoded)
  return sent_encoded

def preprocess_dataset():
  data = pd.read_csv("Data/IMDB Dataset.csv")
  data = data.sample(1000,random_state=24)
  print("Size of the dataset: {0}".format(len(data)))

  text = data["review"].tolist()
  label = data["sentiment"].tolist()
  print(text[1:10])
  print(label[1:10])

  def remove_clean(x):
      x = re.sub(r'<br /><br />','',x)
      x = x.lower()
      x = re.sub(r'[^\w\s]','',x)
      return x

  text_clean = [remove_clean(s) for s in text]

  print(text_clean[1:10])

  nltk.download('punkt')
  text_clean = [word_tokenize(sentence) for sentence in text_clean]

  print(text_clean[1:10])

  nltk.download('stopwords')
  from nltk.corpus import stopwords as sw
  stop_words = sw.words('english')

  text_cleaned =[]
  for tokens in text_clean:
      filtered_sentence = [w for w in tokens if not w in stop_words]
      text_cleaned.append(filtered_sentence)

  print(text_cleaned[1:10])

  nltk.download('wordnet')
  nltk.download('omw-1.4')

  lemmatizer = WordNetLemmatizer()

  text_final = []
  for tokens in text_cleaned:
      lemma_sentence = [lemmatizer.lemmatize(w) for w in tokens]
      text_final.append(lemma_sentence)

  print(text_final[1:10])

  word_set = set()
  for sent in text_final:
    for word in sent:
      word_set.add(word)

  word_set.add('[PAD]')
  word_set.add('[UNKNOWN]')

  word_list = list(word_set)

  word_list.sort()

  word_index = {}
  ind = 0
  for word in word_list:
    word_index[word] = ind
    ind += 1

  emb_dim = 100

  seq_length = 100
  text_pad_encoded = encode_and_add_padding(text_final, seq_length, word_index)

  from sklearn.preprocessing import LabelEncoder
  import numpy as np

  unique_labels = np.unique(label)

  lEnc = LabelEncoder()
  label_encoded = lEnc.fit_transform(label)

  n_class = len(unique_labels)

  train_pad_encoded,test_pad_encoded,label_train_encoded,label_test_encoded = train_test_split(text_pad_encoded,label_encoded,test_size=0.25,random_state=42)

  return n_class, text_final, word_index, label, emb_dim, word_list, label_encoded, train_pad_encoded,test_pad_encoded,label_train_encoded,label_test_encoded, text_pad_encoded
