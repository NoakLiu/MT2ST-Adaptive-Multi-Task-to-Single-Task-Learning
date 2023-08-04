import pandas as pd
import re
import os
import torch
import numpy as np

from classification_func import classification, classification_lr
from update_embed import update_embedding, update_embedding0, update_embedding1, update_embedding2, update_embedding3
from data_preprocessing import preprocess_dataset
from eval_func import sim_loss_cal, word_sim_dir,word_vec_file, read_word_vectors


def original_embedding(word_model):
  emb_table = []
  for i, word in enumerate(word_list):
    if word in word_model:
      emb_table.append(word_model[word])
    else:
      emb_table.append([0]*emb_dim)
  emb_table = np.array(emb_table)
  return emb_table

n_class, text_final, word_index, label, emb_dim, word_list, label_encoded, train_pad_encoded,test_pad_encoded,label_train_encoded,label_test_encoded,text_pad_encoded=preprocess_dataset()

# CBOW
from gensim.models import Word2Vec
wv_cbow_model = Word2Vec(sentences = text_final, vector_size=100, window=5, min_count=5, workers=2, sg=0)
word2vec_cbow_emb = original_embedding(wv_cbow_model.wv)
original_word2vec_cbow_acc = classification_lr(word2vec_cbow_emb,0.05, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded)
word2vec_cbow_emb = original_embedding(wv_cbow_model.wv)
word2vec_cbow_emb = update_embedding0(word2vec_cbow_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded, text_pad_encoded)
original_word2vec_cbow_acc = classification(np.array(list(word2vec_cbow_emb.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
word2vec_cbow_emb = original_embedding(wv_cbow_model.wv)
new_word2vec_cbow_emb = update_embedding0(word2vec_cbow_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
word2vec_cbow_acc = classification(np.array(list(new_word2vec_cbow_emb.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
word2vec_cbow_emb = original_embedding(wv_cbow_model.wv)
new_word2vec_cbow_emb = update_embedding1(word2vec_cbow_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
word2vec_cbow_acc = classification(np.array(list(new_word2vec_cbow_emb.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
word2vec_cbow_emb = original_embedding(wv_cbow_model.wv)
new_word2vec_cbow_emb = update_embedding2(word2vec_cbow_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
word2vec_cbow_acc = classification(np.array(list(new_word2vec_cbow_emb.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
word2vec_cbow_emb = original_embedding(wv_cbow_model.wv)
new_word2vec_cbow_emb = update_embedding3(word2vec_cbow_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
word2vec_cbow_acc = classification(np.array(list(new_word2vec_cbow_emb.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)

# Skip-Gram
wv_sg_model = Word2Vec(sentences = text_final, vector_size=100, window=5, min_count=5, workers=2, sg=1)
word2vec_sg_emb = original_embedding(wv_sg_model.wv)
original_word2vec_sg_acc = classification_lr(word2vec_cbow_emb,0.05, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded)
word2vec_sg_emb0 = update_embedding0(word2vec_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
original_word2vec_sg_acc0 = classification(np.array(list(word2vec_sg_emb0.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
word2vec_sg_emb1 = update_embedding1(word2vec_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
original_word2vec_sg_acc1 = classification(np.array(list(word2vec_sg_emb1.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
word2vec_sg_emb2 = update_embedding2(word2vec_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
original_word2vec_sg_acc2 = classification(np.array(list(word2vec_sg_emb2.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
word2vec_sg_emb3 = update_embedding0(word2vec_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
original_word2vec_sg_acc3 = classification(np.array(list(word2vec_sg_emb3.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)


# CBOW Fast-Text
from gensim.models import FastText
fasttext_cbow_model = FastText(sentences = text_final, vector_size=100, window=5, min_count=5, workers=2, sg=0)
fasttext_cbow_emb = original_embedding(fasttext_cbow_model.wv)
original_fasttext_cbow_acc = classification(fasttext_cbow_emb, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_cbow_emb0 = update_embedding0(fasttext_cbow_emb,n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_cbow_acc0 = classification(np.array(list(new_fasttext_cbow_emb0.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_cbow_emb1 = update_embedding1(fasttext_cbow_emb,n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_cbow_acc1 = classification(np.array(list(new_fasttext_cbow_emb1.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_cbow_emb2 = update_embedding2(fasttext_cbow_emb,n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_cbow_acc2 = classification(np.array(list(new_fasttext_cbow_emb2.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_cbow_emb3 = update_embedding3(fasttext_cbow_emb,n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_cbow_acc3 = classification(np.array(list(new_fasttext_cbow_emb3.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)

# Skip-Gram Fast-Text
from gensim.models import FastText
fasttext_sg_model = FastText(sentences = text_final, vector_size=100, window=5, min_count=5, workers=2, sg=1)
fasttext_sg_emb = original_embedding(fasttext_sg_model.wv)
original_fasttext_sg_acc = classification(fasttext_sg_emb, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_sg_emb0 = update_embedding0(fasttext_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_sg_acc0 = classification(np.array(list(new_fasttext_sg_emb0.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_sg_emb1 = update_embedding1(fasttext_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_sg_acc1 = classification(np.array(list(new_fasttext_sg_emb1.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_sg_emb2 = update_embedding2(fasttext_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_sg_acc2 = classification(np.array(list(new_fasttext_sg_emb2.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_fasttext_sg_emb3 = update_embedding3(fasttext_sg_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_fasttext_sg_acc3 = classification(np.array(list(new_fasttext_sg_emb3.values())),n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)

# Pretrained embedding Glove-Twitter
import gensim.downloader as api
wv_emb_twitter_100 = api.load("glove-twitter-100")
wv_twitter_100_emb = original_embedding(wv_emb_twitter_100)
original_wv_twitter_100_acc = classification(wv_twitter_100_emb, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_wv_twitter_100_emb0 = update_embedding0(wv_twitter_100_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_wv_twitter_100_acc0 = classification(np.array(list(new_wv_twitter_100_emb0.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_wv_twitter_100_emb1 = update_embedding1(wv_twitter_100_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_wv_twitter_100_acc1 = classification(np.array(list(new_wv_twitter_100_emb1.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_wv_twitter_100_emb2 = update_embedding2(wv_twitter_100_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_wv_twitter_100_acc2 = classification(np.array(list(new_wv_twitter_100_emb2.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_wv_twitter_100_emb3 = update_embedding3(wv_twitter_100_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
new_wv_twitter_100_acc3 = classification(np.array(list(new_wv_twitter_100_emb3.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)

# Spareseman Loss Similarity
sim_loss = sim_loss_cal(word_sim_dir,word_vec_file)
wordvecs = read_word_vectors(word_vec_file)
word2vec_cbow_emb = original_embedding(wv_cbow_model.wv)
original_word2vec_cbow_acc = classification(word2vec_cbow_emb, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_word2vec_cbow_emb = update_embedding(word2vec_cbow_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
word2vec_cbow_acc = classification(np.array(list(new_word2vec_cbow_emb.values())),  n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
new_word2vec_cbow_emb3 = update_embedding3(word2vec_cbow_emb, n_class, text_final, word_index, label, emb_dim, word_list, test_pad_encoded, label_encoded)
word2vec_cbow_acc = classification(np.array(list(new_word2vec_cbow_emb3.values())), n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, text_pad_encoded)
