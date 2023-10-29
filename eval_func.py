from gensim.models import Word2Vec

import numpy as np

def word_analogy_evaluation(vectors_file): 
  import torch
  # in this function the vector file should be devrivative from the start to the return
  # so we need not to make any changes on this file to make the loss to be derivative
  # however, there are few changes here: like 
  # wait, firstly, we need input a torch tensor but not a np array (for the vectors file)
  # secondly, we need no use of like these sentences:
  # assignment      W[vocab[word], :] = torch.from_numpy(v)
  # slice     pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]+  W[ind3[subset], :])
  # assginemt      expected_matrix[ind4[subset[setid]]][setid]=1 #第i个列最大的数值为1，为真值，其行坐标为ind14[i]
  # no change for the leaf node
  # tensor ---> req-grad =False--> f(tensor)--->req-grad =True->return value
  # for f, only computation formulas include tensor, no value changes for this one

  vectors = vectors_file

  vocab_words=list(vectors.keys())
  vocab_size = len(vocab_words)

  # create word->index and index->word converter
  vocab = {w: idx for idx, w in enumerate(vocab_words)}
  ivocab = {idx: w for idx, w in enumerate(vocab_words)}

  # create the embedding matrix of shape (vocab_size, dim)
  vector_dim = len(vectors[ivocab[0]])
  #W = np.zeros((vocab_size, vector_dim))
  #W = torch.from_numpy(W)
  W = torch.zeros(vocab_size, vector_dim).float()
  #print(W.requires_grad)
  #print(W.dtype)
  #W.requires_grad_(True)
  for word, v in vectors.items():
      if word == '<unk>':
          continue
      W[vocab[word], :] = torch.from_numpy(v)

  # normalize each word vector to unit length
  # Vectors are usually normalized to unit length before they are used for similarity calculation, making cosine similarity and dot-product equivalent.
  W_norm = torch.zeros(vocab_size, vector_dim).float()
  d = (torch.sum(W ** 2, 1) ** (0.5))
  W_norm = (W.T / d).T
    
  def evaluate_vectors(W,vocab, prefix='./eval/question-data/'):
        """Evaluate the trained word vectors on a variety of tasks"""

        filenames = [
            'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
            'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
            'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
            'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
            'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
            ]

        # to avoid memory overflow, could be increased/decreased
        # depending on system and vocab size
        split_size = 100

        correct_sem = 0; # count correct semantic questions
        correct_syn = 0; # count correct syntactic questions
        correct_tot = 0 # count correct questions
        count_sem = 0; # count all semantic questions
        count_syn = 0; # count all syntactic questions
        count_tot = 0 # count all questions
        full_count = 0 # count all questions, including those with unknown words

        for i in range(len(filenames)):
            with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
                full_data = [line.rstrip().split(' ') for line in f]
                full_count += len(full_data)
                data = [x for x in full_data if all(word in vocab for word in x)]

            if len(data) == 0:
                #print("ERROR: no lines of vocab kept for %s !" % filenames[i])
                #print("Example missing line:", full_data[0])
                continue

            indices = np.array([[vocab[word] for word in row] for row in data])#Note: Great Problem
            indices = torch.LongTensor(indices)
            ind1, ind2, ind3, ind4 = indices.T
            ind1 = torch.LongTensor(ind1)
            ind2 = torch.LongTensor(ind2)
            ind3 = torch.LongTensor(ind3)
            ind4 = torch.LongTensor(ind4)

            predictions = np.zeros((len(indices),))
            predictions = torch.from_numpy(predictions)
            num_iter = int(np.ceil(len(indices) / float(split_size)))
            tot_loss1 = 0
            tot_loss2 = 0
            for j in range(num_iter): #每100个连续的词语做成一个batch进行训练
                subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1))) #subset用于对set中id->整个corpus词语id的重新定位
                subset = torch.LongTensor(subset)
                
                pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                    +  W[ind3[subset], :])

                # normalization
                W_norm_pred = np.zeros(pred_vec.shape)
                W_norm_pred = torch.from_numpy(W_norm_pred)
                d_pred = (torch.sum(pred_vec ** 2, 1) ** (0.5))
                W_norm_pred = (pred_vec.T / d_pred).T

                dist = torch.mm(W, pred_vec.T)

                for k in range(len(subset)):
                    dist[ind1[subset[k]], k] = 0
                    dist[ind2[subset[k]], k] = 0
                    dist[ind3[subset[k]], k] = 0

                # predicted word index
                tmp = torch.argmax(dist,0).flatten()
                tmp = torch.FloatTensor(tmp.float())
                predictions[subset] = tmp.double()#torch.argmax(dist,0).flatten()
            
            dist.requires_grad_(True)
            
            ### construct expected matrix
            expected_matrix = 0.01*np.ones((W.shape[0],len(subset))) #所有词语数目(用于找到wordvec最相近词),此过程中的ind4数目
            expected_matrix = torch.from_numpy(expected_matrix)
            for setid in range(0,len(subset)): #ind4[i]代表对应的正确id
                expected_matrix[ind4[subset[setid]]][setid]=1 #第i个列最大的数值为1，为真值，其行坐标为ind14[i]
            
            expected_matrix.requires_grad_(True)
            
           ### calculate the angle between [0.01,,,,1,,,0.01]
            tot_loss2 += torch.sum(dist*expected_matrix) #similarity of prediction result matrix

            val = (ind4 == predictions) # correct predictions
            for sid in range(0,len(ind4)):
                if(ind4[sid]!=predictions[sid]):
                    tot_loss1 += torch.sum(W[ind4[sid]]*W[int(predictions[sid])])
            count_tot = count_tot + len(ind1)
            correct_tot = correct_tot + sum(val)
            if i < 5:
                count_sem = count_sem + len(ind1)
                correct_sem = correct_sem + sum(val)
            else:
                count_syn = count_syn + len(ind1)
                correct_syn = correct_syn + sum(val)
                
            tot_loss1.requires_grad_(True)
            tot_loss2.requires_grad_(True)
        
        return correct_sem, correct_syn, correct_tot, count_sem, count_syn, count_tot, full_count, tot_loss1, tot_loss2

  correct_sem, correct_syn, correct_tot, count_sem, count_syn, count_tot, full_count,tot_loss1, tot_loss2 \
  = evaluate_vectors(W,vocab, prefix='GloVe/eval/question-data')
    
  semantic_acc = 100 * correct_sem / float(count_sem)
  syntactic_acc = 100 * correct_syn / float(count_syn)
  total_acc = 100 * correct_tot / float(count_tot)
  tot_loss1 =  100*tot_loss1 / float(vocab_size) #100--embedding vector length
  tot_loss2 =tot_loss2 / float(vocab_size) #average of similarity

  #print(total_acc)
  #print(tot_loss1)
  #print(tot_loss2)
  
  return semantic_acc, syntactic_acc, total_acc, tot_loss1, tot_loss2

import math
import numpy
from operator import itemgetter
from numpy.linalg import norm
import sys
import os
import gzip

EPSILON = 1e-6

def euclidean(vec1, vec2):
  diff = vec1 - vec2
  return math.sqrt(diff.dot(diff))

def cosine_sim(vec1, vec2):
  vec1 += EPSILON * numpy.ones(len(vec1))
  vec2 += EPSILON * numpy.ones(len(vec1))
  return vec1.dot(vec2)/(norm(vec1)*norm(vec2)+1)

def assign_ranks(item_dict):
  ranked_dict = {}
  sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(),
                                                     key=itemgetter(1),
                                                     reverse=True)]
  for i, (key, val) in enumerate(sorted_list):
    same_val_indices = []
    for j, (key2, val2) in enumerate(sorted_list):
      if val2 == val:
        same_val_indices.append(j+1)
    if len(same_val_indices) == 1:
      ranked_dict[key] = i+1
    else:
      ranked_dict[key] = 1.*sum(same_val_indices)/(len(same_val_indices)+1)
  return ranked_dict

def correlation(dict1, dict2):
  avg1 = 1.*sum([val for key, val in dict1.iteritems()])/len(dict1)
  avg2 = 1.*sum([val for key, val in dict2.iteritems()])/len(dict2)
  numr, den1, den2 = (0., 0., 0.)
  for val1, val2 in zip(dict1.itervalues(), dict2.itervalues()):
    numr += (val1 - avg1) * (val2 - avg2)
    den1 += (val1 - avg1) ** 2
    den2 += (val2 - avg2) ** 2
  return numr / math.sqrt(den1 * den2)

def spearmans_rho(ranked_dict1, ranked_dict2):
  assert len(ranked_dict1) == len(ranked_dict2)
  if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
    return 0.
  x_avg = 1.*sum([val for val in ranked_dict1.values()])/len(ranked_dict1)
  y_avg = 1.*sum([val for val in ranked_dict2.values()])/len(ranked_dict2)
  num, d_x, d_y = (0., 0., 0.)
  for key in ranked_dict1.keys():
    xi = ranked_dict1[key]
    yi = ranked_dict2[key]
    num += (xi-x_avg)*(yi-y_avg)
    d_x += (xi-x_avg)**2
    d_y += (yi-y_avg)**2
  return num/(math.sqrt(d_x*d_y)+1)

from collections import Counter
from operator import itemgetter

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):
  word_vecs = {}
  #if filename.endswith('.gz'): file_object = gzip.open(filename, 'r')
  #else: file_object = open(filename, 'r')
  print(filename)
  file_object = open(str(filename),"r")

  for line_num, line in enumerate(file_object):
    line = line.strip().lower()
    word = line.split()[0]
    word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vec_val in enumerate(line.split()[1:]):
      word_vecs[word][index] = float(vec_val)
    ''' normalize weight vector '''
    word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return word_vecs

word_sim_dir = "./en"
word_vec_file = "./embed/filtered.txt"

def sim_loss_cal(word_sim_dir,word_vec_file):
  word_vecs = read_word_vectors(word_vec_file)
  #print(word_vecs)
  print('=================================================================================')
  print("%6s" %"Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
  print('=================================================================================')

  avg_spearmans_rho = 0

  for i, filename in enumerate(os.listdir(word_sim_dir)):
      manual_dict, auto_dict = ({}, {})
      not_found, total_size = (0, 0)
      for line in open(os.path.join(word_sim_dir, filename),'r'):
          line = line.strip().lower()
          word1, word2, val = line.split()
          if word1 in word_vecs and word2 in word_vecs:
              manual_dict[(word1, word2)] = float(val)
              auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
          else:
              not_found += 1
          total_size += 1
      print("%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size), end=""),
      print("%15s" % str(not_found), end=""),
      print("%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
      avg_spearmans_rho +=spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
  avg_spearmans_rho /= len(os.listdir(word_sim_dir))
  return avg_spearmans_rho

def sim_loss_cal2(word_sim_dir,word_vec_file):
  word_vecs = word_vec_file
  avg_spearmans_rho = 0

  for i, filename in enumerate(os.listdir(word_sim_dir)):
      manual_dict, auto_dict = ({}, {})
      not_found, total_size = (0, 0)
      for line in open(os.path.join(word_sim_dir, filename),'r'):
          line = line.strip().lower()
          word1, word2, val = line.split()
          if word1 in word_vecs and word2 in word_vecs:
              manual_dict[(word1, word2)] = float(val)
              auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
          else:
              not_found += 1
          total_size += 1
      avg_spearmans_rho +=spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
  avg_spearmans_rho /= len(os.listdir(word_sim_dir))
  return avg_spearmans_rho