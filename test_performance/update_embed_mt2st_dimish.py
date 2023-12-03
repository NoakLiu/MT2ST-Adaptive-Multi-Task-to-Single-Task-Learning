import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from eval_func import word_analogy_evaluation, word_sim_dir, word_vec_file, sim_loss_cal, sim_loss_cal2
def update_embedding(emb_table, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded, test_pad_encoded,
                      label_test_encoded, alpha):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  losses = []
  train_accs = []

  class Bi_LSTM_Attention(nn.Module):
      def __init__(self):
          super(Bi_LSTM_Attention, self).__init__()
          self.emb = nn.Embedding(emb_table.shape[0], emb_table.shape[1])
          self.emb.weight.data.copy_(torch.from_numpy(emb_table))
          self.emb.weight.requires_grad = True
          self.lstm = nn.LSTM(emb_table.shape[1], n_hidden, bidirectional=True)
          self.encoder_fc = nn.Linear(2 * n_hidden, n_class)
          # self.activation = nn.ReLU()

      # https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb
      # output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      def attention_net(self, output, final_state):
          hidden = final_state.view(-1, n_hidden * 2,
                                    1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
          attn_weights = torch.bmm(output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
          soft_attn_weights = F.softmax(attn_weights, 1).to(device)
          # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
          context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device)
          # context : [batch_size, n_hidden * num_directions(=2)]
          return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

      def forward(self, X):
          x = self.emb(X)  # input : [batch_size, len_seq, embedding_dim]
          x = x.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]
          hidden_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
          hidden_out = torch.cat((final_hidden_state[0, :, :], final_hidden_state[1, :, :]), 1)
          return self.encoder_fc(hidden_out), self.emb

  from sklearn.metrics import accuracy_score
  n_hidden = 256
  n_emb = emb_dim
  total_epoch = 25
  model = Bi_LSTM_Attention().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  input_torch = torch.from_numpy(np.array(train_pad_encoded)).to(device)
  target_torch = torch.from_numpy(np.array(label_train_encoded)).view(-1).to(device)

  for epoch in range(total_epoch):  
      
      model.train()
        
      optimizer.zero_grad()
      
      outputs, lstm_emb = model(input_torch) 
      
      back_emb = {}
      
      for i, word in enumerate(word_list):
        back_emb[word] = lstm_emb(torch.LongTensor([i])).detach().numpy().flatten()
      _, _, total_acc, loss_1, loss_2 = word_analogy_evaluation(back_emb)
      avg_loss = 10.24*(total_acc+loss_1+loss_2)/3
      avg_loss = avg_loss*alpha
      alpha = alpha * alpha
      print("weight adjusted",avg_loss)
      loss = criterion(outputs, target_torch)-avg_loss
      loss.backward()
      optimizer.step()
        
      predicted = torch.argmax(outputs, -1)
      acc= accuracy_score(predicted.cpu().numpy(),target_torch.cpu().numpy())

      losses.append(loss.item())
      train_accs.append(acc)

      print('Epoch: %d, loss: %.5f, train_acc: %.2f' %(epoch + 1, loss.item(), acc))

  print('Finished Training')

  input_torch = torch.from_numpy(np.array(test_pad_encoded)).to(device)

  outputs, _ = model(input_torch)
  predicted = torch.argmax(outputs, -1)

  from sklearn.metrics import classification_report
  print(classification_report(label_test_encoded,predicted.cpu().numpy()))

  return losses, train_accs


def update_embedding0(emb_table, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded, alpha):
  # model for Text
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  import torch.nn.functional as F

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  losses = []
  train_accs = []

  class Bi_LSTM_Attention(nn.Module):
      def __init__(self):
          super(Bi_LSTM_Attention, self).__init__()
          self.emb = nn.Embedding(emb_table.shape[0], emb_table.shape[1])
          self.emb.weight.data.copy_(torch.from_numpy(emb_table))
          self.emb.weight.requires_grad = True
          self.lstm = nn.LSTM(emb_table.shape[1], n_hidden, bidirectional=True)
          self.encoder_fc = nn.Linear(2 * n_hidden, n_class)
          # self.activation = nn.ReLU()

      # https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb
      # output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      def attention_net(self, output, final_state):
          hidden = final_state.view(-1, n_hidden * 2,
                                    1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
          attn_weights = torch.bmm(output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
          soft_attn_weights = F.softmax(attn_weights, 1).to(device)
          # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
          context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device)
          # context : [batch_size, n_hidden * num_directions(=2)]
          return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

      def forward(self, X):
          x = self.emb(X)  # input : [batch_size, len_seq, embedding_dim]
          x = x.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]
          hidden_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
          hidden_out = torch.cat((final_hidden_state[0, :, :], final_hidden_state[1, :, :]), 1)
          return self.encoder_fc(hidden_out), self.emb
      
  print("test_pad.shape:",len(test_pad_encoded))
  print("test_pad[0].shape:",len(test_pad_encoded[0]))

  from sklearn.metrics import accuracy_score
  n_hidden = 256
  n_emb = emb_dim
  total_epoch = 25
  model = Bi_LSTM_Attention().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  input_torch = torch.from_numpy(np.array(train_pad_encoded)).to(device)
  target_torch = torch.from_numpy(np.array(label_train_encoded)).view(-1).to(device)

  for epoch in range(total_epoch):

      model.train()

      optimizer.zero_grad()

      outputs, lstm_emb = model(input_torch)

      back_emb = {}

      for i, word in enumerate(word_list):
        back_emb[word] = lstm_emb(torch.LongTensor([i])).detach().numpy().flatten()
      _, _, total_acc, loss_1, loss_2 = word_analogy_evaluation(back_emb)
      avg_loss = 10.24*(total_acc+loss_1+loss_2)/3
      avg_loss = avg_loss*alpha
      alpha = alpha * alpha
      print("weight adjusted",avg_loss)
      loss = criterion(outputs, target_torch)
      loss.backward()
      optimizer.step()

      predicted = torch.argmax(outputs, -1)
      acc= accuracy_score(predicted.cpu().numpy(),target_torch.cpu().numpy())

      losses.append(loss.item())
      train_accs.append(acc)

      print('Epoch: %d, loss: %.5f, train_acc: %.2f' %(epoch + 1, loss.item(), acc))

  print('Finished Training')

  input_torch = torch.from_numpy(np.array(test_pad_encoded)).to(device)

  outputs,_ = model(input_torch)
  predicted = torch.argmax(outputs, -1)

  from sklearn.metrics import classification_report
  print(classification_report(label_test_encoded,predicted.cpu().numpy()))

  return losses, train_accs

def update_embedding1(emb_table, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded,
                          test_pad_encoded, label_test_encoded, alpha):
  # model for Text
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  import torch.nn.functional as F

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  losses = []
  train_accs = []

  class Bi_LSTM_Attention(nn.Module):
      def __init__(self):
          super(Bi_LSTM_Attention, self).__init__()
          self.emb = nn.Embedding(emb_table.shape[0], emb_table.shape[1])
          self.emb.weight.data.copy_(torch.from_numpy(emb_table))
          self.emb.weight.requires_grad = True
          self.lstm = nn.LSTM(emb_table.shape[1], n_hidden, bidirectional=True)
          self.encoder_fc = nn.Linear(2 * n_hidden, n_class)
          # self.activation = nn.ReLU()

      # https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb
      # output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      def attention_net(self, output, final_state):
          hidden = final_state.view(-1, n_hidden * 2,
                                    1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
          attn_weights = torch.bmm(output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
          soft_attn_weights = F.softmax(attn_weights, 1).to(device)
          # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
          context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device)
          # context : [batch_size, n_hidden * num_directions(=2)]
          return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

      def forward(self, X):
          x = self.emb(X)  # input : [batch_size, len_seq, embedding_dim]
          x = x.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]
          hidden_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
          hidden_out = torch.cat((final_hidden_state[0, :, :], final_hidden_state[1, :, :]), 1)
          return self.encoder_fc(hidden_out), self.emb

  from sklearn.metrics import accuracy_score
  n_hidden = 256
  n_emb = emb_dim
  total_epoch = 25
  model = Bi_LSTM_Attention().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  input_torch = torch.from_numpy(np.array(train_pad_encoded)).to(device)
  target_torch = torch.from_numpy(np.array(label_train_encoded)).view(-1).to(device)

  for epoch in range(total_epoch):

      model.train()

      optimizer.zero_grad()

      print("input_torch", input_torch.shape) #torch.Size([250, 100])

      outputs, lstm_emb = model(input_torch)

      back_emb = {}

      for i, word in enumerate(word_list):
        back_emb[word] = lstm_emb(torch.LongTensor([i])).detach().numpy().flatten()
      _, _, total_acc, loss_1, loss_2 = word_analogy_evaluation(back_emb)
      sim_loss = 33*sim_loss_cal2(word_sim_dir,back_emb)
      sim_loss = sim_loss*alpha
      alpha = alpha * alpha
      print("sparsemans loss",sim_loss)
      loss = criterion(outputs, target_torch)+sim_loss
      loss.backward()
      optimizer.step()

      predicted = torch.argmax(outputs, -1)
      acc= accuracy_score(predicted.cpu().numpy(),target_torch.cpu().numpy())

      losses.append(loss.item())
      train_accs.append(acc)

      print('Epoch: %d, loss: %.5f, train_acc: %.2f' %(epoch + 1, loss.item(), acc))

  print('Finished Training')

  input_torch = torch.from_numpy(np.array(test_pad_encoded)).to(device)

  outputs, _ = model(input_torch)
  predicted = torch.argmax(outputs, -1)

  from sklearn.metrics import classification_report
  print(classification_report(label_test_encoded,predicted.cpu().numpy()))

  return losses, train_accs

def update_embedding2(emb_table, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded,
                        test_pad_encoded, label_test_encoded, alpha):
  # model for Text
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  import torch.nn.functional as F

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  losses = []
  train_accs = []

  class Bi_LSTM_Attention(nn.Module):
      def __init__(self):
          super(Bi_LSTM_Attention, self).__init__()
          self.emb = nn.Embedding(emb_table.shape[0], emb_table.shape[1])
          self.emb.weight.data.copy_(torch.from_numpy(emb_table))
          self.emb.weight.requires_grad = True
          self.lstm = nn.LSTM(emb_table.shape[1], n_hidden, bidirectional=True)
          self.encoder_fc = nn.Linear(2 * n_hidden, n_class)
          # self.activation = nn.ReLU()

      # https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb
      # output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      def attention_net(self, output, final_state):
          hidden = final_state.view(-1, n_hidden * 2,
                                    1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
          attn_weights = torch.bmm(output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
          soft_attn_weights = F.softmax(attn_weights, 1).to(device)
          # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
          context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device)
          # context : [batch_size, n_hidden * num_directions(=2)]
          return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

      def forward(self, X):
          x = self.emb(X)  # input : [batch_size, len_seq, embedding_dim]
          x = x.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]
          hidden_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
          hidden_out = torch.cat((final_hidden_state[0, :, :], final_hidden_state[1, :, :]), 1)
          return self.encoder_fc(hidden_out), self.emb

  from sklearn.metrics import accuracy_score
  n_hidden = 256
  n_emb = emb_dim
  total_epoch = 25
  model = Bi_LSTM_Attention().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  input_torch = torch.from_numpy(np.array(train_pad_encoded)).to(device)
  target_torch = torch.from_numpy(np.array(label_train_encoded)).view(-1).to(device)

  for epoch in range(total_epoch):  
      
      model.train()
        
      optimizer.zero_grad()
      
      outputs, lstm_emb = model(input_torch) 
      
      back_emb = {}
      
      for i, word in enumerate(word_list):
        back_emb[word] = lstm_emb(torch.LongTensor([i])).detach().numpy().flatten()
      _, _, total_acc, loss_1, loss_2 = word_analogy_evaluation(back_emb)
      avg_loss = 0.1024*(total_acc+loss_1+loss_2)/3
      avg_loss = avg_loss * alpha
      alpha = alpha * alpha
      print("weight adjusted",avg_loss)
      loss = criterion(outputs, target_torch)-avg_loss
      loss.backward()
      optimizer.step()
        
      predicted = torch.argmax(outputs, -1)
      acc= accuracy_score(predicted.cpu().numpy(),target_torch.cpu().numpy())

      losses.append(loss.item())
      train_accs.append(acc)

      print('Epoch: %d, loss: %.5f, train_acc: %.2f' %(epoch + 1, loss.item(), acc))

  print('Finished Training')

  input_torch = torch.from_numpy(np.array(test_pad_encoded)).to(device)

  outputs, _ = model(input_torch)
  predicted = torch.argmax(outputs, -1)

  from sklearn.metrics import classification_report
  print(classification_report(label_test_encoded,predicted.cpu().numpy()))

  return losses, train_accs

def update_embedding3(emb_table, n_class, emb_dim, word_list, train_pad_encoded, label_train_encoded,
                        test_pad_encoded, label_test_encoded, alpha):
  # model for Text
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  import torch.nn.functional as F

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  losses = []
  train_accs = []

  class Bi_LSTM_Attention(nn.Module):
      def __init__(self):
          super(Bi_LSTM_Attention, self).__init__()
          self.emb = nn.Embedding(emb_table.shape[0], emb_table.shape[1])
          self.emb.weight.data.copy_(torch.from_numpy(emb_table))
          self.emb.weight.requires_grad = True
          self.lstm = nn.LSTM(emb_table.shape[1], n_hidden, bidirectional=True)
          self.encoder_fc = nn.Linear(2 * n_hidden, n_class)
          # self.activation = nn.ReLU()

      # https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb
      # output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      def attention_net(self, output, final_state):
          hidden = final_state.view(-1, n_hidden * 2,
                                    1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
          attn_weights = torch.bmm(output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
          soft_attn_weights = F.softmax(attn_weights, 1).to(device)
          # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
          context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device)
          # context : [batch_size, n_hidden * num_directions(=2)]
          return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

      def forward(self, X):
          x = self.emb(X)  # input : [batch_size, len_seq, embedding_dim]
          x = x.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]
          hidden_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = Variable(torch.zeros(1 * 2, x.shape[1], n_hidden)).to(
              device)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
          hidden_out = torch.cat((final_hidden_state[0, :, :], final_hidden_state[1, :, :]), 1)
          return self.encoder_fc(hidden_out), self.emb

  from sklearn.metrics import accuracy_score
  n_hidden = 256
  n_emb = emb_dim
  total_epoch = 25
  model = Bi_LSTM_Attention().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  input_torch = torch.from_numpy(np.array(train_pad_encoded)).to(device)
  target_torch = torch.from_numpy(np.array(label_train_encoded)).view(-1).to(device)

  for epoch in range(total_epoch):  
      
      model.train()
        
      optimizer.zero_grad()
      
      outputs, lstm_emb = model(input_torch) 
      
      back_emb = {}
      
      for i, word in enumerate(word_list):
        back_emb[word] = lstm_emb(torch.LongTensor([i])).detach().numpy().flatten()
      _, _, total_acc, loss_1, loss_2 = word_analogy_evaluation(back_emb)
      sim_loss = 3.3*sim_loss_cal2(word_sim_dir,back_emb)
      print("sparsemans loss",sim_loss)
      avg_loss = 10.24*(total_acc+loss_1+loss_2)/3
      avg_loss = avg_loss * alpha
      alpha = alpha * alpha
      print("word analogy weight adjusted",avg_loss)
      loss = criterion(outputs, target_torch)-avg_loss+sim_loss
      loss.backward()
      optimizer.step()
        
      predicted = torch.argmax(outputs, -1)
      acc= accuracy_score(predicted.cpu().numpy(),target_torch.cpu().numpy())

      losses.append(loss.item())
      train_accs.append(acc)

      print('Epoch: %d, loss: %.5f, train_acc: %.2f' %(epoch + 1, loss.item(), acc))

  print('Finished Training')

  input_torch = torch.from_numpy(np.array(test_pad_encoded)).to(device)

  outputs, _ = model(input_torch)
  predicted = torch.argmax(outputs, -1)

  from sklearn.metrics import classification_report
  print(classification_report(label_test_encoded,predicted.cpu().numpy()))

  return losses, train_accs