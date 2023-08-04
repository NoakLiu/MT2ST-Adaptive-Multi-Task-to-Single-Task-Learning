import numpy as np

def classification(emb_table, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded):
  # model for Text
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  import torch.nn.functional as F

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  class Bi_LSTM_Attention(nn.Module):
      def __init__(self):
          super(Bi_LSTM_Attention, self).__init__()
          self.emb = nn.Embedding(emb_table.shape[0],emb_table.shape[1])
          self.emb.weight.data.copy_(torch.from_numpy(emb_table))
          self.emb.weight.requires_grad = False
          self.lstm = nn.LSTM(emb_table.shape[1], n_hidden, bidirectional=True)
          self.encoder_fc = nn.Linear(2*n_hidden, n_class)
          #self.activation = nn.ReLU()

      # https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb
      # output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      def attention_net(self, output, final_state):
          hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
          attn_weights = torch.bmm(output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
          soft_attn_weights = F.softmax(attn_weights, 1).to(device)
          # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
          context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device)
          #return context, soft_attn_weights # context : [batch_size, n_hidden * num_directions(=2)]
          return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

      def forward(self, X):
          x = self.emb(X) # input : [batch_size, len_seq, embedding_dim]
          # 32, 28, 100
          x = x.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]
          hidden_state = Variable(torch.zeros(1*2, x.shape[1], n_hidden)).to(device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = Variable(torch.zeros(1*2,  x.shape[1], n_hidden)).to(device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

          # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          #output, final_hidden_state = self.lstm(x)
          #output, (final_hidden_state, final_cell_state) = self.lstm(x)
          output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
          #output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
          #attn_output, attention = self.attention_net(output, final_hidden_state)
          #features = self.activation(self.encoder_fc(attn_output)) # model : [batch_size, num_classes]
          #x = self.emb(x)
          #lstm_out, (h_n, c_n) = self.lstm(x)
          hidden_out = torch.cat((final_hidden_state[0,:,:],final_hidden_state[1,:,:]),1)
          #z = self.encoder_fc(hidden_out)
          return self.encoder_fc(hidden_out)

  from sklearn.metrics import accuracy_score
  n_hidden = 256
  n_emb = emb_dim
  total_epoch = 25
  model = Bi_LSTM_Attention().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.005)
  input_torch = torch.from_numpy(np.array(train_pad_encoded)).to(device)
  target_torch = torch.from_numpy(np.array(label_train_encoded)).view(-1).to(device)

  for epoch in range(total_epoch):

      model.train()

      optimizer.zero_grad()

      outputs = model(input_torch)
      loss = criterion(outputs, target_torch)
      loss.backward()
      optimizer.step()

      predicted = torch.argmax(outputs, -1)
      acc= accuracy_score(predicted.cpu().numpy(),target_torch.cpu().numpy())

      print('Epoch: %d, loss: %.5f, train_acc: %.2f' %(epoch + 1, loss.item(), acc))

  print('Finished Training')

  model.eval()

  input_torch = torch.from_numpy(np.array(test_pad_encoded)).to(device)

  outputs = model(input_torch)
  predicted = torch.argmax(outputs, -1)

  from sklearn.metrics import classification_report
  print(classification_report(label_test_encoded,predicted.cpu().numpy()))


def classification_lr(emb_table,learning_rate, n_class, emb_dim, train_pad_encoded, label_train_encoded, test_pad_encoded, label_test_encoded):
  # model for Text
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.autograd import Variable
  import torch.nn.functional as F

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  class Bi_LSTM_Attention(nn.Module):
      def __init__(self):
          super(Bi_LSTM_Attention, self).__init__()
          self.emb = nn.Embedding(emb_table.shape[0],emb_table.shape[1])
          self.emb.weight.data.copy_(torch.from_numpy(emb_table))
          self.emb.weight.requires_grad = False
          self.lstm = nn.LSTM(emb_table.shape[1], n_hidden, bidirectional=True)
          self.encoder_fc = nn.Linear(2*n_hidden, n_class)
          #self.activation = nn.ReLU()

      # https://colab.research.google.com/github/ngduyanhece/nlp-tutorial/blob/master/4-3.Bi-LSTM%28Attention%29/Bi_LSTM%28Attention%29_Torch.ipynb
      # output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
      def attention_net(self, output, final_state):
          hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
          attn_weights = torch.bmm(output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
          soft_attn_weights = F.softmax(attn_weights, 1).to(device)
          # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
          context = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2).to(device)
          #return context, soft_attn_weights # context : [batch_size, n_hidden * num_directions(=2)]
          return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]

      def forward(self, X):
          x = self.emb(X) # input : [batch_size, len_seq, embedding_dim]
          # 32, 28, 100
          x = x.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]
          hidden_state = Variable(torch.zeros(1*2, x.shape[1], n_hidden)).to(device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          cell_state = Variable(torch.zeros(1*2,  x.shape[1], n_hidden)).to(device) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

          # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
          #output, final_hidden_state = self.lstm(x)
          #output, (final_hidden_state, final_cell_state) = self.lstm(x)
          output, (final_hidden_state, final_cell_state) = self.lstm(x, (hidden_state, cell_state))
          #output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
          #attn_output, attention = self.attention_net(output, final_hidden_state)
          #features = self.activation(self.encoder_fc(attn_output)) # model : [batch_size, num_classes]
          #x = self.emb(x)
          #lstm_out, (h_n, c_n) = self.lstm(x)
          hidden_out = torch.cat((final_hidden_state[0,:,:],final_hidden_state[1,:,:]),1)
          #z = self.encoder_fc(hidden_out)
          return self.encoder_fc(hidden_out)

  from sklearn.metrics import accuracy_score
  n_hidden = 32#256
  n_emb = emb_dim
  total_epoch = 25
  model = Bi_LSTM_Attention().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  input_torch = torch.from_numpy(np.array(train_pad_encoded)).to(device)
  target_torch = torch.from_numpy(np.array(label_train_encoded)).view(-1).to(device)

  for epoch in range(total_epoch):

      model.train()

      optimizer.zero_grad()

      outputs = model(input_torch)
      loss = criterion(outputs, target_torch)
      loss.backward()
      optimizer.step()

      predicted = torch.argmax(outputs, -1)
      acc= accuracy_score(predicted.cpu().numpy(),target_torch.cpu().numpy())

      print('Epoch: %d, loss: %.5f, train_acc: %.2f' %(epoch + 1, loss.item(), acc))

  print('Finished Training')

  model.eval()

  input_torch = torch.from_numpy(np.array(test_pad_encoded)).to(device)

  outputs = model(input_torch)
  predicted = torch.argmax(outputs, -1)

  from sklearn.metrics import classification_report
  print(classification_report(label_test_encoded,predicted.cpu().numpy()))