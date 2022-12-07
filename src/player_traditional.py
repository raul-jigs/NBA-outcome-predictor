# import library
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
# defining dataset class
from torch.utils.data import Dataset, DataLoader
from pytorchtools import EarlyStopping

# defining the network
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
  def __init__(self,input_shape, layers=[24, 16, 4]):
    super(Net,self).__init__()
    self.num_layers = len(layers)
    self.linears = nn.ModuleList()
    self.linears.append(nn.Linear(input_shape, layers[0]))
    for i in range(1, len(layers)):
        self.linears.append(nn.Linear(layers[i-1], layers[i]))
        # self.linears.append(nn.Dropout1d(0.2))
        self.linears.append(nn.ReLU())
        
    self.linears.append(nn.Linear(layers[-1], 1))
    # self.fc1 = nn.Linear(input_shape,24)
    # self.fc2 = nn.Linear(24,16)
    # self.fc3 = nn.Linear(16,4)
    # self.fc4 = nn.Linear(4, 1)
  def forward(self,x):
    for i, l in enumerate(self.linears):
        if i < self.num_layers-1:
            x = l(x)
            # x = torch.relu(l(x))
        else:
            x = torch.sigmoid(l(x))

    return x

class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length



pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# read the traditional data
df = pd.read_csv('../data/NBA_Traditional_Stats_Dataset.csv')

# check if there's missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("Missing Values:")
print(missing_data.head())

# drop the data that are not our inputs
X = df.drop(columns=['W/L', 'Match Up', 'Game Date', 'V P8 +/-'])

y = df['W/L']

# Hyperparameter Tuning Code 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, shuffle = True)
# scalar = StandardScaler()
# X_train = scalar.fit_transform(X_train)
# X_val =scalar.transform(X_val) 
# X_test = scalar.transform(X_test)

# grid_search=[]
# for i in [4, 8, 16]:
#     for j in range(2, 33, 2):
#         for k in range(2, 33, 2):
#             classifier_NN = MLPClassifier((i, j, k), max_iter = 5000, solver = 'sgd', alpha = 0.1, activation = 'relu', learning_rate='adaptive', random_state=42)
#             classifier_NN.fit(X_train, y_train)
#             y_pred_NN = classifier_NN.predict(X_val)  
#             grid_search.append([i, j, k, accuracy_score(y_val, y_pred_NN)])
#             print(i, j, k, accuracy_score(y_val, y_pred_NN))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

trainset = dataset(X_train,y_train.values)
# testset = dataset(X_test, y_test.values)

# print(X_train.shape)
#DataLoader
trainloader = DataLoader(trainset,batch_size=256,shuffle=True)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)

# x_train = torch.tensor(X_train,dtype=torch.float32, requires_grad=False)
# y_train = torch.tensor(y_train.values,dtype=torch.float32, requires_grad=False)

x_test = torch.tensor(X_test,dtype=torch.float32, requires_grad=False)
y_test = torch.tensor(y_test.values,dtype=torch.float32, requires_grad=False)


#hyper parameters
learning_rate = 0.001
epochs = 1000
# Model , Optimizer, Loss

model = Net(input_shape=X_train.shape[1],layers = [16])
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.8)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = nn.BCELoss()


def train_model(model, patience, n_epoch):
  
  early_stopping = EarlyStopping(patience=patience, verbose=True)
  valid_losses = []
  avg_valid_losses = [] 
  losses_all = []
  accur_all = []

  for i in range(n_epoch):
      model.train()
      accur = []
      losses = []
      for j,(x_tr,y_tr) in enumerate(trainloader):
          #calculate output
          output = model(x_tr)
          # output = model(x_train)

          #calculate loss

          loss = loss_fn(output,y_tr.reshape(-1,1))
          # loss = loss_fn(output,y_train.reshape(-1,1))

          losses.append(loss.item())
          #accuracy
          # predicted = model(torch.tensor(X_train,dtype=torch.float32))
          # acc = (output.reshape(-1).detach().numpy().round() == y_train.numpy()).mean()
          acc = (output.reshape(-1).detach().numpy().round() == y_tr.numpy()).mean()
          accur.append(acc)
          #backprop
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          # validation
      losses_all.append(sum(losses)/len(losses))
      accur_all.append(sum(accur)/len(accur))

      model.eval()
      with torch.no_grad():
        output = model(x_test)
        loss = loss_fn(output,y_test.reshape(-1,1))
        acc = (output.reshape(-1).detach().numpy().round() == y_test.numpy()).mean()
        
      if i%50 == 0:
        print("epoch {} average train accuracy: {} average loss: {} test accuracy : {}".format(i, accur_all[-1], losses_all[-1],acc))
        
      early_stopping(loss, model)
      if early_stopping.early_stop:
        print("epoch {} average train accuracy: {} average loss: {} test accuracy : {}".format(i, accur_all[-1], losses_all[-1],acc))
        print("Early stopping")
        break



  #plotting the loss
  plt.plot(losses_all)
  plt.title('Loss vs Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('loss')
  plt.savefig("./loss.png")
  plt.show()

  plt.plot(accur_all)
  plt.title('Accuracy vs Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('accuracy')
  plt.savefig("./accuracy.png")
  plt.show()
