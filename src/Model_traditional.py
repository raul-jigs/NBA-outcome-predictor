import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import torch
import numpy as np
import matplotlib.pyplot as plt
#defining dataset class
from torch.utils.data import Dataset, DataLoader

#defining the network
from torch import nn
from torch.nn import functional as F

# dropoutParam = 0
# learning_rate = 0.01
# layersParam= [24, 16, 4]

def save_model(model, path):
  torch.save(model, path)

def load_model(path):
  model = torch.load(path)
  return model

def deep_learning(dropoutParam, learning_rate, layersParam, type_of_data, path=None):

  class Net(nn.Module):
    def __init__(self,input_shape, layers=[24, 16, 4]):
      super(Net,self).__init__()
      self.num_layers = len(layers)
      self.linears = nn.ModuleList()
      self.linears.append(nn.Linear(input_shape, layers[0]))
      for i in range(1, len(layers)):
          self.linears.append(nn.Linear(layers[i-1], layers[i]))
          self.linears.append(nn.Dropout1d(dropoutParam))
          self.linears.append(nn.ReLU())
          
      self.linears.append(nn.Linear(layers[-1], 1))
      
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

  # dataset files found in Clean_Data.zip
  if type_of_data.lower() == 'traditional':
    df = pd.read_csv('data/CleanData/NBA_Traditional_Stats_Dataset_new.csv')
  elif type_of_data.lower:
    df = pd.read_csv('../data/NBA_Advanced_Stats_Dataset_new.csv')

  def clean_data():
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print("Missing Values:")
    print(missing_data)


  X = df.drop(columns=['W/L', 'Match Up', 'Game Date'])

  y = df['W/L']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 430, shuffle = True)
  scalar = StandardScaler()
  X_train = scalar.fit_transform(X_train)
  X_test = scalar.transform(X_test)

  trainset = dataset(X_train,y_train.values)
  # print(X_train.shape)

  # uncomment below to predict on test values
  # testset = dataset(X_test, y_test.values)

  
  # DataLoader
  trainloader = DataLoader(trainset,batch_size=256,shuffle=True)

  # uncomment below to predict on test values
  # testloader = DataLoader(testset, batch_size=64, shuffle=False)

  # x_train = torch.tensor(X_train,dtype=torch.float32, requires_grad=False)
  # y_train = torch.tensor(y_train.values,dtype=torch.float32, requires_grad=False)

  x_test = torch.tensor(X_test,dtype=torch.float32, requires_grad=False)
  y_test = torch.tensor(y_test.values,dtype=torch.float32, requires_grad=False)


  #hyper parameters
  epochs = 600
  # Model , Optimizer, Loss

  model = Net(input_shape=X_train.shape[1],layers = layersParam)
  # Adam optimizer
  optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
  # SGD optimizer
  # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.8)

  # Binary cross entropy loss
  loss_fn = nn.BCELoss()

  # Mean Squared Error loss
  # loss_fn = nn.MSELoss()

  # Negative Log Likelihood Loss
  # loss_fn = nn.NLLLoss()

  #forward loop
  losses_all = []
  accur_all = []

  for i in range(1,epochs+1):
      model.train()
      accur = []
      losses = []
      for j,(x_tr,y_tr) in enumerate(trainloader):
          #calculate output
          output = model(x_tr)

          #calculate loss
          
          loss = loss_fn(output,y_tr.reshape(-1,1))
          
          losses.append(loss.item())
          #accuracy
          acc = (output.reshape(-1).detach().numpy().round() == y_tr.numpy()).mean()
          accur.append(acc)
          #backprop
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          # validation
      losses_all.append(sum(losses)/len(losses))
      accur_all.append(sum(accur)/len(accur))
      
      

      if i % 200 == 0:
          model.eval()
          with torch.no_grad():
              output = model(x_test)
              acc = (output.reshape(-1).detach().numpy().round() == y_test.numpy()).mean()
          print("epoch {} average train accuracy: {} average loss: {} test accuracy : {}".format(i, accur_all[-1], losses_all[-1],acc))

  if path != None:
    save_model(model, path)
  return model


def plot_loss(losses_all, accur_all):
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

# hyper parameter tuning
def hyper_parameter_tuning(learning_rate, dropoutParams, type_of_data):
  for i in range(len(dropoutParams)): # dropout values
        for j in range(16, 49, 8): # number of neurons in first layer 
            for k in range(10, 31, 4): # number of neurons in second layer
                for l in range(2, 19, 4): # number of neurons in third layer
                    print(f"Learning rate: {learning_rate}   Network: [{j}, {k}, {l}]  Dropout: {dropoutParams[i]}")
                    deep_learning(dropoutParam=dropoutParams[i], learning_rate=learning_rate, layersParam=[j, k, l], type_of_data)

def best_traditional_train_model(path=None):
  deep_learning(dropoutParam=0.2, learning_rate=0.01, layersParam=[16, 10, 6], type_of_data='traditional', path=path)

def best_advanced_train_model(path=None):
  deep_learning(dropoutParam=0.2, learning_rate=0.01, layersParam=[16, 10, 6], type_of_data='advanced', path=path)

if __name__ == '__main__':
  learning_rate = 0.01
  dropoutParams = [0, 0.2]
  # you can change the paths to save the models wherever you would like to save them
  traditional_path = '../src/Traditional-NBA-outcome-predictor-model.pt'
  advanced_path = '../src/Advanced-NBA-outcome-predictor-model.pt'
  traditional_model = best_traditional_train_model(traditional_path)
  advanced_model = best_advanced_train_model(advanced_path)


