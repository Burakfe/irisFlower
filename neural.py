import torch
import torch.nn as nn #neural network 
import torch.nn.functional as F #move data forward in the network
import pandas as pd


#Create a model class that inherits from nn.Module
class Model(nn.Module):
    #Input Layer (4 Features of the flower) --> 
    # Hidden Layer1 (number of neurons) -->
    # H2(n) --> 
    # Output Layer (3 classes of iris flowers) 
    
    def __init__(self, in_features = 4 , h1 = 10, h2 = 10, out_features = 3): #fully connect neural networks
        super().__init__()
        # Define the layers o the model
        #Linear function fully connects neurons to each other with the random weights it acquire
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x
    # Set the random seed for reproducibility
    torch.manual_seed(36)
# Create an instance of the model
model = Model()

#Dataset çektim
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)
#Change the dataset for usage
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

#Train Test Split; X and Y, X is the features and Y is the outcome

x = my_df.drop('variety' , axis = 1)
y = my_df['variety']

#Convert these to numpy arrays
x = x.values
y = y.values

from sklearn.model_selection import train_test_split
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 36) # it will 20% test and 80 % train (% of the total 'dataset'), test_size = 0.2
#Random_state ensures reproducibility; good for debugging, comparing models
#Convert X features to float tensors (Tensors are used to store data (like images, features, or weights))
x_train = torch.FloatTensor(x_train) #every number in the dataset is float so use FloatTensor
x_test = torch.FloatTensor(x_test) 

#Convert y features to long tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Create the criterion of model to measure error: how far are predictions from the data
criterion = nn.CrossEntropyLoss()

#Choose optimizer (I use Adam optimizer) lr= learning rate (if error doesn't go down after a bunch of iterations) , lower the better
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) #Model parameters are fc1, fc2 and fc_out
#If I run this multiple times and the error does not go down, change the learning rate (add another zero etc.)

#Train the MODEL (one run thru all the training data in our network)
epochs = 100 #number of one full cycle in the training dataset, epoch increases: more precise (If I have 120 training data all training data going thru counts as 1 epoch)
losses = []
for i in range(epochs):
    #Go forward and make a prediction
    y_pred = model.forward(x_train) #try to get predicted results
    
    #Measure the loss/error; can be high at first
    loss = criterion(y_pred, y_train) 
    #y_predict vs y_train values
    
    #Keep track of losses to see things work correctly
    losses.append(loss)
    
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')
    #Do the back propagation: to learn better send back the error rate through the network, by this the network (Linear func) will determine weights accordingly
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#evaluate the model on test data set
with torch.no_grad(): #turn off back propogation
    y_eval = model.forward(x_test) #x_test are features, y_eval will be the predictions
    loss =criterion(y_eval, y_test)
    #this and the epoch must be close 

correct = 0
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model.forward(data) 
        print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'Total correct answer # is {correct}')