class Data(Dataset):
    def __init__(self, path, y):
        super().__init__()
        df = pd.read_csv(path, usecols=y, low_memory=False)
        df = df[df.bert_feat.notna()].reset_index(drop=True)
        feat_list = [torch.tensor(eval(x)) for x in df['bert_feat']]
        self.x = torch.cat(feat_list).view(-1, 1024).float()
        df.fillna(-1, inplace=True)
        self.y = df.columns[0:-1]
        self.y_list = [torch.from_numpy(df[i].values) for i in self.y]
        self.y = torch.cat(self.y_list).view(len(self.y_list), -1)
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[:,index]
    
    def num_tasks(self):
        return len(self.y_list)
      
class Net(nn.Module):
    def __init__(self, D_in: int, hidden: int, D_out: int, num_tasks: int, dropout=0.1):
        '''
        D_in (int): input dimension
        hidden (int): number of hidden layers
        D_out (int): output dimension or the number of classes
        num_tasks (int): number of tasks to predict
        dropout (float): percentage of neurons to be masked
        '''
        super().__init__()
        self.num_tasks = num_tasks
        self.bottlenet = nn.Linear(D_in, hidden)
        self.obj1 = nn.Linear(hidden, hidden)
        self.obj2 = nn.Linear(hidden, hidden)
        self.obj3 = nn.Linear(hidden, D_out)
        self.obj4 = nn.Linear(hidden, D_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        yhat = torch.tensor([])
        x = self.dropout(self.bottlenet(x))
        x1 = torch.sigmoid(self.dropout(F.relu(self.obj1c(self.dropout(self.obj1b(self.dropout(self.obj1a(x))))))))
        x2 = torch.sigmoid(self.dropout(F.relu(self.obj2(x)))) 
        x3 = torch.sigmoid(self.dropout(F.relu(self.obj3(x))))
        x4 = torch.sigmoid(self.dropout(F.relu(self.obj4c(self.dropout(self.obj4b(self.dropout(self.obj4a(x))))))))
        yhat = torch.cat((x1, x2, x3, x4), 1)
        return yhat
      
class Net(nn.Module):
    def __init__(self, D_in: int, hidden: int, D_out: int, num_tasks: int, dropout=0.1):
        '''
        D_in (int): input dimension
        hidden (int): number of hidden layers
        D_out (int): output dimension or the number of classes
        num_tasks (int): number of tasks to predict
        dropout (float): percentage of neurons to be masked
        '''
        super().__init__()
        self.num_tasks = num_tasks
        self.bottlenet = nn.Linear(D_in, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, D_out) for i in range(num_tasks)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        yhat = torch.tensor([])
        x = self.dropout(self.bottlenet(x))
        for task in self.layers:
            y = torch.sigmoid(self.dropout(F.relu(task(x))))
            yhat = torch.cat((yhat, y), 1)
        return yhat
    
model = Net(D_in, hidden, D_out, num_tasks, dropout)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def criterion(yhat, y):
    mask = [y != -1]
    y, yhat = y[mask], yhat[mask]
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(yhat, y)
    return loss

from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, start_lr = 0.0001, end_lr=0.03, num_iter=100)
lr_finder.plot(suggest_lr=True) # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state.

def train_accuracy(yhat, y):
    mask = [y != -1]
    y, yhat = y[mask], torch.round(yhat[mask])
    correct = (y == yhat).sum().item()
    count = len(y)
    accuracy = correct / count
    return accuracy
    

def train(model, criterion, train_loader, validation_loader, optimizer, epochs=1): 
    useful_stuff = {'training_loss': [], 'training_accuracy':[], 'validation_accuracy': [], 'F1': []} 
    for epoch in range(epochs):

        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = model(x.view(-1, 1024))
            yhat = yhat.type(torch.FloatTensor) # [n x k]
            y = y.type(torch.FloatTensor) # [n x k]
            loss = criterion(yhat, y)
            training_accuracy = train_accuracy(yhat, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(round(loss.data.item(), 2))
            useful_stuff['training_accuracy'].append(round(training_accuracy, 2))

        model.eval()
        correct, tp, fp, fn = 0, 0, 0, 0
        for i, (x, y) in enumerate(validation_loader):
            with torch.no_grad():
                yhat = model(x.view(-1, 1024)).round()

                correct = torch.sum((yhat == y), dim=0)
                total = torch.sum(y != -1, dim=0)
                accuracy = (correct/total).numpy().ravel()
                useful_stuff['validation_accuracy'].append(accuracy)
           
                tp = ((yhat == y) & (y == 1)).sum().item()
                fp = ((yhat == 1) & (y == 0)).sum().item()
                fn = ((yhat == 0) & (y == 1)).sum().item()
                F1 = round(tp / (tp + 0.5 * (fp + fn) + 1e-9), 2)
                useful_stuff['F1'].append(F1)

        if (epoch % 50 == 0) & (epoch != 0):
            plot_accuracy = pd.DataFrame(useful_stuff['validation_accuracy'])
            pl.clf()
            pl.figure(figsize=(20,10))
            pl.plot(useful_stuff['training_loss'], '-b', lw=1, label='training loss')
            pl.plot(useful_stuff['training_accuracy'], '-g', lw=1, label='training accuracy')
            
            pl.plot(useful_stuff['validation_accuracy'], '.m', lw=1, label='validation accuracy')
            # plot multiple accuracy
            pl.plot(plot_accuracy[0], '-m', lw=1, label='validation accuracy cyp2d6i')
            pl.plot(plot_accuracy[1], '-k', lw=1, label='validation accuracy NR-ER-LBD')
            pl.plot(plot_accuracy[2], '-y', lw=1, label='validation accuracy cyp2c9i')
            pl.plot(plot_accuracy[3], '-c', lw=1, label='validation accuracy ames')
            pl.plot(plot_accuracy[4], ':m', lw=1, label='validation accuracy cyp2c19i')
            pl.plot(useful_stuff['F1'], '-.c', label='validation F1')
            pl.title('metrics')
            pl.legend()
            pl.ylim(0, 1)
            pl.xlabel('epoch')
            display.display(pl.gcf())
            display.clear_output(wait=True)
            time.sleep(1.0)
        else:
            pass
    return useful_stuff
