'''
training of source model
'''

import torch.nn as nn
import torch.nn.functional as F 
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from optparse import OptionParser
import wandb 
from pdb import set_trace as bp
import sys, os 

PATH = "../"  

sys.path.insert(0, os.path.abspath('.'))
from   util.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




class Net_f(nn.Module):
    def __init__(self):
        super(Net_f, self).__init__()
        googlenet = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
        self.feature=torch.nn.Sequential(*list(googlenet.children())[0:18])
        self.fc1 = nn.Linear(1024,32)
        self.fc2 = nn.Linear(32,10)
        self.BN = nn.BatchNorm1d(10)

    def forward(self,x):
        out=self.feature(x)
        out=out.view(-1,1024)
        out=F.relu(self.fc1(out))
        out=self.fc2(out)
        out=self.BN(out)

        return out     

class Net_g(nn.Module):
    def __init__(self,num_class=10, dim=10):
        super(Net_g, self).__init__()

        self.fc=nn.Linear(num_class, dim)

    def forward(self,x):
        out=self.fc(x)

        return out


def load_data_all(batch_size=100):
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes



class train_all():

    def __init__(self, path = PATH, all=True, batch_size=100):

        self.num_class = 10  #if all else 1 (we do not change the dimension of y in )
        #self.train_loader, self.test_loader , self.classes =
        self.load_data(batch_size)
        self.model_f = Net_f().to(device)
        self.model_g = Net_g(num_class = self.num_class).to(device)
        self.path = path

    def load_data(self,batch_size):
        self.train_loader, self.test_loader , self.classes = load_data_all(batch_size =batch_size)

    
    def corr(self,f,g):
        k = torch.mean(torch.sum(f*g,1))
        return k
        
    def cov_trace(self,f,g):
        cov_f = torch.mm(torch.t(f),f) / (f.size()[0]-1.)
        cov_g = torch.mm(torch.t(g),g) / (g.size()[0]-1.)
        return torch.trace(torch.mm(cov_f, cov_g))

    def train(self, num_epochs = 20, lr = 0.0001, print_loss=True) -> None:

        self.model_f.train()
        self.model_g.train()

        optimizer_fg = torch.optim.Adam(list(self.model_f.parameters())+list(self.model_g.parameters()), lr=lr)

        total_step = len(self.train_loader)

        for epoch in range(num_epochs):

            for i, (images, labels) in enumerate(self.train_loader):
                
                labels_one_hot = torch.zeros(len(labels), self.num_class).scatter_(1, labels.view(-1,1), 1)

                # Forward pass
                optimizer_fg.zero_grad()
                f = self.model_f(Variable(images).to(device))
                g = self.model_g(Variable(labels_one_hot).to(device))

                loss = (-2)*self.corr(f,g) + 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum() + self.cov_trace(f,g)

                loss.backward()

                optimizer_fg.step()
                
                wandb.log({"loss":loss.item()})
                if print_loss and (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        print('Finished Training')
        wandb.finish()


    def test_model(self, fit_train=False,save_features=False):
        # Test the model
        self.model_f.eval()
        self.model_g.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        result=dict()
        with torch.no_grad():
            if fit_train: 
                feature_path = 'data/feature_trn' 
                result['trn_acc'] =  self.test_model_batch( data_loader=self.train_loader,
                                                           feature_path = feature_path) 
                print('Train Accuracy of the model on the train images: {} %'.format(100 * result['trn_acc'] ))
            
            feature_path = 'data/feature_tst' 
            result['tst_acc'] =  self.test_model_batch( data_loader=self.test_loader,
                                                       feature_path=feature_path) 
            print('Test Accuracy of the model on the test images: {} %'.format(100 * result['tst_acc']))
        
        return result
    
    def test_model_batch(self, data_loader ,feature_path=None):
            acc = 0
            total = 0
            features_f = []
            features_g = []
            labellist = [] 
            labellist_p = []
            for images, labels in data_loader:
                
                labels = labels.numpy()
                fc = self.model_f(Variable(images).to(device)).data.cpu().numpy()
                f_mean = np.sum(fc,axis = 0) / fc.shape[0]
                fcp = fc - f_mean  # f centered, dim = (batch_size x 10)
                
                L = torch.eye(self.num_class)
                gc = self.model_g(Variable(L).to(device)).data.cpu().numpy()
                gce = np.sum(gc,axis = 0) / gc.shape[0]
                gcp = gc - gce   # g centered, dim = (10 x 10)
                fgp = np.dot(fcp, gcp.T)
                acc += (np.argmax(fgp, axis = 1) == labels).sum()
                total += len(images)
                if feature_path:
                    features_f.append( fcp)
                    features_g.append(gcp)
                    labellist.append(labels)
                    labellist_p.append( np.argmax(fgp, axis = 1))
            if feature_path:
                features_f = np.concatenate(features_f, axis=0)
                features_g = np.concatenate(features_g, axis=0)
                labellist  =   np.concatenate(labellist,axis=0) 
                labellist_p  =   np.concatenate(labellist_p, axis=0) 
                np.savez( feature_path,f=features_f,g=features_g,
                         labels=labellist,labels_p=labellist_p )
                
            score = float(acc) / total

            return score 

    def tuning(self):
        pass
        
    def save_model(self, dir = 'model/', test = True): 
        save_path_f =  dir + 'f_task_all.pth' #self.path +
        torch.save(self.model_f.state_dict(), save_path_f)
        save_path_g = dir + 'g_task_all.pth' #self.path + 
        torch.save(self.model_g.state_dict(), save_path_g)
        print('saved model to ',save_path_f ,  save_path_g)
        
    def load_model(self, dir='model/' ): 
        save_path_f =  dir + 'f_task_all.pth' 
        save_path_g = dir + 'g_task_all.pth'  
        self.model_f.load_state_dict(torch.load(save_path_f))
        self.model_g.load_state_dict(torch.load(save_path_g))
        print('successfully loaded saved model', save_path_f ,  save_path_g )

def get_args():
    parser = OptionParser()
    parser.add_option('--test_only', dest='test_only', 
                        default='true', type='str', help='only test the model using saved parameters, without training the model')
    (options, args) = parser.parse_args()
    return options

 
if __name__ == '__main__':
    args = get_args() 
    batch_size = 512
    num_epochs = 16
    lr = 0.0001
    num_class = 10

    train_s = train_all(path = PATH, batch_size=batch_size)
    if not str2bool(args.test_only):
        print('start initialization')
        
        wandb.init(
            # set the wandb project where this run will be logged
            project="self-transfer",
            
            # track hyperparameters and run metadata
            config={"model_name":"source model",
            "learning_rate": lr,
            "architecture": "FGnet",
            "dataset": "CIFAR-10",
            "epochs": num_epochs,
            "batch_size": batch_size
            }
        )
        
        print('finish initialization')
        train_s.train(num_epochs = num_epochs, lr=lr)  
        train_s.save_model()
    else:
        train_s.load_model() 
    train_s.test_model(fit_train=True,save_features=True)

