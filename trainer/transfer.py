'''
Transfer using finetuning based on source model f from train_s
'''


import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from optparse import OptionParser
import wandb 
from pdb import set_trace as bp
import sys, os 

PATH = "../"  

sys.path.insert(0, os.path.abspath('.'))
from  util.utils import *
from trainer.train_s import train_all 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


DATA_PATH =  "data/"
MODEL_PATH = "model/"
SAVE_PATH =  "output/"
N_TASK = 5


class SubLoader(torchvision.datasets.CIFAR10):
    def __init__(self, *args, category_list=[], **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        # use all classes for test
        if not self.train:
            return
        
        # use only few classes for train
        labels = np.array(self.targets)
        mask =  np.isin(labels, category_list)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()

"""
    category_list is a list of ids. e.g. [0,1] refers to classes ['plane','car']
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""
def load_data_partial(category_list,batch_size, train):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if train:

        trainset = SubLoader(category_list=category_list, 
                            root='./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=2)
    else:
        testset = SubLoader(category_list=category_list,
                            root='./data', train=False, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, 
                                             num_workers=2)
    classes_names = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = [classes_names[i] for i in category_list]
    print('load', 'train' if train else 'test', 
          'dataset, selected classes',classes )
    return loader,  classes


class transfer_from_all(train_all):
    def __init__(self, id=0, batch_size=100):
        self.id = id
        #super(transfer_from_all, self).__init__(path = PATH, all=False, batch_size=batch_size)
        super().__init__(path = PATH, all=False, batch_size=batch_size)
        print(MODEL_PATH)
        self.model_f.load_state_dict(torch.load(MODEL_PATH+'f_task_all.pth', map_location=device))
        print('init',self.id)

    
    # by default, it loads train data only     
    def load_data(self, batch_size):
        self.train_loader,self.train_classes = load_data_partial(category_list=[self.id],
                                                                  batch_size = batch_size, 
                                                                  train=True)
        
    
    def load_test_data(self,batch_size):
        self.test_loader,self.test_classes = load_data_partial( np.arange(self.num_class) ,
                                                               batch_size=batch_size, 
                                                               train=False)
         
    def HGRloss(self,f,g):
        return  (-2)*self.corr(f,g) + 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum()\
              + self.cov_trace(f,g)
    
    def HGR_sc_loss(self,f,g):

        cov_f = torch.mm(torch.t(f),f) / (f.size()[0]-1.) 
        tr_covf= torch.trace( cov_f )
        return (-2)*self.corr(f,g) #+ tr_covf #+ 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum()\
              
    
    # if train_f = True: update both f and g. otherwise, only retrain g
    def finetune(self, train_f=False, num_epochs = 20, lr = 0.0001, print_loss=True):
        
        self.model_g.train()

        if train_f:
            self.model_f.train()
            optimizer_fg = torch.optim.Adam(list(self.model_f.parameters())+list(self.model_g.parameters()), lr=lr)
        else:
            self.model_f.eval()
            optimizer_fg = torch.optim.Adam(list(self.model_g.parameters()), lr=lr)

        total_step = len(self.train_loader)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                
                labels_one_hot = torch.zeros(len(labels), self.num_class).scatter_(1, labels.view(-1,1), 1)

                # Forward pass
                optimizer_fg.zero_grad()
                f = self.model_f(Variable(images).to(device))
                g = self.model_g(Variable(labels_one_hot).to(device))

                loss = self.HGR_sc_loss(f,g)  #(-2)*self.corr(f,g) + 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum() + self.cov_trace(f,g)

                loss.backward()
                optimizer_fg.step()
                wandb.log({"-2corr(f,g)": (-2)*self.corr(f,g),
                          # "2tr(E[f]E[g])": 2*((torch.sum(f,0)/f.size()[0])*(torch.sum(g,0)/g.size()[0])).sum(),
                          #   "cov_trace":  torch.trace( torch.mm(torch.t(f),f) / (f.size()[0]-1.) ),
                             "loss":loss.item()})
                if print_loss and (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        print('Finished Training')
        wandb.finish()
    
    # to do
    def test_model(self, data_loader ,save_features=False):
        # For each model, extract all features and Test the model on all training data, get the raw score

        self.model_f.eval()
        self.model_g.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        result=dict() 
        with torch.no_grad():
            """
            if fit_train: 
                feature_path = 'data/feature_trn' 
                result['trn_acc'] =  self.test_model_batch( data_loader=self.train_loader,
                                                           feature_path = feature_path) 
                print('Train Accuracy of the model on the train images: {} %'.format(100 * result['trn_acc'] ))
            """
            #feature_path = 'data/feature_tst_ft' + self.id  
            result['tst_score'] =  self.test_model_batch( data_loader=data_loader) 
            print('Test Accuracy of the model on the test images: {} %'.format(100 * result['tst_score']))
        
        return result
    
    # to do
    def test_model_batch(self, data_loader ,feature_path=None):
        acc = 0
        total = 0
        """
        features_f = []
        features_g = []
        labellist = [] 
        labellist_p = []
        """
        score = []
        
        for (images, labels) in data_loader:
            
            labels = labels.numpy()
            fc = self.model_f(Variable(images).to(device)).data.cpu().numpy()
            f_mean = np.sum(fc,axis = 0) / fc.shape[0]
            fcp = fc - f_mean  # f centered, dim = (batch_size x 10)
            
            L = torch.eye(self.num_class)
            gc = self.model_g(Variable(L).to(device)).data.cpu().numpy()
            gce = np.sum(gc,axis = 0) / gc.shape[0]
            gcp = gc - gce   # g centered, dim = (10 x 10)
            fgp = np.dot(fcp, gcp.T)
            score.append(fgp)
            #acc += (np.argmax(fgp, axis = 1) == labels).sum()
            #total += len(images)
            """
            if feature_path:
                features_f.append( fcp)
                features_g.append(gcp)
                labellist.append(labels)
                labellist_p.append( np.argmax(fgp, axis = 1))
            """
        """
        if feature_path:
            features_f = np.concatenate(features_f, axis=0)
            features_g = np.concatenate(features_g, axis=0)
            labellist  =   np.concatenate(labellist,axis=0) 
            labellist_p  =   np.concatenate(labellist_p, axis=0) 
            np.savez( feature_path,f=features_f,g=features_g,
                        labels=labellist,labels_p=labellist_p )
        """    
        score = np.concatenate(score, axis=0)
        return score 
    
    def tuning(self):
        pass

    def save_model(self, train_f = False, dir = 'model/', test = True):

        print('----------------Save Model for id = '+str(self.id)+'--------------------')
        """
        self.finetune(train_f = train_f)

        if test:
            self.test_model()
        """
        if train_f:
            print('*finetuning both f and g*')
            save_path_f =  dir + 'f_task_transfer_ft'+str(self.id)+'.pth'
            torch.save(self.model_f.state_dict(), save_path_f) #self.path +
        if train_f:
            save_path_g =  dir + 'g_task_transfer_ft'+str(self.id)+'.pth'
        else:
            save_path_g =  dir + 'g_task_transfer'+str(self.id)+'.pth'
        torch.save(self.model_g.state_dict(), save_path_g) #self.path +

    def load_model(self, dir='model/'):
        
        save_path_f =  dir + 'f_task_transfer_ft'+str(self.id)+ '.pth' 
        save_path_g = dir + 'g_task_transfer_ft'+str(self.id)+ '.pth' 
        self.model_f.load_state_dict(torch.load(save_path_f))
        self.model_g.load_state_dict(torch.load(save_path_g))
        print('successfully loaded saved model', save_path_f ,  save_path_g )

        

def get_args():
    parser = OptionParser()
    parser.add_option('--test_only', dest='test_only', 
                        default='true', type='str', 
                        help='only test the model using saved parameters, without training the model')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    args = get_args()
    batch_size = 512
    num_epochs =  15
    lr = 0.0001
    num_class = 10

    """
    dim=(ntest x nclasses) 
    result = []
    
    for each class
        if train:
            finetune this class
            save model
        else: 
            load this class
        evaluate this class on test data
        store test result (score) in result
    
    aggregate result by argmax
    evaluate result against ground truth
    """

    test_score = [] # to be updated
    # load all test data 
    test_loader ,classes= load_data_partial(np.arange(num_class),batch_size=10000, train=False)
    acc=0
    for id in range(num_class):
        print("initialize classifier for class",id,"...............")

        # initialize class model and data loader
        train_id = transfer_from_all(id=id, batch_size=batch_size )

        if not str2bool(args.test_only): 
            # fine tune the model on a single class
            wandb.init(
                # set the wandb project where this run will be logged
                project="self-transfer-id-trainf_ft",
                
                # track hyperparameters and run metadata
                config={"model_name":"source model",
                "learning_rate": lr,
                "architecture": "FGnet",
                "dataset": "CIFAR-10",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "class_id":id
                }
            )
                    
            # finetune for class i and save model
            train_id.finetune(train_f= True) 
            train_id.save_model()
        else:
            # load the finetuned model for class i 
            train_id.load_model()
        
        # use the finetuned model for class i on test data
        # results are stored
        result = train_id.test_model(test_loader,save_features=False) #todo
        test_score.append(result['tst_score'])
    # aggregate the scores of all columns
    test_score = np.concatenate(test_score, axis=1)

    # take argmax of test score, then evaluate accuracy
    test_labels =[]
    total = 0
    for _, labels in test_loader:
        test_labels.append(labels)  
        total += len(labels)
    test_labels  =   np.concatenate(test_labels,axis=0) 
    acc += (np.argmax(test_score, axis = 1) == test_labels).sum()
    score = float(acc) / total
    print("testing accuracy for aggregated classifier:",score)