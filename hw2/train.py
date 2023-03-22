from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import classification_report

class EarlyStopping:
    def __init__(self,patience=10, verbose=True, delta=0, path='checkpoint/checkpoint.pt'):
        self.patience=patience
        self.verbose=verbose
        self.counter=0
        self.best_score = None
        self.early_stop=False
        self.val_loss_min = np.Inf
        self.delta=delta
        self.path=path
    
    def save_checkpoint(self,val_loss,model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
    def __call__(self, val_loss, model):
        score =- val_loss
        
        if self.best_score is None:
            self.best_score=score
            self.save_checkpoint(val_loss,model)
        
        elif self.best_score+self.delta>score:
            self.counter+=1
            print('Earlystopping 동작: {0} out of {1}'.format(self.counter,self.patience))
            if self.counter>=self.patience:
                self.early_stop=True
        else:
            self.best_score=score
            self.save_checkpoint(val_loss,model)
            self.counter=0
            
#학습&test
def running_loop(dataloader,device,optimizer, model,loss_fn, is_train=True):
    
    correct=0
    tot_loss=0
    process=tqdm(dataloader)
    num_batches=len(dataloader)
    num_data=len(dataloader.dataset) 
    if is_train:
        for X,y in process:
            X=X.to(device)
            y=y.to(device)
            output=model(X)
            loss=loss_fn(output, y)
            tot_loss=+loss.item()
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #acc 계산
            correct += (output.argmax(1) == y).type(torch.float).sum().item()
        
        tot_loss/= num_batches
        acc = correct/num_data
        return tot_loss, acc
    
    else:
        model.eval()
        with torch.no_grad():
            for X,y in process:
                X=X.to(device)
                y=y.to(device)
                output = model(X) 
                tot_loss=+loss_fn(output, y).item()
                #acc 계산
                correct += (output.argmax(1) == y).type(torch.float).sum().item()
            
        tot_loss/=num_batches
        acc=correct/num_data
        
        return tot_loss, acc
    

def test_model(dataloader, model, loss_fn, device):
    categories=[str(i) for i in range(10)]
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        cnt=0
        preds = torch.tensor([]) #예측
        targets = torch.tensor([]) #실제 정답
        
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            #클래스 별 precision-recall 계산
            predicted=pred.argmax(1) #텐서형태임
            
            print('{}번째 step\n'.format(cnt+1))
            preds=torch.cat((preds,predicted.cpu()),0)
            targets=torch.cat((targets,y.cpu()),0)
            cnt+=1
       
        report_total=classification_report(targets.cpu(),preds.cpu(), target_names=categories, zero_division=0)

    total_loss /= num_batches
    correct /= size
    
    print("-------------total classification report-------------------\n")
    print('total classes: ',categories)
    print('클래스 갯수: ', len(categories))
    
    print(report_total)
    return total_loss, correct
    

    
