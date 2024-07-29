import tsaug 
import numpy as np
import torch
from model import sshda
import torch.nn as nn
import torch.optim as optim
from numpy import load
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn.init as init
from utils import data_loading
from utils import EarlyStopping
from sklearn.metrics import f1_score
from utils import dropout,identité, all_elements_same, exponential_scaling, cosine_scaling
import matplotlib.pyplot as plt
import time
from utils import my_transformation
from collections import Counter

s=0
L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)
L2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2019_modif.npz',allow_pickle=True)
L2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2020_modif.npz',allow_pickle=True)
R2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2018_modif.npz',allow_pickle=True)
R2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2019_modif.npz',allow_pickle=True)
R2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2020_modif.npz',allow_pickle=True)
T2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2018_modif.npz',allow_pickle=True)
T2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2019_modif.npz',allow_pickle=True)
T2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2020_modif.npz',allow_pickle=True)

n_epochs= 100
warmup = 70

data = [[L2018],[T2019]] # [[L2018,L2019,L2020,R2018,R2019,R2020],[T2019]] # 

train_dataloader_s,dates,data_shape = data_loading([T2018],nbr_dom=2,role="source",fraction=0.3,nbr_ssl=None)
train_dataloader_t, train_dataloader_ssl_t,  test_dataloader,dates,data_shape = data_loading([T2019],nbr_dom=2,role="target",fraction=0.01,nbr_ssl=1000)
data_shape=(data_shape[0],data_shape[2],data_shape[1])

config={'emb_size':64,'num_heads':8,'Data_shape':data_shape,'Fix_pos_encode':'tAPE','Rel_pos_encode':'eRPE','dropout':0.2,'dim_ff':64}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nb_dom = 2
model = sshda(config,11,nb_dom).to(device)

learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
#transformation = tsaug.AddNoise(scale=0.01)
#transformation = tsaug.Quantize(n_levels=20)
#transformation = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3)
#transformation = dropout(p=0.8)
#transformation = identité
tau = 0.8
valid_f1 = 0.
liste_transformation = [tsaug.TimeWarp(n_speed_change=6, max_speed_ratio=5) ,tsaug.AddNoise(scale=0.05),tsaug.Quantize(n_levels=35) , dropout(p=0.5)]
transformation = my_transformation(0.6,liste_transformation,device)
for n in range(n_epochs):
    seuil_classe = {}
    r=0
    print(f'éqpoue {n+1}')
    tot_pred = []
    tot_labels = []
    start = time.time()
    pred_count = []
    count_thr = []
    count_lab_thr  = []
    alp = (n-warmup)/(n_epochs-warmup)*1/2+1/2
    tau = 0.7
    
    if n < warmup :
        pred_target = []
        label_target = []
        for a,b in zip(train_dataloader_s,train_dataloader_ssl_t):#xm_batch_s, y_batch_s, dom_batch_s,xm_batch_t, y_batch_t, dom_batch_t
           
          
            xm_batch_s,y_batch_s,dom_batch_s=a[0],a[1],a[2]
            xm_batch_t, y_batch_t, dom_batch_t = b[0],b[1],b[2]
            x_batch_s,m_batch_s = xm_batch_s[:,:,:2],xm_batch_s[:,:,2] # m_batch correspond aux mask du batch
            x_batch_t,m_batch_t = xm_batch_t[:,:,:2],xm_batch_t[:,:,2]
            
            
            
            x_batch_s = x_batch_s.to(device)
            m_batch_s = m_batch_s.to(device)
            y_batch_s = y_batch_s.to(device)
            dom_batch_s = dom_batch_s.to(device)
            x_batch_t = x_batch_t.to(device)
            m_batch_t = m_batch_t.to(device)
            y_batch_t = y_batch_t.to(device)
            dom_batch_t = dom_batch_t.to(device)
            

            optimizer.zero_grad()
            pred_lab_s,pred_dom_s,emb_lab_s, emb_dom_s, pred_lab_t,pred_dom_t,emb_lab_t, emb_dom_t = model(x_batch_s, m_batch_s,x_batch_t, m_batch_t)
            
            emb_lab_s = nn.functional.normalize(emb_lab_s)
            emb_dom_s = nn.functional.normalize(emb_dom_s)
            emb_lab_t = nn.functional.normalize(emb_lab_t)
            emb_dom_t = nn.functional.normalize(emb_dom_t)
            
            loss_ortho_s = torch.mean(torch.sum(emb_lab_s*emb_dom_s,dim=1))
            loss_ortho_t = torch.mean(torch.sum(emb_lab_t*emb_dom_t,dim=1))
            
            loss_ortho_st = torch.mean(torch.sum(emb_lab_t*emb_lab_s,dim=1))
           
                     
           
           
            loss_lab_s = loss_fn(pred_lab_s, y_batch_s)
            loss_lab_t = loss_fn(pred_lab_t,y_batch_t)
            loss_dom_s = loss_fn(pred_dom_s,dom_batch_s)
            loss_dom_t = loss_fn(pred_dom_t,dom_batch_t)
            
            loss =  loss_lab_s + loss_lab_t + torch.abs(loss_ortho_st) + 1/2*(loss_dom_s+loss_dom_t) + 1/2*(loss_ortho_s+loss_ortho_t)
            
            loss.backward()
            optimizer.step()
            
            
    else :
        
        tot_pred_pl = []
        tot_labels_pl = []
        a=0 
        
        for a,b in zip(train_dataloader_s,train_dataloader_t):
            xm_batch_s,y_batch_s,dom_batch_s=a[0],a[1],a[2]
            xm_batch_t, y_batch_t, dom_batch_t = b[0],b[1],b[2]
            x_batch_s,m_batch_s = xm_batch_s[:,:,:2],xm_batch_s[:,:,2] # m_batch correspond aux mask du batch
            x_batch_t,m_batch_t = xm_batch_t[:,:,:2],xm_batch_t[:,:,2]
            
            
            
            x_batch_s = x_batch_s.to(device)
            m_batch_s = m_batch_s.to(device)
            y_batch_s = y_batch_s.to(device)
            dom_batch_s = dom_batch_s.to(device)
            x_batch_t = x_batch_t.to(device)
            m_batch_t = m_batch_t.to(device)
            y_batch_t = y_batch_t.to(device)
            dom_batch_t = dom_batch_t.to(device)
            
            model.eval()
            pred_lab_s,pred_dom_s,emb_lab_s, emb_dom_s, pred_lab_t,pred_dom_t,emb_lab_t, emb_dom_t = model(x_batch_s, m_batch_s,x_batch_t, m_batch_t)
            pred_arg = torch.tensor([ torch.argmax(pred_lab_t[k]) for k in range(len(pred_lab_t))])
            unique_labels = torch.unique(pred_arg)
            i_ul = torch.where(y_batch_t == -1)
            i_l = torch.where(y_batch_t != -1) 
            max_prob, y_ul = torch.max(F.softmax(pred_lab_t,dim=1), dim=1)
            y_ul[max_prob<tau] = -1
            
            xl_batch,ml_batch, doml_batch = x_batch_t[i_l],m_batch_t[i_l],dom_batch_t[i_l]
            yl_batch = y_batch_t[i_l].clone().detach()

            label_seuil = y_ul[torch.where(y_ul != -1)]
            
            lb_seuil ,counts = torch.unique(label_seuil, return_counts = True)
            lb_seuil,counts = lb_seuil, counts
            
            if len(counts) != 0 :
                max_count = max(counts)
            else :
                max_count = 0
            dict_seuil = {lb_seuil[i].item(): counts[i].item() for i in range(len(counts))}
            
            for lab in unique_labels: 
                lab = lab.item()
                if lab in lb_seuil : 
                    seuil_classe[lab] = (1+dict_seuil[lab]/max_count)/(3-dict_seuil[lab]/max_count)*tau
                else :
                    seuil_classe[lab] = tau*1/2
            
            max_prob_b, yul_batch = torch.max(F.softmax(pred_lab_t[i_ul],dim=1), dim=1)
            yul_batch[max_prob_b < seuil_classe] = -1
            [torch.argmax(pred_lab_t[k]) if max(F.softmax(pred_lab_t[k]))>seuil_classe[torch.argmax(pred_lab_t[k]).item()] else torch.tensor(-1) for k in i_ul]
            
    
    
    
    
    
    
    
            #yul_batch = [ torch.argmax(pred_lab[k]) if max(F.softmax(pred_lab[k]))>0.7 else torch.tensor(-1) for k in i_ul]  # pseudo label pour les données non labelisée 
            
                                                                            
                                                                                                                # non labélisées
            ind_loss_pl = torch.where(yul_batch != -1)               # indices des pseudo-labels conservés
            count_thr.append(ind_loss_pl)
            #hehe = [torch.argmax(pred_lab[k]) for k in range (len(pred_lab_t))]
           # pred_count.append(hehe)
            #hehe =torch.tensor(hehe).to('cpu')
            #hehe = hehe.numpy()
            
            #sese = torch.tensor(yul_batch)
            #count_lab_thr.append(sese.cpu().detach().numpy())
            #hehe = torch.tensor(hehe)    
            yul_batch = torch.tensor(yul_batch).to(device)
            yul_batch = yul_batch.to(torch.int64)
            #tot_pred_pl.append(hehe[i_ul].cpu().detach().numpy())                    # on ajoute dans cette liste les pseudo-labels de confiance
            #tot_labels_pl.append(yul_batch_info.cpu().detach().numpy())              # on ajoute dans la liste les vrais labels correspondant aux pseudo-labels              
            model.train()
            optimizer.zero_grad()
            xul_batch, mul_batch, domul_batch = x_batch_t[i_ul], m_batch_t[i_ul], dom_batch_t[i_ul]
            xul_batch, mul_batch,domul_batch = xul_batch.to(device), mul_batch.to(device), domul_batch.to(device)
            xul_batch,mul_batch = transformation.augment(xul_batch,mul_batch)    # on applique la transformation de données sur les données non labélisées
            #xul_batch=np.array(xul_batch.cpu())
            #xul_batch = transformation.augment(xul_batch)
            
            
            xul_batch = torch.tensor(xul_batch).to(device)
            
            x_batch_t =   torch.cat((xl_batch,xul_batch),axis=0)
            y_batch_t = torch.cat((yl_batch,yul_batch),axis=0)
            m_batch_t = torch.cat((ml_batch,mul_batch),axis=0) 
            dom_batch_t = torch.cat((doml_batch,domul_batch),axis=0) 
            ind_loss = torch.where(y_batch_t != -1)  # ici on ne conserve que les éléments pour lesquels on a un label 
                                                                                                            #ou un pseudo label de confiance
            #ind_loss_l =  [k for k in range(len(y_batch)) if y_batch[k] != torch.tensor(-1) and k < nbr_lb ] # les indices pour la loss des labélisés
            #ind_loss_ul =  [k for k in range(len(y_batch)) if y_batch[k] != torch.tensor(-1) and k > nbr_lb ] # les indices pour la loss des pseudo-labélisés
            
            
            pred_lab_s,pred_dom_s,emb_lab_s, emb_dom_s, pred_lab_t,pred_dom_t,emb_lab_t, emb_dom_t = model(x_batch_s, m_batch_s,x_batch_t, m_batch_t)
            
            emb_lab_s = nn.functional.normalize(emb_lab_s)
            emb_dom_s = nn.functional.normalize(emb_dom_s)
            emb_lab_t = nn.functional.normalize(emb_lab_t)
            emb_dom_t = nn.functional.normalize(emb_dom_t)
            
            loss_ortho_s = torch.mean(torch.sum(emb_lab_s*emb_dom_s,dim=1))
            loss_ortho_t = torch.mean(torch.sum(emb_lab_t*emb_dom_t,dim=1))
            
            loss_ortho_st = torch.mean(torch.sum(emb_lab_t*emb_lab_s,dim=1))
           
                     
           
           
            loss_lab_s = loss_fn(pred_lab_s, y_batch_s)
            loss_lab_t = loss_fn(pred_lab_t[ind_loss],y_batch_t[ind_loss])
            loss_dom_s = loss_fn(pred_dom_s,dom_batch_s)
            loss_dom_t = loss_fn(pred_dom_t,dom_batch_t)
            
            loss =  loss_lab_s + loss_lab_t  + 1/2*(loss_dom_s+loss_dom_t) + 1/2*(torch.abs(loss_ortho_s)+torch.abs(loss_ortho_t)) #+ torch.abs(loss_ortho_st)
            
            loss.backward()
            optimizer.step()
            #pred_npy = np.argmax(pred_lab_t.cpu().detach().numpy(), axis=1)
            #tot_pred.append( pred_npy )
            #tot_labels.append( y_batch_t.cpu().detach().numpy())
            
    
            
    #print(time.time()-start)    
    #tot_pred = np.concatenate(tot_pred)
    #tot_labels = np.concatenate(tot_labels)
    #fscore = f1_score(tot_pred, tot_labels, average="weighted")
    #fscore_a = np.round(fscore,3)
    #print(f'f_score {fscore_a}')
    
   # if n>warmup :
    #    
    #    tot_labels_pl=np.concatenate(tot_labels_pl)
    #    tot_pred_pl = np.concatenate(tot_pred_pl)
    #   
    #    pred_count = torch.tensor(pred_count).to('cpu').numpy()
    #    
    #    
    #    fscore_pseudo = f1_score(tot_pred_pl,tot_labels_pl, average= "weighted")
    #    fscore_pseudo_a = np.round(fscore_pseudo,5)
    #    print(f'le fscore sur les pseudo labels est {fscore_pseudo_a}')
    #    #print(Counter(tot_pred_pl))
    #    #print(Counter(tot_labels_pl))
    #    pred_count=np.concatenate(pred_count)
    #    #print(Counter(pred_count))
    #    count_thr = np.concatenate(count_thr)
    #    print(len(count_thr))
    #    k = np.concatenate(count_lab_thr)
    #    print(Counter(k))
        
        
      
    
tot_pred = []
tot_labels = []
k=0
s=0
for xm_batch, y_batch in test_dataloader:
    x_batch,mask_batch = xm_batch[:,:,:2],xm_batch[:,:,2]
    #print(torch.all(x_batch.eq(x_batch[0,:])).item())
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    mask_batch = mask_batch.to(device)
    pred_lab_s,pred_dom_s,emb_lab_s, emb_dom_s, pred_lab_t,pred_dom_t,emb_lab_t, emb_dom_t = model(x_batch_s, m_batch_s,x_batch_t, m_batch_t)
    
    pred_npy = np.argmax(pred_lab_t.cpu().detach().numpy(), axis=1)
    
    
    tot_pred.append( pred_npy )
    tot_labels.append( y_batch.cpu().detach().numpy())
tot_pred = np.concatenate(tot_pred)
tot_labels = np.concatenate(tot_labels)
print(tot_pred[:32],tot_pred[100:132])
print(np.unique(tot_pred))
print(tot_labels[:32])
print(Counter(tot_pred))
print(Counter(tot_labels))
fscore= f1_score(tot_pred, tot_labels, average="weighted")
print(fscore)
