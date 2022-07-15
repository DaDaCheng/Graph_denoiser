import torch
import numpy as np
from model import MLP
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from utils import adj_from_distance,prepare_train_dataset,prepare_test_dataset_asycn,update_A_async,updata_A_sycn
import warnings


class WPGD(object):
    def __init__(self,num_hops=2,lr=0.01,weight_decay=0.0005,walk_len=4,val_ratio=0.05,batch_size=512):
        self.n_neighbors=4
        self.num_hops=num_hops
        self.batch_size=batch_size
        self.val_ratio=val_ratio
        self.walk_len=walk_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=MLP(2*self.walk_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        warnings.filterwarnings("ignore")
    def train(self,loader,epoch):
        self.model.train()
        loss_epoch=0
        for data in loader:  
            input = data['data'].to(self.device)
            label = data['label'].to(self.device)
            if input.shape[0]==1:
                continue
            out = self.model(input)
            loss = self.criterion(out.view(-1), label)  
            self.optimizer.zero_grad()
            loss.backward()  
            self.optimizer.step()
            loss_epoch=loss_epoch+loss.item()
        return loss_epoch/len(loader)

    def test(self,loader,output_AUC=True):
        self.model.eval()
        loss_epoch=0
        scores = torch.tensor([]).to(self.device)
        labels = torch.tensor([])
        for data in loader:  
            input = data['data'].to(self.device)
            label = data['label'].to(self.device)
            if input.shape[0]==1:
                continue
            out = self.model(input)
            scores = torch.cat((scores,out),dim = 0)
            labels = torch.cat((labels,label.view(-1,1).cpu().clone().detach()),dim = 0)
        scores_np = scores.cpu().clone().detach().numpy()
        labels = labels.cpu().clone().detach().numpy()
        if output_AUC:
            return scores_np.reshape(-1) ,roc_auc_score(labels, scores_np)
        else:
            return scores_np.reshape(-1)




    def target_graph(self,A,input_form='adj',n_neighbors=4,directed=False,train_epoch=20):
        print('Learning target graph ......')
        if input_form=='adj':
            pass
        elif input_form=='dist':
            A=adj_from_distance(A,n_neighbors=n_neighbors,directed=directed)
        elif input_form=='feat':
            A=kneighbors_graph(M, n_neighbors=n_neighbors).toarray()
        A=A.astype(np.int)
        self.directed=directed
        if not directed:
            A=A+A.T
            A[A>0]=1
        train_loader,val_loader=prepare_train_dataset(A,walk_len=self.walk_len,
            num_hops=self.num_hops,batch_size=self.batch_size,val_ratio=self.val_ratio,directed=self.directed)


        for epoch in range(train_epoch):
            loss_epoch = self.train(train_loader,epoch)
            _,AUC = self.test(val_loader)
            if AUC>0.99 and epoch>10:
                break
            print(AUC)
        return AUC

    def denoise_async(self,A,input_form='adj',n_neighbors=4,denoise_scale=1.0):
        print('Dsenoising input graph ......')
        if input_form=='adj':
            N=A.shape[0]
            A_range=torch.ones(N,N)*np.sum(A)/(N**2)*denoise_scale
            A_range=torch.bernoulli(A_range)
            A_range=A_range.numpy()
            if self.directed:
                A_range[range(N)]=0
            else:
                A_range=np.triu(A_range,k=1)
                A_range=A_range+A_range.T
                A_range[A_range>0]=1
                A=A+A.T
                A[A>0]=1
            A=A.astype(np.int)
            A_range=A_range.astype(np.int)
        elif input_form=='dist':
            A_range=adj_from_distance(A,n_neighbors=int(n_neighbors*(1+denoise_scale)),directed=self.directed)
            N=A_range.shape[0]
        elif input_form=='feat':
            A_range=kneighbors_graph(A, n_neighbors=int(n_neighbors*(1+denoise_scale))).toarray()
            if self.directed:
                pass
            else:
                A_range=np.triu(A_range,k=1)
                A_range=A_range+A_range.T
                A_range[A_range>0]=1
            N=A_range.shape[0]
        test_loader,edge_index=prepare_test_dataset_asycn(A,A_range,walk_len=self.walk_len,num_hops=self.num_hops,
            batch_size=self.batch_size,directed=self.directed)

        scores,AUC=self.test(test_loader,output_AUC=False)
        new_A=update_A_async(A,edge_index,scores,N,n_neighbors,self.directed)
        return new_A



    def denoise_sync(self,A,input_form='adj',n_neighbors=4,denoise_scale=1.0,thre=0.2):
        print('Dsenoising input graph ......')
        ## only delete links, only for undirected
        ## TODO:...
        A=A+A.T
        A[A>0]=1
        A=A.astype(np.int)
        N=A.shape[0]
        node_index_list=np.arange(N,dtype=np.int)
        Asparse=sparse.csr_matrix(A)
        for i in range(N):
            node_index=i
            check_list=node_index_list[A[node_index]==1]
            #node_index=np.random.randint(N)
            Asparse=updata_A_sycn(Asparse,check_list,node_index,n_neighbors,walk_len=self.walk_len,directed=self.directed,test=self.test,thre=thre)
        A=Asparse.todense()
        A=np.array(A,dtype=np.int)
        A=A+A.T
        A[A>0]=1
        return A

if __name__ == "__main__":
    N=30
    A=torch.ones(N,N)*0.1
    A1=torch.bernoulli(A).numpy()
    A2=torch.bernoulli(A).numpy()
    Graph_denoiser=WPGD()
    Graph_denoiser.target_graph(A1)
    Graph_denoiser.denoise_sync(A2)

