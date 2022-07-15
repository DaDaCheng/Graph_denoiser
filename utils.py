from sklearn.neighbors import kneighbors_graph
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from scipy import sparse
import torch


def adj_from_distance(X,n_neighbors=10,directed=False):
    A=np.zeros((X.shape[0],X.shape[1]),dtype=np.int)
    for i in range(X.shape[0]):
        diss_list=X[i]
        neb_list=np.argsort(diss_list)[1:n_neighbors+1]
        A[i][neb_list]=1
    if directed:
        return A
    else:
        A=A+A.T
        A[A>0]=1
        return A

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx,:].float(), 'label': float(self.label[idx])}
        return sample


def neighbors(fringe, A, outgoing=True):
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)
    return res

def k_hop_subgraph(src, dst, A, A_csc, num_hops=2,directed=True):
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]
    return subgraph


def prepare_train_dataset(A,walk_len=4,num_hops=2,batch_size=512,val_ratio=0.05,directed=False):
    N=A.shape[0]
    edge_index=np.arange(N*N)

    if directed:
        pos_edge_index=edge_index[A.reshape(-1)==1]
        matrix_index_mask=np.ones(A.shape,dtype=bool)
        matrix_index_mask[A==0]=False
        matrix_index_mask[np.eye(A.shape[0], dtype=int)==1]=True
        neg_edge_index=edge_index[matrix_index_mask.reshape(-1)==False]
        A=sparse.csr_matrix(A)
        A_csc=A.tocsc()
    else:
        # A=A+A.T
        # A[A>0]=1
        Atriu=np.triu(A,k=1)
        pos_edge_index=edge_index[Atriu.reshape(-1)==1]
        matrix_index_mask=np.ones(A.shape,dtype=bool)
        matrix_index_mask[A==0]=False
        matrix_index_mask[np.eye(A.shape[0], dtype=int)==1]=True
        matrix_index_mask= np.bitwise_not(matrix_index_mask)
        neg_edge_index=edge_index[np.triu(matrix_index_mask,k=1).reshape(-1)==True]
        A=sparse.csr_matrix(A)
        A_csc=None

    neg_edge_index=np.random.choice(neg_edge_index,len(pos_edge_index),replace=False)
    edge_index=np.concatenate((pos_edge_index,neg_edge_index),axis=0)
    train_data_list=np.empty((len(edge_index),2*walk_len),dtype=np.int64)
    train_label_list=np.empty(len(edge_index))

    index=0
    #for i in tqdm(edge_index):
    for i in edge_index:
        xi=i//N
        yi=i%N
        A_t=k_hop_subgraph(xi,yi,A,A_csc,num_hops=num_hops,directed=directed)
        A_tp=A_t.copy()
        A_tm=A_t.copy()
        A_tp[0,1]=1
        A_tm[0,1]=0
        wp=np.empty((walk_len,2))
        A_tp_t=A_tp.copy()
        A_tm_t=A_tm.copy()
        A_tp_t=A_tp@A_tp_t
        A_tm_t=A_tm@A_tm_t
        for j in range(walk_len):
            A_tp_t=A_tp@A_tp_t
            A_tm_t=A_tm@A_tm_t
            wp[j,0]=(A_tp_t[0,1])
            wp[j,1]=(A_tm_t[0,1])
        train_data_list[index]=wp.reshape(-1)
        train_label_list[index]=A_t[0,1]
        index=index+1

    train_data_list=torch.tensor(train_data_list,dtype=torch.int)
    train_label_list=torch.tensor(train_label_list,dtype=torch.int)

    perm = torch.randperm(train_data_list.size(0))
    train_data_list=train_data_list[perm]
    train_label_list=train_label_list[perm]

    num_div=int(val_ratio*train_data_list.size(0))
    train_data,val_data=train_data_list[num_div:],train_data_list[:num_div]
    train_label,val_label=train_label_list[num_div:],train_label_list[:num_div]
    train_loader = DataLoader(dataset=MyDataset(train_data,train_label), 
                                               batch_size=batch_size, 
                                               shuffle=True)
    val_loader = DataLoader(dataset=MyDataset(val_data,val_label), 
                                               batch_size=batch_size, 
                                               shuffle=True)

    return train_loader,val_loader


def prepare_test_dataset_asycn(A,A_range,walk_len=4,num_hops=2,batch_size=512,directed=False):
    N=A.shape[0]
    edge_index=np.arange(N*N)

    A=A.astype(np.int)
    A_range=A_range.astype(np.int)
    if directed:
        A_rangeDiff=A_range-A
        A_rangeDiff[A_rangeDiff<0]=0
        pos_edge_index=edge_index[A.reshape(-1)==1]
        neg_edge_index=edge_index[A_rangeDiff.reshape(-1)==1]
        A=sparse.csr_matrix(A)
        A_csc=A.tocsc()
    else:
        A=A+A.T
        A[A>0]=1
        A_range=A_range+A_range.T
        A_range[A_range>0]=1
        A_rangeDiff=A_range-A
        A_rangeDiff[A_rangeDiff<0]=0
        pos_edge_index=edge_index[np.triu(A,k=1).reshape(-1)==1]
        neg_edge_index=edge_index[np.triu(A_rangeDiff,k=1).reshape(-1)==1]
        A=sparse.csr_matrix(A)
        A_csc=None


    edge_index=np.concatenate((pos_edge_index,neg_edge_index),axis=0)
    edge_index=np.unique(edge_index)
    edge_index=edge_index.astype(np.int64)
    test_data_list=np.empty((len(edge_index),2*walk_len),dtype=np.int64)
    test_label_list=np.empty(len(edge_index))
    
    index=0
    for i in edge_index:
        xi=i//N
        yi=i%N
        A_t=k_hop_subgraph(xi,yi,A,A_csc,directed=directed)
        A_tp=A_t.copy()
        A_tm=A_t.copy()
        A_tp[0,1]=1
        A_tm[0,1]=0
        wp=np.empty((walk_len,2))
        A_tp_t=A_tp.copy()
        A_tm_t=A_tm.copy()
        A_tp_t=A_tp@A_tp_t
        A_tm_t=A_tm@A_tm_t
        for j in range(walk_len):
            A_tp_t=A_tp@A_tp_t
            A_tm_t=A_tm@A_tm_t
            wp[j,0]=(A_tp_t[0,1])
            wp[j,1]=(A_tm_t[0,1])
        test_data_list[index]=wp.reshape(-1)
        test_label_list[index]=A_t[0,1]
        index=index+1

    test_data=torch.tensor(test_data_list,dtype=torch.int)
    test_label=torch.tensor(test_label_list,dtype=torch.int)
    test_loader = DataLoader(dataset=MyDataset(test_data,test_label), 
                                            batch_size=batch_size, 
                                            shuffle=False)

    return test_loader,edge_index







def update_A_async(A,edge_index,scores,N,n_neighbors,directed,fix_n=False):
    if not directed:
        edge_index_p=edge_index//N
        edge_index_q=edge_index%N
        edge_index_l=edge_index_p+edge_index_q*N
        edge_index=np.concatenate((edge_index,edge_index_l),axis=0)
        scores=np.concatenate((scores,scores),axis=0)
    perm=np.argsort(edge_index)
    edge_index=edge_index[perm]
    scores=scores[perm]
    degree=A.sum(axis=1).astype(np.int)
    A_new=np.zeros((N,N),dtype=np.int64)
    index=0
    for i in range(N):
        edge_node=[]
        score_node=[]
        while edge_index[index]<N*(i+1):
            edge_node.append(edge_index[index]%N)
            score_node.append(scores[index])
            index=index+1
            if index==len(edge_index):
                break

        edge_node=np.array(edge_node,dtype=np.int64)
        score_node=np.array(score_node)
        perm=np.argsort(score_node)
        if fix_n:
            edge_node=edge_node[perm[-n_neighbors[i]:]]
        else:
            edge_node=edge_node[perm[-degree[i]:]]
        A_new[i,edge_node]=1

    if not directed:
        A_new=A_new+A_new.T
        A_new[A_new>0]=1
    return A_new





def updata_A_sycn(A,check_list,node_index,n_neighbors,walk_len,directed,test,thre):
    N=A.shape[0]
    node_index_list=np.arange(N)
    #check_list
    test_data_list=np.empty((len(check_list),2*walk_len),dtype=np.int64)
    test_label_list=np.empty(len(check_list))
    index=0
    for i in check_list:
        xi=node_index
        yi=i
        A_t=k_hop_subgraph(xi,yi,A,None,directed=directed)
        #A_t=A_t.tolil()
        A_tp=A_t.copy()
        A_tm=A_t.copy()
        A_tp[0,1]=1
        A_tm[0,1]=0
        wp=np.empty((walk_len,2))
        A_tp_t=A_tp.copy()
        A_tm_t=A_tm.copy()
        A_tp_t=A_tp@A_tp_t
        A_tm_t=A_tm@A_tm_t
        for j in range(walk_len):
            A_tp_t=A_tp@A_tp_t
            A_tm_t=A_tm@A_tm_t
            wp[j,0]=(A_tp_t[0,1])
            wp[j,1]=(A_tm_t[0,1])
        test_data_list[index]=wp.reshape(-1)
        test_label_list[index]=A_t[0,1]
        index=index+1
    test_data=torch.tensor(test_data_list,dtype=torch.int)
    test_label=torch.tensor(test_label_list,dtype=torch.int)
    test_loader = DataLoader(dataset=MyDataset(test_data,test_label), 
                                            batch_size=N, 
                                            shuffle=False)
    scores=test(test_loader,output_AUC=False)
    

    
    delet_index=check_list[scores<thre]
    for i in delet_index:
        A[node_index,i]=0
        A[i,node_index]=0
    # argsort_score=np.argsort(-scores)
    # argsort_index=check_list[argsort_score]

    # for i in argsort_index[:n_neighbors]:
    #     if i !=node_index:
    #         A[node_index,i]=1
    #         A[i,node_index]=1
    # for i in argsort_index[n_neighbors:]:
    #     A[node_index,i]=0
    #     A[i,node_index]=0

    return A


