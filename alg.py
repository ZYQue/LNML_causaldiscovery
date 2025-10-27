import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import roc_auc_score
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

GPR_CHOLESKY_LOWER = True

def get_mdl_train(x,x_parents):
    #print("x:\n"+str(x))
    #print("x_parents:\n"+str(x_parents))
    if x_parents.empty:
        ''' 
        x = np.asarray(x)
        mu_hat = np.mean(x)
        sigma_hat = np.std(x, ddof=0)
        dist_obj = norm(loc=mu_hat, scale=sigma_hat)

        logpdfs = dist_obj.logpdf(x)
        total_ll = np.sum(logpdfs)
        
        mdl_lik_train = -total_ll

        #K=np.identity(x.size)
        #K[np.diag_indices_from(K)] += 1e-1
        
        mdl_pen_train = 0
        mdl_norm_train = 0 #1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0])))
        '''
        x_parents=pd.DataFrame(np.ones(np.shape(x)))
    
        x_parents = np.asarray(x_parents).reshape(-1,1)
        kernel=ConstantKernel(1.0, (1e-3, 1e3))+ WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
        gpr=GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x_parents,x)

        # Model Complexity parameter alpha and MDL model score
        #K[np.diag_indices_from(K)] += 1e-1
        ''' 
        mat = np.eye(x_parents.shape[0]) * sigma**2 + K
        alpha = np.linalg.solve(mat, x)
        gpr.alpha_=alpha
        mdl_pen_train=alpha.T @ K @ alpha # alpha.T @ mat @ alpha is used in original code
        '''
        mdl_pen_train = 0

        # MDL data score/log likelihood
        mdl_lik_train = -gpr.log_marginal_likelihood_value_
        # precompute for self.mdl_score():
        #L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        #gpr.L_ = L

        # MDL normalization term
        # let this term be 0 if original normalization term is INF
        mdl_norm_train = 0 # 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**-2 * K))

    else:
        x_parents = np.asarray(x_parents)
        if x_parents.ndim==1:
            #print(f'one dim x_parents: {x_parents}')
            x_parents=x_parents.reshape(-1,1)
        kernel=RBF(1.0)+WhiteKernel(noise_level=1.0)
        gpr=GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x_parents,x)
        K=gpr.kernel_(x_parents)
        sigma=1

        # Model Complexity parameter alpha and MDL model score
        #K[np.diag_indices_from(K)] += 1e-1
        #print("K:\n"+str(K))
        mat = np.eye(x_parents.shape[0]) * sigma**2 + K
        alpha = np.linalg.solve(mat, x)
        gpr.alpha_=alpha
        mdl_pen_train=alpha.T @ K @ alpha # alpha.T @ mat @ alpha is used in original code

        # MDL data score/log likelihood
        mdl_lik_train = -gpr.log_marginal_likelihood_value_
        # precompute for self.mdl_score():
        L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        gpr.L_ = L

        # MDL normalization term
        # let this term be 0 if original normalization term is INF
        mdl_norm_train = 0 # 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**-2 * K))
    #print("mdl_lik_train:"+str(mdl_lik_train)+"\nmdl_pen_train"+str(mdl_pen_train)+"\nmdl_norm_train"+str(mdl_norm_train))
    return mdl_lik_train + mdl_pen_train + mdl_norm_train


def get_pair_scores(x_data,x_parents,y_data,y_parents):
    # input: data of little system related to node X and Y (X<Y), cf==False if no confounder Z
    # output: scores of X->Y, Y->X, X<-Z->Y, no edge, confidence
    score_XY, score_YX, score_cf, score_no, confidence = 0,0,0,0,0

    #print("get_pair_scores()")
    #print("x_data:\n"+str(x_data))
    #print("\nx_parents:\n"+str(x_parents))
    #print("y_data:\n"+str(y_data))
    #print("\ny_parents:\n"+str(y_parents))
    #print("\nZ:\n:"+str(Z))


    # gpr_x: node x and its parents
    mdl_x=get_mdl_train(x_data,x_parents)
    # gpr_xy: node x and its parents and new parent y
    xp_yd=pd.concat([x_parents, y_data],axis=1)
    xp_yd.columns=xp_yd.columns.astype(str)
    mdl_xy=get_mdl_train(x_data,xp_yd)
    # gpr_y: node y and its parents
    mdl_y=get_mdl_train(y_data,y_parents)
    # gpr_yx: node y and its parents and new parent x
    yp_xd=pd.concat([y_parents, x_data],axis=1)
    yp_xd.columns=yp_xd.columns.astype(str)
    mdl_yx=get_mdl_train(y_data,yp_xd)

    score_xy = mdl_x+mdl_xy # score of x causes y
    score_yx = mdl_y+mdl_yx # score of y causes x
    score_z = mdl_xy+mdl_yx
    score_no = mdl_x+mdl_y
    print("\n\nscore_xy:"+str(score_xy)+"\nscore_yx:"+str(score_yx)+"\nscore_z:"+str(score_z)+"\nscore_no:"+str(score_no))

    gain_XY=max(0,score_no-score_xy)
    gain_YX=max(0,score_no-score_yx)
    confidence=abs(gain_XY-gain_YX)
    #confidence=max(score_xy,score_yx) - min(score_xy,score_yx)

    edge_type = get_edge_type(score_xy,score_yx,score_z,score_no,direction_only=False, threshold=0)
    if edge_type == 3 or edge_type == 4:
        confidence = 0
    
    return score_xy, score_yx, score_z, score_no, confidence


def get_index(conf,c):
    # find the index pair of node with c-th max confidence
    while c>0:
        max_ind = np.unravel_index(np.argmax(conf, axis=None), conf.shape)
        conf[max_ind]=0
        c=c-1
        #print(conf)
        #print(c)
    if np.sum(conf)==0:
        return -1, -1
    else:
        ind = np.unravel_index(np.argmax(conf, axis=None), conf.shape)
        return ind[0],ind[1]


def get_edge_type(s1,s2,s3,s4,direction_only=False, threshold=200):
    # input: the score of X->Y, Y->X, X<-Z->Y, no edge.
    # output: 1,2,3,4, respectively
    if direction_only==True:
        if s1<s2:
            return 1
        else:
            return 2


    if abs(s1-s2)>=threshold:
        if min(s1,s2,s3,s4)==s1:
            return 1
        elif min(s1,s2,s3,s4)==s2:
            return 2
        elif min(s1,s2,s3,s4)==s3:
            return 3
        else:
            return 4
    else: # gap between X->Y and X<-Y is small, show psudo-collinearity
        if s3<s4:
            return 3
        else:
            return 4


def check_cycle(x,y,edge_type,graph):
    # output: True for cycle exists, False otherwise
    #print("x:"+str(x))
    #print("y:"+str(y))
    #print("edge_type:"+str(edge_type))
    #print("graph:\n"+str(graph))
    if edge_type == 1:
        # edge X->Y
        children = np.where(graph[y,:]==1)[0]
        while children.size>0:
            if np.any(children==x):
                return True
            pop=children[0]
            pop_c=np.where(graph[pop,:]==1)[0]
            children=np.delete(children,0)
            children=np.concatenate((children,pop_c))
        return False
    elif edge_type == 2:
        # edge Y->X
        children = np.where(graph[x,:]==1)[0]
        while children.size>0:
            if np.any(children==y):
                return True
            pop=children[0]
            pop_c=np.where(graph[pop,:]==1)[0]
            children=np.delete(children,0)
            children=np.concatenate((children,pop_c))
        return False
    else:
        return False


def add_edge(ix,iy,edge_type,graph,searching):
    dim=graph.shape[0]
    last_child=[]
    searching[ix,iy]=0
    searching[iy,ix]=0
    if edge_type==1:
        # add edge X->Y
        graph[ix,iy]=1
        graph[iy,:]=0 # put the edge between iy and its children back to queue
        searching[:,iy]=1 # reset the flags
        searching[iy,:]=1
        searching[iy,iy]=0
        p_tmp=np.where(abs(graph[:,iy])==1)[0] # flags of the edge between iy and its parents are remained
        parents=[x for x in p_tmp if x < dim]
        searching[parents,iy]=0
        searching[iy,parents]=0
        last_child=[iy]
    elif edge_type==2:
        # add edge Y->X
        graph[iy,ix]=1
        graph[ix,:]=0 # put the edge between ix and its children back to queue
        searching[:,ix]=1 # reset the flags
        searching[ix,:]=1
        searching[ix,ix]=0
        p_tmp=np.where(abs(graph[:,ix])==1)[0] # flags of the edge between iy and its parents are remained
        parents=[x for x in p_tmp if x < dim]
        searching[parents,ix]=0
        searching[ix,parents]=0
        last_child=[ix]
    '''
    elif edge_type==3:
        # X<-Z->Y
        graph[ix,iy]=3
        graph[iy,ix]=3
        last_child=[ix,iy]
    elif edge_type==4:
        # no edge between X and Y
        graph[ix,iy]=4
        graph[iy,ix]=4
    '''
    return graph, searching, last_child


def get_graph_mdl(graph, data):
    # input: data and causal graph
    # output: mdl score for the whole graph
    score=0
    dim=graph.shape[0]

    if not isinstance(graph, np.ndarray):
        graph = graph.to_numpy()

    for i in range(dim):
        i_parents = np.where(graph[:,i]==1)[0]
        score += get_mdl_train(data.iloc[:,i], data.iloc[:,i_parents.tolist()])

    return score


def check_pseudo_colli(x_data,x_parents,y_data,y_parents,no_edge=False):
    # output: 
    length = x_data.shape[0]
    score_xy, score_yx, score_z, score_no, confidence = 0,0,0,0,0
    score_xy, score_yx, score_z, score_no, confidence = get_pair_scores(x_data,x_parents,y_data,y_parents)
    if no_edge == False:
        return get_edge_type(score_xy, score_yx, score_z, score_no, direction_only=False, threshold=length*5)
    else:
        if score_z<score_no:
           return 3
        else:
           return 4
    


def forward_search(graph,data):
    length, dim = data.shape
    mdl_score=math.inf

    score=np.zeros((2,dim,dim)) #(0,X,Y) for X->Y, (0,Y,X) for Y->X, (1,min(X,Y), max(X,Y)) for Y<-Z->X, (1,max(X,Y),min(X,Y)) for no edge
    confidence=np.zeros((dim,dim))      # MAX(X->Y, Y->X) - MIN(X->Y, Y->X)
    if graph.sum()==0:
        searching=np.ones((dim,dim))
        np.fill_diagonal(searching,0)       # matrix searching becomes a zero matrix when each pair of nodes are checked
    else:
        searching = (graph==0).astype(int)
        np.fill_diagonal(searching,0)

    last_child=[]
    initial=True
    
    while(searching.sum()!=0):
        for i in range(dim):
            for j in range(i+1,dim):
                # compute the score in different cases in each pair of the nodes (i<j)
                #print("\ni:"+str(i)+"\tj:"+str(j))
                if searching[i,j]==0:
                    continue
                if (not (i in last_child or j in last_child)) and initial==False :
                    continue
                i_parents = np.where(graph[:,i]==1)[0]
                j_parents = np.where(graph[:,j]==1)[0]
                #print("i's parents:" +str(i_parents.tolist()))
                #print("j's parents:" +str(j_parents.tolist()))
                #print("data:\n"+str(data))
                score[0,i,j], score[0,j,i], score[1,i,j], score[1,j,i], confidence[i,j] = get_pair_scores(data.iloc[:,i],data.iloc[:,i_parents.tolist()],data.iloc[:,j],data.iloc[:,j_parents.tolist()])

        initial=False

        #print(f'confidence:\n{confidence}')
        
        # find the pairwise nodes ix, iy with max confidence
        counter=0
        while True: # loop to avoid cycle appears
            ix,iy = get_index(confidence,counter)
            #print("ix:"+str(ix)+"\tiy"+str(iy))
            if ix==-1 and iy==-1:       # no residue edge, search finished
                searching=np.zeros((dim,dim))
                break

            edge_type = get_edge_type(score[0,ix,iy], score[0,iy,ix], score[1,ix,iy], score[1,iy,ix], direction_only=True)
            #print("edge_type:"+str(edge_type))

            if check_cycle(ix,iy,edge_type,graph) == True: # new added edge leads to cycle, skip to next pair of nodes
                counter=counter+1
                continue

            # add the edge to graph
            graph, searching, last_child = add_edge(ix,iy,edge_type,graph,searching)
            confidence[ix,iy]=0
            #print("confidence:\n"+str(confidence))
            #print("graph:\n"+str(graph))
            #print("searching:\n"+str(searching))
            break
        
    print("FINAL graph:\n"+str(graph))
    #mdl_score=get_graph_mdl(graph, data)
    return graph
    

def identify_latent(graph_fs, data):
    length, dim = data.shape
    DAG=graph_fs.copy()
    searching = DAG.copy()

    while(searching.sum()!=0):
        for i in range(dim):
            for j in range(dim):
                if DAG[i,j]==0 or searching[i,j]==0:
                    continue
                print(f'i: {i}\tj: {j}')
                i_parents = np.where(DAG[:,i]==1)[0]
                j_parents = np.where(DAG[:,j]==1)[0]
                no_ij_edge=False
                tmp_DAG=DAG.copy()
                tmp_DAG[i,j]=0
                if get_graph_mdl(tmp_DAG, data)<get_graph_mdl(DAG, data):
                    no_ij_edge=True
                    #print('No edge.')
                edge_type = check_pseudo_colli(data.iloc[:,i],data.iloc[:,i_parents.tolist()],data.iloc[:,j],data.iloc[:,j_parents.tolist()],no_edge=no_ij_edge)
                #print(f'edge_type: {edge_type}')
                if edge_type == 3:
                    DAG[i,j]=3
                    DAG[j,i]=3
                    j_children = np.where(DAG[j,:]==1)[0]
                    searching[j,j_children]=1
                    searching[i,j]=0
                elif edge_type == 4:
                    DAG[i,j]=0
                    searching[i,j]=0
                else:
                    searching[i,j]=0
        #print(f'DAG:\n{DAG}')
        #print(f'searching:\n{searching}')

    return DAG, False # False for end the algorithm


if __name__ == '__main__':
    size=1000
    sets=20
    dir_name='data/art/multi/multi_size'+str(size)+'n6z1poly3/'
    result_dir = dir_name+'results/'
    try:
        os.makedirs(result_dir)
        print(f"Directory '{result_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory'{result_dir}' already exist.")

    for i in range(sets):
        print("===DATASET "+str(i)+"===")
        data = pd.read_csv(dir_name+str(i)+'_noZ.csv')
        length, dim = data.shape
        #print(f'data"{data}')
        true_graph = pd.read_csv(dir_name+str(i)+'_DAG_noZ.csv',header=None)
        print(f'true_graph:{true_graph}')
        graph_fs = forward_search(np.zeros((dim,dim)),data)
        #graph_fs =np.array([[0,1,0,1,1],[0,0,0,0,1],[0,1,0,1,1],[0,1,0,0,1],[0,0,0,0,0]])
        loop = True
        while loop:
            esti_graph,loop = identify_latent(graph_fs,data)

        np.savetxt(result_dir+'/'+str(i)+'_DAG.csv',esti_graph.astype(int),delimiter=",",fmt="%d")

        mdl_score_true = get_graph_mdl(true_graph, data)
        #print(f'mdl_score_true: {mdl_score_true}')
        mdl_score_fs = get_graph_mdl(graph_fs, data)
        #print(f'mdl_score_fs: {mdl_score_fs}')
        mdl_score_esti = get_graph_mdl(esti_graph, data)
        #print(f'mdl_score_esti: {mdl_score_esti}')
        mdl_score_null = get_graph_mdl(np.zeros((5,5)),data)
        #print(f'mdl_score_null: {mdl_score_null}')
        scores = np.array([mdl_score_true,mdl_score_fs,mdl_score_esti,mdl_score_null],dtype=float)
        np.savetxt(result_dir+'/'+str(i)+'_scores.csv',scores.astype(float),delimiter=",",fmt="%.6f")
