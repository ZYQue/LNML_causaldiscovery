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
        mdl_norm_train = 0 # 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma**-2 * K))
    #print("mdl_lik_train:"+str(mdl_lik_train)+"\nmdl_pen_train"+str(mdl_pen_train)+"\nmdl_norm_train"+str(mdl_norm_train))
    return mdl_lik_train + mdl_pen_train + mdl_norm_train

def get_mdl(data):
    score_xy, score_yx, score_z, score_no = 0,0,0,0

    mdl_x = get_mdl_train(data["x"],data[['x_pa1','x_pa2']])
    mdl_y = get_mdl_train(data["y"],data['y_pa1'])
    print("mdl_x:"+str(mdl_x)+"\nmdl_y:"+str(mdl_y))
    mdl_yx = get_mdl_train(data["x"],data[['y','x_pa1','x_pa2']]) # y causes x
    mdl_xy = get_mdl_train(data["y"],data[['x','y_pa1']]) # x causes y
    print("mdl_x:"+str(mdl_x)+"\nmdl_y:"+str(mdl_y)+"\nmdl_yx:"+str(mdl_yx)+"\nmdl_xy:"+str(mdl_xy))

    score_xy = mdl_x+mdl_xy # score of x causes y
    score_yx = mdl_y+mdl_yx # score of y causes x
    score_z = mdl_xy+mdl_yx
    score_no = mdl_x+mdl_y
    print("\n\nscore_xy:"+str(score_xy)+"\nscore_yx:"+str(score_yx)+"\nscore_z:"+str(score_z)+"\nscore_no:"+str(score_no))
    

    return np.array([score_xy, score_yx, score_z, score_no])

if __name__ == '__main__':
    size=200
    dir_name = 'data/art/pairm/pairm_size'+str(size)+'sigma1poly1/'
    sets = 20
    ground_truth = pd.read_csv(dir_name+'truth.csv').to_numpy().squeeze()
    ans=np.zeros(sets)
    diffs=np.zeros(sets)
    ans_div=np.zeros(sets) # process answer with 3 and 4 divided
    ans_tog=np.zeros(sets) # process answer with 3 and 4 count together as 3

    for i in range(sets):
        print("===DATASET "+str(i)+"===")
        data = pd.read_csv(dir_name+str(i)+'.csv')
        #data_scaled = (data-data.min())/(data.max()-data.min())
        #data_scaled = (data-data.mean())/data.std()
        #print(data)
        score=np.zeros(4)
        #mdl_xy, mdl_yx, mdl_z, mdl_no= get_mdl(data)
        scores = get_mdl(data)
        print(f"{[np.argmin(scores)]} scores: {scores}")
        ans[i]=np.argmin(scores)+1
        #print("diff:"+str(mdl_xy-mdl_yx))
        diffs[i]=abs(scores[0]-scores[1])
        if diffs[i]<size/2: #small gap between mdl_xy and mdl_yx
            ans_div[i]=np.argmin(scores[-2:])+2+1
            ans_tog[i]=3
        else:
            ans_div[i]=np.argmin(scores[:2])+1 
            ans_tog[i]=np.argmin(scores[:2])+1 

    #auc = roc_auc_score(ground_truth,-diffs)
    #print(f"AUC = {auc:.4f}")
    print(f'ground_truth: {ground_truth}')
    print(f'answer_origin: {ans.astype(int)}')
    print(f'accuary: {np.mean(np.asarray(ground_truth)==ans)}')

    truth_pros = np.copy(np.asarray(ground_truth))
    truth_pros[truth_pros==4]=3
    print(f'truth_pros: {truth_pros.astype(int)}')
    print(f'answer_tog: {ans_tog.astype(int)}')
    print(f'answer_div: {ans_div.astype(int)}')
    print(f'ground_truth: {ground_truth}')
    print(f'acc_tog: {np.mean(truth_pros==ans_tog)}')
    print(f'acc_div: {np.mean(np.asarray(ground_truth)==ans_div)}')
    print(f'diffs: {diffs}')

    binary_truth = np.copy(np.asarray(ground_truth))
    binary_truth[binary_truth==2]=1
    binary_truth[binary_truth>1]=0
    auc = roc_auc_score(binary_truth,diffs)
    print(f"AUC = {auc:.4f}")
