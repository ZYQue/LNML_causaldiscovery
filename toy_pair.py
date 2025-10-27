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
        
        x_parents = np.asarray(x_parents).reshape(-1, 1)
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
        x_parents = np.asarray(x_parents).reshape(-1, 1)
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

    mdl_x = get_mdl_train(data["x"],pd.DataFrame())
    mdl_y = get_mdl_train(data["y"],pd.DataFrame())
    #print("mdl_x:"+str(mdl_x)+"\nmdl_y:"+str(mdl_y))
    mdl_yx = get_mdl_train(data["x"],data["y"]) # y causes x
    mdl_xy = get_mdl_train(data["y"],data["x"]) # x causes y
    #print("mdl_x:"+str(mdl_x)+"\nmdl_y:"+str(mdl_y)+"\nmdl_yx:"+str(mdl_yx)+"\nmdl_xy:"+str(mdl_xy))

    score_xy = mdl_x+mdl_xy # score of x causes y
    score_yx = mdl_y+mdl_yx # score of y causes x
    score_z = mdl_xy+mdl_yx
    score_no = mdl_x+mdl_y
    print("\n\nscore_xy:"+str(score_xy)+"\nscore_yx:"+str(score_yx)+"\nscore_z:"+str(score_z)+"\nscore_no:"+str(score_no))
    

    return score_xy, score_yx, score_z, score_no

if __name__ == '__main__':
    dir_name = 'data/art/pair_size200sigma02poly3/'
    sets = 20
    ground_truth = pd.read_csv(dir_name+'truth.csv')
    diffs=np.zeros(sets)
    print("ground_truth:"+str(ground_truth))

    for i in range(sets):
        print("===DATASET "+str(i)+"===")
        data = pd.read_csv(dir_name+str(i)+'.csv')
        #data_scaled = (data-data.min())/(data.max()-data.min())
        #data_scaled = (data-data.mean())/data.std()
        #print(data)
        mdl_xy, mdl_yx, mdl_z, mdl_no= get_mdl(data)
        print("diff:"+str(mdl_xy-mdl_yx))
        diffs[i]=abs(mdl_xy-mdl_yx)

    auc = roc_auc_score(ground_truth,-diffs)
    print(f"AUC = {auc:.4f}")
