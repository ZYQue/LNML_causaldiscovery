import os
import numpy as np
import pandas as pd


def get_DAG(n_nodes,p):
    rng = np.random.default_rng()

    order = rng.permutation(n_nodes)

    adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    for k in range(n_nodes):
        for l in range(k + 1, n_nodes):
            if rng.random() < p:
                src = order[k]
                dst = order[l]
                adj[src, dst] = 1
    print(f'order: {order}')
    print(f'adj: {adj}')
    indeg = adj.sum(axis=0)
    outdeg = adj.sum(axis=1)
    mask_root = (indeg == 0)
    confounders = np.where(mask_root & (outdeg == 2))[0]
    print(f'confounders: {confounders}')
    return adj,order,confounders

def gen_data(adj,n_nodes,n_samples,order,mech='poly',root_range=(-3,3), coeff_range=(-2,2),degree=3,noise_sigma=0.3,root_mean=0,root_std=1.0):
    rng = np.random.default_rng()
    edge_coeffs = {}
    root_low, root_high=root_range
    low, high = coeff_range
    for src in range(n_nodes):
        for dst in range(n_nodes):
            if adj[src, dst] == 1:
                edge_coeffs[(src, dst)] = rng.uniform(low, high, size=degree + 1)
    print(f'edge_coeffs: {edge_coeffs}')

    X = np.zeros((n_samples, n_nodes), dtype=float)

    # nodes with no parents
    parents = [np.where(adj[:, i] == 1)[0].tolist() for i in range(n_nodes)]
    roots = [i for i in range(n_nodes) if len(parents[i]) == 0]

    # generate data in order
    for idx in order:
        pa = parents[idx]
        if len(pa) == 0:
            # nodes with no parents
            X[:, idx] = rng.uniform(root_low, root_high, size=n_samples)+rng.normal(loc=root_mean, scale=root_std, size=n_samples)
        else:
            # nodes with parents
            val = np.zeros(n_samples, dtype=float)
            for src in pa:
                coeff = edge_coeffs[(src, idx)]  # c0..c_degree
                # compute c0 + c1*x + c2*x^2 + ...
                xj = X[:, src]
                poly = np.zeros(n_samples, dtype=float)
                power = np.ones(n_samples, dtype=float)  # x^0
                for d in range(degree + 1):
                    poly += coeff[d] * power
                    power *= xj
                val += poly
            # add noise
            val += rng.normal(0.0, noise_sigma, size=n_samples)
            X[:, idx] = val
    cols = [f"{i}" for i in range(n_nodes)]
    df = pd.DataFrame(X, columns=cols)
    
    return df


if __name__ == '__main__':
    n=6 # number of nodes (include latent confounders)
    n_z=1 # number of latent confounder
    p=0.25 # probabilty for Erdos-Renyi
    size=500 # size of each data set
    sets = 20 # number of data sets
    deg = 3 # degree of polynomial
    
    dir_name = 'multi_size'+str(size)+'n'+str(n)+'z'+str(n_z)+'poly'+str(deg)

    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory'{dir_name}' already exist.")

    for i in range(sets):
        while True:
            DAG,order,confounders = get_DAG(n,p)
            if confounders.size < n_z:
                print('No latent confounder. REGENERATE.')
            else:
                break
        indices = np.where(DAG[confounders[0]]==1)[0]
        #print(f'indices:{indices}')
        for k in indices:
            for j in indices:
                if k!=j:
                    DAG[k,j]=3
        DAG_new = np.delete(np.delete(DAG,confounders[0],axis=0),confounders[0],axis=1)
        print(f'DAG_new:{DAG_new}')
        data = gen_data(DAG,n,size,order,mech='poly',degree=deg)
        data.to_csv(dir_name+'/'+str(i)+'.csv',index=False)
        np.savetxt(dir_name+'/'+str(i)+'_order.csv',order.astype(int),delimiter=",",fmt="%d")
        np.savetxt(dir_name+'/'+str(i)+'_DAG.csv',DAG.astype(int),delimiter=",",fmt="%d")
        data_new=data.drop(columns=data.columns[confounders[0]])
        data_new.to_csv(dir_name+'/'+str(i)+'_noZ.csv',index=False)
        np.savetxt(dir_name+'/'+str(i)+'_DAG_noZ.csv',DAG_new.astype(int),delimiter=",",fmt="%d")
