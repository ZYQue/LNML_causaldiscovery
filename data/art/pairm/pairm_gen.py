import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gen_data(dir_name,n,Z):
    low=-3
    high=3
    sigma_node=0.2
    sigma=0.2
    poly_degree=1
    coeff_range=(-2,2)
    seed=42
    
    rng = np.random.default_rng()

    #poly_coeffs = np.zeros((5,poly_degree + 1))
    #poly_coeffs[:,poly_degree-1]=1
    poly_coeffs = rng.uniform(coeff_range[0], coeff_range[1], size=(5,poly_degree + 1)) #0,1:x_pa->X, 2:y_pa->Y, 3:Z->X, 4:Z->Y
    #print(f"poly_coeffs: {poly_coeffs}")
    
    node_base = rng.uniform(low, high, size=(4,n)) #nodes w/o parents: 1,2:x_pa, 3:y_pa, x:Z
    node = node_base + rng.normal(loc=0.0, scale=sigma_node, size=(4,n))
    #print(f"node: {node}")

    if Z==1: # X->Y
        x_clean = np.polyval(poly_coeffs[0], node[0])+np.polyval(poly_coeffs[1], node[1])
        x = x_clean + rng.normal(loc=0.0, scale=sigma, size=n)
        
        y_clean = np.polyval(poly_coeffs[2], node[2])+np.polyval(poly_coeffs[4], x)
        y = y_clean + rng.normal(loc=0.0, scale=sigma, size=n)
    elif Z==2: # X<-Y
        y_clean = np.polyval(poly_coeffs[2], node[2])
        y = y_clean + rng.normal(loc=0.0, scale=sigma, size=n)

        x_clean = np.polyval(poly_coeffs[0], node[0])+np.polyval(poly_coeffs[1], node[1])+np.polyval(poly_coeffs[3], y)
        x = x_clean + rng.normal(loc=0.0, scale=sigma, size=n)
    elif Z==3: # X<-Z->Y
        x_clean = np.polyval(poly_coeffs[0], node[0])+np.polyval(poly_coeffs[1], node[1])+np.polyval(poly_coeffs[3], node[3])
        x = x_clean + rng.normal(loc=0.0, scale=sigma, size=n)

        y_clean = np.polyval(poly_coeffs[2], node[2])+np.polyval(poly_coeffs[4], node[3])
        y = y_clean + rng.normal(loc=0.0, scale=sigma, size=n)
    elif Z==4: # not dependet
        x_clean = np.polyval(poly_coeffs[0], node[0])+np.polyval(poly_coeffs[1], node[1])
        x = x_clean + rng.normal(loc=0.0, scale=sigma, size=n)

        y_clean = np.polyval(poly_coeffs[2], node[2])
        y = y_clean + rng.normal(loc=0.0, scale=sigma, size=n)

    df = pd.DataFrame({"x": x, "y": y, "x_pa1":node[0],"x_pa2":node[1],"y_pa1":node[2]})
    return df


if __name__ == '__main__':
    n = 1000 # size of each data set
    sets = 20 # number of data sets
    Z_random = np.random.choice([1,2,3,4],size=sets,p=[0.25,0.25,0.25,0.25]) # 1 if X->Y, 2 if X<-Y, 3 if X<-Z->Y, 4 if not dependent
    print(f"Z_random: {Z_random}")

    dir_name = 'pairm_size'+str(n)+'sigma02poly1'

    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory'{dir_name}' already exist.")

    for i in range(sets):
        data=gen_data(dir_name,n,Z_random[i])
        data.to_csv(dir_name+'/'+str(i)+'.csv',index=False)
        '''
        plt.figure(figsize=(6, 4))
        plt.scatter(data["x"], data["y"], color="steelblue", alpha=0.7)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
        '''

    pd.DataFrame(Z_random).to_csv(dir_name+'/truth.csv',index=False)
