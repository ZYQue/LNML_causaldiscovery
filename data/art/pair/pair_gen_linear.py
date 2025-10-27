import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gen_data(dir_name,n,Z):
    z_low=-5
    z_high=5
    sigma=0.2
    poly_degree=1
    coeff_range=(-10,10)
    seed=42
    
    rng = np.random.default_rng()

    poly_coeffs = rng.uniform(coeff_range[0], coeff_range[1], size=poly_degree + 1)
    #print(f"poly_coeffs: {poly_coeffs}")
    poly_coeffs2 = rng.uniform(coeff_range[0], coeff_range[1], size=poly_degree + 1)
    #print(f"poly_coeffs2: {poly_coeffs2}")
    
    z_base = rng.uniform(z_low, z_high, size=n)
    z = z_base + rng.normal(loc=0.0, scale=sigma, size=n)

    y_clean = np.polyval(poly_coeffs, z)
    y = y_clean + rng.normal(loc=0.0, scale=sigma, size=n)

    x_clean = np.polyval(poly_coeffs2, z)
    x = x_clean + rng.normal(loc=0.0, scale=sigma, size=n)

    if Z==0:
        df = pd.DataFrame({"x": z, "y": y})
    elif Z==1:
        df = pd.DataFrame({"x": x, "y": y})
    return df


if __name__ == '__main__':
    n = 1000 # size of each data set
    sets = 20 # number of data sets
    Z_random = np.random.choice([0,1],size=sets,p=[0.5,0.5]) # False if no latent confounder Z, True if Z may exists

    dir_name = 'pair_size'+str(n)+'sigma02poly1'

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
