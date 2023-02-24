# alpha_sk = 0.5 # for creating skewed data used to learn R
# eta = 1.0#0.99
# batch_size = 128
# # ns = 1 #specify number of style features
# epochs = 100


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import torch
from numpy import load
import sys, json
from itertools import product
from sklearn import preprocessing

import torch
import mctorch.nn as mnn
import mctorch.optim as moptim
from hsic_calculator import HSIC, normalized_HSIC
import pandas as pd
from pandas import read_csv
from sklearn.metrics import mean_squared_error



def obj(z, e, W, n_s=1):
    z = torch.from_numpy(z).float()
    e = torch.from_numpy(e).float()
    MI_content_style = normalized_HSIC(torch.matmul(z, W[:,:n_s]), torch.matmul(z, W[:,n_s:]))
    MI_conten_env = normalized_HSIC(torch.matmul(z,W[:,n_s:]), e)
    MI_style_env = normalized_HSIC(torch.matmul(z,W[:,:n_s]), e)
    loss = (MI_content_style + MI_conten_env) - MI_style_env
    return loss

def get_exp_results(seed=0, year='year1', season='season1', eta=1.0, batch_size=128, epochs=100, learning_rate=0.01, ns=1):
    np.random.seed(seed)
    # Read Data
    train = read_csv('./data/'+year+'_'+season+'_train.csv')
    train = train.loc[np.random.choice(train.index, size=int(0.8*len(train)))].reset_index(drop=True)
    x_train = train.iloc[:, 2:].to_numpy() 
    y_train = np.array(train["cnt"])
    domain_labels = np.array(train["season"])
    
    test = read_csv('./data/'+year+'_'+season+'_test.csv')
    test = test.loc[np.random.choice(test.index, size=int(0.8*len(test)))].reset_index(drop=True)
    x_test = test.iloc[:, 2:].to_numpy()
    y_test = np.array(test["cnt"])
    
    # Standardize the training data.
    sscaler = preprocessing.StandardScaler()
    sscaler.fit(x_train)
    x_train = sscaler.transform(x_train)
    
    # Standardize the test data.
    sscaler = preprocessing.StandardScaler()
    sscaler.fit(x_test)
    x_test = sscaler.transform(x_test)
 
    dtype = torch.FloatTensor
    n = x_train.shape[0]
    d = x_train.shape[1]
    k = int(x_train.shape[1]*eta) # % of original number of features


#    # Initialize R
#    # Obtain prediction coefficients of domain labels
#     lr_model_d = LogisticRegression(random_state=0).fit(x_train, domain_labels)
#     d_coefficients = lr_model_d.coef_.reshape(-1,1)
#     R1 = torch.from_numpy(d_coefficients / np.linalg.norm(d_coefficients))
#     print("Printing R1 shape!!!!!!!")
#     print(R1.shape)
    
    R = mnn.Parameter(manifold=mnn.Stiefel(d,k-1)).float()
#     R = torch.cat((R1, R2), 1)

    print("Initial R")
    print(R)


    # Optimize - passing data in mini-batches
    
    optimizer = moptim.rAdagrad(params = [R], lr=learning_rate)
    
    saved_R = None
    best_loss = 1e5
    checkpoint = {}
    for epoch in range(epochs):
#         print("Inside first loop!!!!!!!!!!!!!!!!!!")
        for index in range(0, len(x_train), batch_size):
#             print("Inside the second loop!!!!!!!!!!!!!!!!!!")
            train_data_subset = x_train[index:index+batch_size]
            style_labels_subset = domain_labels[index:index+batch_size]
            loss = obj(train_data_subset, style_labels_subset, R, ns) 
#             print("printing loss")
#             print(loss)
            # saving R with the smallest loss value so far
            if loss < best_loss:
                best_loss = loss
                print("Saving R, at epoch ", epoch)
                saved_R = R
                checkpoint = {'epoch': epoch, 'loss': loss, 'R': R}
                torch.save(checkpoint, 'checkpoint') 
                print("loss: ", loss)            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

#     checkpoint

    print("R after optimization")
    print(R)
    # (R.T)@R

    # Load saved R
    #R_mat = torch.load('checkpoint')['R']
    R_mat = saved_R
    
    # Obtain post-processed features
    f_train = x_train @ R_mat.detach().numpy()  
    f_test = x_test @ R_mat.detach().numpy()
    
 
    # concatenate domain labels with f_train and x_train
    t_labels_f_train = np.concatenate((domain_labels.reshape(-1,1), f_train), axis=1)
    t_labels_x_train = np.concatenate((domain_labels.reshape(-1,1), x_train), axis=1)
    corr_f_matrix = np.corrcoef(t_labels_f_train.T)
    corr_x_matrix = np.corrcoef(t_labels_x_train.T)
    corr_special = np.abs(corr_f_matrix[0,1])
    corr_ns_f_norm = np.sqrt((corr_f_matrix[0,2:]**2).mean()) 
    x_corr_ns_f_norm = np.sqrt((corr_x_matrix[0,:]**2).mean()) 


    ########### Baseline results ###############
    # Trained on original baseline features, tested on colored features - no spurious correlations 
    print("number of original features")
    print(x_train.shape)
    logistic_regression_on_baseline_og = LinearRegression().fit(x_train,y_train)                                     
    # baseline_ood_accuracy = logistic_regression_on_baseline_og.score(x_test, y_test)
    y_pred = logistic_regression_on_baseline_og.predict(x_test)
    baseline_ood_mse = mean_squared_error(y_test, y_pred)
    print("baseline_ood_mse")
    print(baseline_ood_mse)   
#     baseline_ood_accuracy = logistic_regression_on_baseline_og.score(z_test_t, y_test_og)
    ####################################


    # trained on original post-processed features, tested on transformed post-processed 
    # features without style features - no spurious correlations  
    print("number of post processed features")
    print(f_train[:,1:].shape)
    lr_model_new_HSIC_no_sp = LinearRegression().fit(f_train[:,1:],y_train)
    y_pred_h = lr_model_new_HSIC_no_sp.predict(f_test[:,1:])
    #y_pred_h = lr_model_new_HSIC_no_sp.predict(f_test[:,1:])
    new_HSIC_ood_mse = mean_squared_error(y_test, y_pred_h)
    print("new_HSIC_ood_mse")
    print(new_HSIC_ood_mse)
    
    print("corr_f_matrix")
    print(corr_f_matrix)
    print("corr_x_matrix")
    print(corr_x_matrix)
    
    
#     results_log['corr_f_matrix'] = corr_f_matrix
#     results_log['corr_x_matrix'] = corr_x_matrix
    
    
#     new_HSIC_ood_accuracy = lr_model_new_HSIC_no_sp.score(f_test_t[:,1:], y_test_og)


#     corr_special = np.abs(corr_matrix[0,1])
#     corr_ns_f_norm = np.sqrt((corr_matrix[0,5:]**2).mean()) 
#     x_corr_ns_f_norm = np.sqrt((corr_x_matrix[0,:]**2).mean()) 
#     corr_matrix = np.corrcoef(t_labels_f_train.T)
#     corr_x_matrix = np.corrcoef(t_labels_x_train.T)

    # put all the results in a dictionary
    results_log = {}
    results_log['seed'] = seed
    results_log['year'] = year
    results_log['season'] = season
    results_log['eta'] = eta
    results_log['batch_size'] = batch_size
    results_log['epochs'] = epochs
    results_log['learning_rate'] = learning_rate
#     results_log['corr_f_matrix'] = corr_f_matrix
#     results_log['corr_x_matrix'] = corr_x_matrix
    results_log['corr_special'] = corr_special
    results_log['corr_ns_f_norm'] = corr_ns_f_norm
    results_log['x_corr_ns_f_norm'] = x_corr_ns_f_norm
    
#     results_log['baseline_indist_accuracy'] = baseline_indist_accuracy
#     results_log['new_HSIC_indist_accuracy'] = new_HSIC_indist_accuracy
    results_log['baseline_ood_mse'] = baseline_ood_mse            
    results_log['new_HSIC_ood_mse'] = new_HSIC_ood_mse  

    return results_log

if __name__ == "__main__":
    ITERS = range(20)
    years = ['year1', 'year2'] 
    seasons = ['season1', 'season2', 'season3', 'season4'] 
    
    etas = [0.8,1.0]#[0.9,0.95, 0.99, 1.0]#0.99
    batch_sizes = [128]#[64,128,256,512]
    nss = [1] #specify number of style features
    epoch_sizes = [100]
    learning_rates = [0.01, 0.001,0.0001]
    
    #transf_types = ['girl_sketch_val', 'dog_sketch_val', 'picasso_original_val', 'picasso_dog_val']  
    #alphas = [0.5,0.75,0.90,0.95,0.99,1.0] 
    #lamdas= [1,10]#[1,10,50,1000]
    #etas = [0.95]#[0.99]#[0.90,0.93,0.95,0.98,1.0]

    #grid = list(product(datasets, extractors, transf_types, alphas, lamdas,etas,ITERS))
    
    #grid = list(product(datasets, extractors, transf_types, lamdas,etas))
    grid = list(product(learning_rates, epoch_sizes, nss, batch_sizes, etas, seasons, years,ITERS))

    i = int(float(sys.argv[1]))
    #dataset, extractor, transf_type, alpha, lamda, eta, ITER = grid[i]
    learning_rate, epoch_size, ns, batch_size, eta, season, year,ITER = grid[i]

    #results_log = get_exp_results(alpha = alpha, seed=int(ITER), lamda=lamda, extractor=extractor, 
    #                             transf_type=transf_type, dataset=dataset, eta=eta)
    
    results_log = get_exp_results(seed=int(ITER), year=year, season=season, eta=eta, batch_size=batch_size, 
                                  epochs=epoch_size, learning_rate=learning_rate, ns=ns)
    with open(f'summary_bike/summary_{i}.json', 'w') as fp:
        json.dump(results_log, fp)