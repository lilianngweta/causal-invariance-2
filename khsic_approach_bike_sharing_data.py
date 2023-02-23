
# alpha_sk = 0.5 # for creating skewed data used to learn R
# eta = 1.0#0.99
# batch_size = 128
# ns = 1 #specify number of style features
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
from torch.autograd import Variable

from hsic_calculator import HSIC, normalized_HSIC
import pandas as pd
from pandas import read_csv
from sklearn.metrics import mean_squared_error





# # Function for binarizing labels
# def binarize(y):    
#     y = np.copy(y) > 5
#     return y.astype(int)

# # Function for creating spurious correlations on 
# def create_spurious_corr(z, z_t, y_og, spu_corr= 0.1, binarize_label=True):
#     y_bin = binarize(y_og)
#     mod_labels = np.logical_xor(y_bin, np.random.binomial(1, spu_corr, size=len(y_bin)))
    
#     modified_images = z_t[mod_labels]
#     unmodified_images = z[~mod_labels]
#     all_z = np.concatenate((modified_images, unmodified_images), axis=0)
#     style_labels = np.concatenate((np.zeros(len(modified_images)), np.ones(len(unmodified_images))), axis=None)
    
#     all_img_labels = None
    
#     if binarize_label:
#         modified_imgs_labels = y_bin[mod_labels]
#         unmodified_imgs_labels = y_bin[~mod_labels]
#         all_img_labels = np.concatenate((modified_imgs_labels, unmodified_imgs_labels), axis=None)
#     else:
#         modified_imgs_labels = y_og[mod_labels]
#         unmodified_imgs_labels = y_og[~mod_labels]
#         all_img_labels = np.concatenate((modified_imgs_labels, unmodified_imgs_labels), axis=None)    
        
#     return all_z, all_img_labels, style_labels.astype(int)


# # Read Data

# z_train_og = load('./data/Z_train_og_cifar10_resnet.npy')
# z_train_t = load('./data/Z_train_rotated_cifar10_resnet.npy')

# z_test_og = load('./data/Z_test_og_cifar10_resnet.npy')
# z_test_t = load('./data/Z_test_rotated_cifar10_resnet.npy')

# y_train_og = load('./data/train_labels_cifar10.npy')

# y_test_og = load('./data/test_labels_cifar10.npy')


# # Create spurious correlations on train and test sets

# z_train_sk, train_labels_sk, t_labels_sk = create_spurious_corr(z_train_og, z_train_t, y_train_og, 
#                                          spu_corr= alpha_sk, binarize_label=False)

# z_train, train_labels, _ = create_spurious_corr(z_train_og, z_train_t, y_train_og, 
#                                          spu_corr= alpha, binarize_label=False)

# z_test_indist, indist_test_labels, _ = create_spurious_corr(z_test_og, z_test_t, y_test_og, 
#                                                          spu_corr= alpha, binarize_label=False)

# z_test_ood, ood_test_labels, _ = create_spurious_corr(z_test_og, z_test_t, y_test_og, 
#                                                          spu_corr= 1-alpha, binarize_label=False)



# # concatenate original and colored features
# z_train_og_t = np.concatenate((z_train_og, z_train_t), axis=0)
# t_train_labels = np.concatenate((np.zeros(len(z_train_og)), np.ones(len(z_train_t))), axis=None) 
# z_test_og_t = np.concatenate((z_test_og, z_test_t), axis=0)
# t_test_labels = np.concatenate((np.zeros(len(z_test_og)), np.ones(len(z_test_t))), axis=None) 


# # concatenate features with sytle labels..style labels are in column 0
# t_labels_z_train_og_t = np.concatenate((t_train_labels.reshape(-1,1), z_train_og_t), axis=1)

# # shuffle data in t_labels_z_train_og_t
# np.random.shuffle(t_labels_z_train_og_t)

# shuffled_train_og_t = t_labels_z_train_og_t[:,1:]
# shuffled_t_train_labels = t_labels_z_train_og_t[:,:1]


########## MNIST Data Distribution Per Domain ############


# # class distribution in original and colored images - class distribution is skewed
# style_and_img_labels_z_train_sk_df = pd.DataFrame(np.concatenate((t_labels_sk.reshape(-1,1),
#                                                                   train_labels_sk.reshape(-1,1),z_train_sk), axis=1))

# print("class distribution - column 1 is class labels - column 0 is domain/environment labels")
# class_distribution_per_domain = style_and_img_labels_z_train_sk_df.groupby([1,0]).count().iloc[:,0:1]
# print(class_distribution_per_domain)


# # shuffle data in style_and_img_labels_z_train_sk_df
# style_and_img_labels_z_train_sk = style_and_img_labels_z_train_sk_df.to_numpy()
# np.random.shuffle(style_and_img_labels_z_train_sk)

# shuffled_train_og_t = style_and_img_labels_z_train_sk[:,2:]
# shuffled_t_train_labels = style_and_img_labels_z_train_sk[:,:1]



# Find rotation matrix R by optimization----using KHSIC loss

# # Define Objective function 
# def obj(z, e, W, n_s=1):
#     print("inside obj!!!!!!!")
#     z = torch.from_numpy(z).float()
#     e = torch.from_numpy(e).float()
#     MI_content_style = HSIC(torch.matmul(z, W[:,:n_s]), torch.matmul(z, W[:,n_s:]))
#     print("printing MI_content_style")
#     print(MI_content_style)
#     MI_conten_env = HSIC(torch.matmul(z,W[:,n_s:]), e)
#     print("printing MI_conten_env")
#     print(MI_conten_env)
#     MI_style_env = HSIC(torch.matmul(z,W[:,:n_s]), e)
#     print("printing MI_style_env")
#     print(MI_style_env)
#     loss = (MI_content_style + MI_conten_env) - MI_style_env
#     return loss





# def obj(z, e, W, n_s=1):
#     z = torch.from_numpy(z).float()
#     e = torch.from_numpy(e).float()
#     MI_content_style = normalized_HSIC(torch.matmul(z, W[:,:n_s]), torch.matmul(z, W[:,n_s:]))
#     MI_conten_env = normalized_HSIC(torch.matmul(z,W[:,n_s:]), e)
#     MI_style_env = normalized_HSIC(torch.matmul(z,W[:,:n_s]), e)
#     loss = (MI_content_style + MI_conten_env) - MI_style_env
#     return loss



def obj(z, e, W1, W2):
    z = torch.from_numpy(z).float()
    e = torch.from_numpy(e).float()
    MI_content_style = normalized_HSIC(torch.matmul(z, W1), torch.matmul(z, W2))
    MI_conten_env = normalized_HSIC(torch.matmul(z, W2), e)
    MI_style_env = normalized_HSIC(torch.matmul(z,W1), e)
    loss = (MI_content_style + MI_conten_env) - MI_style_env
    return loss

def get_exp_results(seed=0, year='year1', season='season1', eta=1.0, batch_size=128, epochs=100, learning_rate=0.01):
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


    # Initialize R
    # Obtain prediction coefficients of domain labels
    lr_model_d = LogisticRegression(random_state=0).fit(x_train, domain_labels)
    d_coefficients1 = lr_model_d.coef_[0].reshape(-1,1)
    d_coefficients2 = lr_model_d.coef_[1].reshape(-1,1)
    d_coefficients3 = lr_model_d.coef_[2].reshape(-1,1)
    print("lr_model_d.coef_ shape!!!!!")
    print(lr_model_d.coef_.shape)
    
    ns = lr_model_d.coef_.shape[0]
    C1 = torch.from_numpy(d_coefficients1 / np.linalg.norm(d_coefficients1)).float()
    C2 = torch.from_numpy(d_coefficients2 / np.linalg.norm(d_coefficients2)).float()
    C3 = torch.from_numpy(d_coefficients3 / np.linalg.norm(d_coefficients3)).float()
#     print("Printing x_train shape!!!!!!!")
#     print(x_train.shape)
#     print(type(domain_labels[0]))
#     print(np.unique(domain_labels))
    
#     R1 = d_coefficients / np.linalg.norm(d_coefficients)
    
    print("Printing C3 shape!!!!!!!")
    print(C3.shape)
#     print(R3)
    
    
#     R1 = Variable(C1, requires_grad=False).double()
#     R2 = Variable(C2, requires_grad=False).double()
#     R3 = Variable(C3, requires_grad=False).double()
    
    R1 = torch.cat((C1,C2,C3), 1)
    
    R2 = mnn.Parameter(manifold=mnn.Stiefel(d,k)).float()
    print("Printing R1 shape!!!!!!!")
    print(R1.shape)
#     R = torch.cat((R1, R2, R3, R4), 1)

#     print("Initial R")
#     print(R)


    # Optimize - passing data in mini-batches
    
#     optimizer = moptim.rAdagrad(params = [R], lr=learning_rate)
    optimizer = moptim.rAdagrad(params = [R2], lr=learning_rate)
    
    saved_R = None
    best_loss = 1e5
    checkpoint = {}
    for epoch in range(epochs):
#         print("Inside first loop!!!!!!!!!!!!!!!!!!")
        for index in range(0, len(x_train), batch_size):
#             print("Inside the second loop!!!!!!!!!!!!!!!!!!")
            train_data_subset = x_train[index:index+batch_size]
            style_labels_subset = domain_labels[index:index+batch_size]
            loss = obj(train_data_subset, style_labels_subset, R1, R2) 
#             print("printing loss")
#             print(loss)
            # saving R with the smallest loss value so far
            if loss < best_loss:
                best_loss = loss
                print("Saving R, at epoch ", epoch)
                R = torch.cat((R1, R2), 1)
                saved_R = R
                checkpoint = {'epoch': epoch, 'loss': loss, 'R': R}
                torch.save(checkpoint, 'checkpoint') 
                print("loss: ", loss)            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

#     checkpoint

#     print("R after optimization")
#     print(R)
    # (R.T)@R

    # Load saved R
    #R_mat = torch.load('checkpoint')['R']
    R_mat = saved_R
    # Obtain post-processed features
    f_train = x_train @ R_mat.detach().numpy()  
    # f_train = z_train @ R_mat.detach().numpy()
    # f_test_indist = z_test_indist @ R_mat.detach().numpy()
    # f_test_ood = z_test_ood @ R_mat.detach().numpy()
    f_test = x_test @ R_mat.detach().numpy()
#     f_test_t = z_test_t @ R_mat.detach().numpy()
    # f_test_og_t = z_test_og_t @ R_mat.detach().numpy()



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
#     new_HSIC_ood_accuracy = lr_model_new_HSIC_no_sp.score(f_test_t[:,1:], y_test_og)

    # put all the results in a dictionary
    results_log = {}
    results_log['seed'] = seed
    results_log['year'] = year
    results_log['season'] = season
    
    results_log['eta'] = eta
    results_log['batch_size'] = batch_size
    results_log['epochs'] = epochs
    results_log['learning_rate'] = learning_rate
    
#     results_log['baseline_indist_accuracy'] = baseline_indist_accuracy
#     results_log['new_HSIC_indist_accuracy'] = new_HSIC_indist_accuracy
    results_log['baseline_ood_mse'] = baseline_ood_mse            
    results_log['new_HSIC_ood_mse'] = new_HSIC_ood_mse  

    return results_log

if __name__ == "__main__":
    ITERS = range(20)
    years = ['year1', 'year2'] 
    seasons = ['season1', 'season2', 'season3', 'season4'] 
    
    etas = [0.85,1.0]#[0.9,0.95, 0.99, 1.0]#0.99
    batch_sizes = [128]#[64,128,256,512]
    # n_styles = [3] #specify number of style features
    epoch_sizes = [100]
    learning_rates = [0.01, 0.001,0.0001]
    
    #transf_types = ['girl_sketch_val', 'dog_sketch_val', 'picasso_original_val', 'picasso_dog_val']  
    #alphas = [0.5,0.75,0.90,0.95,0.99,1.0] 
    #lamdas= [1,10]#[1,10,50,1000]
    #etas = [0.95]#[0.99]#[0.90,0.93,0.95,0.98,1.0]

    #grid = list(product(datasets, extractors, transf_types, alphas, lamdas,etas,ITERS))
    
    #grid = list(product(datasets, extractors, transf_types, lamdas,etas))
    grid = list(product(learning_rates, epoch_sizes, batch_sizes, etas, seasons, years,ITERS))

    i = int(float(sys.argv[1]))
    #dataset, extractor, transf_type, alpha, lamda, eta, ITER = grid[i]
    learning_rate, epoch_size, batch_size, eta, season, year,ITER = grid[i]

    #results_log = get_exp_results(alpha = alpha, seed=int(ITER), lamda=lamda, extractor=extractor, 
    #                             transf_type=transf_type, dataset=dataset, eta=eta)
    
    results_log = get_exp_results(seed=int(ITER), year=year, season=season, eta=eta, batch_size=batch_size, 
                                  epochs=epoch_size, learning_rate=learning_rate)
    with open(f'summary_bike/summary_{i}.json', 'w') as fp:
        json.dump(results_log, fp)
        
        
