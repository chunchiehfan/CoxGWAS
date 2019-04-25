import numpy as np

########################################################################
#
# Module repositories 
#
########################################################################
def intersect_mtlb(a, b):
    '''
    This is the intersection function similar to matlab intersect
    However, the indexing is always sorted according to vector a
    '''
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    ia = ia[np.isin(a1, c)]
    ib = ib[np.isin(b1, c)]
    idx = ia.argsort()
    return c, ia[idx], ib[idx] 

def progressBar(value, endvalue, bar_length=20):
    '''
    Self-refreshing progress bar function
    '''
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def SimpleImpute(x):
    '''
    genotype matrix in X
    i as the slicing index
    '''
    me = np.nanmean(x)
    het = np.nanstd(x)
    x_s = (x - me)/het
    x_sd = np.nan_to_num(x_s)
    return x_sd, me, het 

def res_multiply(x, res_score):
  beta = np.dot(x, res_score)
  return beta

def sandwitch_multiply(x, C):
  beta_var = np.matmul(np.matmul(x.transpose(), C), x)
  return beta_var

def multiply_1d(x, C):
  beta_var = np.matmul(C, x)
  return beta_var

def proc_SNP_1d(x,res_score):
    '''
    This is the function wrapper for single SNP process
    Adapt for dask and numba
    '''        
    x_sd, me, het = SimpleImpute(x)
    beta = res_multiply(x_sd, res_score)
    return beta        

def proc_SNP_2d(x,res_score1,res_score2):
    '''
    This is the function wrapper for single SNP process
    Adapt for dask and numba
    '''
    x_sd = delayed(SimpleImpute)(x)
    beta1 = delayed(res_multiply)(x_sd, res_score1)
    beta2 = delayed(res_multiply)(x_sd, res_score2)
    return beta1, beta2

def proc_SNP_1d_var(x, res_score, C):
    '''
    This is the function wrapper for single SNP process
    Adapt for dask and numba
    '''        
    x_sd, me, het = SimpleImpute(x)
    beta = res_multiply(x_sd, res_score)
    #beta_se = sandwitch_multiply(x_sd, C)
    return beta, me, het

def proc_SNP_1d_var2(x):
    '''
    This is the function wrapper for single SNP process
    Adapt for dask and numba
    '''        
    x_sd, me, het = SimpleImpute(x)
    beta = res_multiply(x_sd, res_surv)
    beta_se = sandwitch_multiply(x_sd, C)
    return beta, beta_se, me, het


