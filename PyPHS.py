#!/usr/bin/env python
#
# Fast approximation on GWAS Cox regressions
# Version 0.1.0
#

################################################################
#
# Dependencies
#
################################################################ 

# Py packages
import numpy as np
from numba import jit
import pandas as pd
import dask as dk
from dask import delayed
import dask.array as da
import argparse, datetime, gzip, os, errno
from pandas_plink import read_plink
from sklearn import preprocessing 
from lifelines import CoxPHFitter

# Adding threads
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

# Dealing with IO
import h5py

# Debug purpose
import logging, sys, warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Loading functional modules
from PHS_modules import *

__version__ = '0.1.0'
MASTHEAD = "\n\n*********************************************************************\n"
MASTHEAD += "*\n"
MASTHEAD += "* GWAS Cox regressions \n"
MASTHEAD += "* Version {V}\n".format(V=__version__)
MASTHEAD += "* (C) 2019 \n"
MASTHEAD += "* Chun Chieh Fan\n"
MASTHEAD += "*\n"
MASTHEAD += "*********************************************************************\n\n"


# Command line arguements
parser = argparse.ArgumentParser()
parser.add_argument("--chunk", dest='chunk', default=5000,
  help='Chunk size, by default 5000 SNPs a time')
parser.add_argument("--out", dest='outpath', 
  default=None,
  help='Output prefix for the algorithm')
parser.add_argument("--pheno", dest='phenof',
  default=None,
  help='Phenotype spreadsheet for the analysis')
parser.add_argument("--keep", dest='subj_list',
  default=None,
  help='Subject list to subset if needed') 
parser.add_argument("--covar-name", dest='covname',
  default=None,
  help='Covariate names in the phenotype spreadsheet for the analysis')
parser.add_argument("--T", dest='Tname',
  default=None,
  help='Variable for time-to-event in the phenotype spreadsheet for the analysis')
parser.add_argument("--event", dest='ename',
  default=None,
  help='Variable for event in the phenotype spreadsheet for the analysis')
parser.add_argument("--thread", dest='pl_flag', default=4,
  help='Number of threads to process arrays')
parser.add_argument("--bfile", dest='geno_prefix', default=None,
  help='Prefix for genotype data in Plink binary format')


# Parsing arguments
args = parser.parse_args()
if args.outpath is None:
  raise ValueError('Must provide the output destination')
if args.phenof is None:
  raise ValueError('Must provide phenotype spreadsheet')
if args.geno_prefix is None:
  raise ValueError('Must provide genotype files in plink format')

# Path for incoming data
outpath = args.outpath
pheno_path = args.phenof
geno_prefix = args.geno_prefix
covname = args.covname.split(',')
event_name = args.ename
T_name = args.Tname

# Chunk size, memory related
chunk_size = int(args.chunk)

# Thread parameters
pl_flag = int(args.pl_flag)
pool = ThreadPool(pl_flag)
dk.config.set(pool=pool)
nCores = cpu_count()

###### Initiate Logger
try:
    os.remove(outpath + '.log')
except OSError:
    pass
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(outpath + '.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

###### Assemble the input info
cline = "PyPHS \n--bfile " + args.geno_prefix + "\n" + "--chunk " + str(args.chunk) + "\n" + "--keep " + args.subj_list + "\n" + "--pheno " + args.phenof + "\n" + "--T " + args.Tname + "\n" + "--event " + args.ename + "\n" + "--covar-name " + args.covname + "\n" + "--thread " + str(args.pl_flag) + "\n--out " + args.outpath + "\n"



################################################################
#
# Analytic pipeline
#
################################################################
logger.info(MASTHEAD)
logger.info(cline)
# Get phenotype spreadsheet first
# The phenotype spreadsheet has been curated, see note in
# UKB processing CAD
logger.info('Reading phenotype file - ' + pheno_path)
pheno = pd.read_csv(pheno_path)
pheno['IID'] = pheno['IID'].apply(str)


###############################################
# Need to check this part
# Augment the data to randomly perturb T
# The purpose is for efficient approximation
# pheno[T_name] = np.array(pheno[T_name]) + np.random.rand(len(pheno[T_name]))/100
################################################

pheno = pheno[['IID', T_name, event_name] + covname].dropna().reset_index(drop=True)
logger.info(str(pheno.shape[0]) + ' subjects found with non-missing phenotypes\n')

# Parse additional filters
if args.subj_list is not None:
  logger.info('Extracting subjects from ' + args.subj_list)
  subjlist = pd.read_csv(args.subj_list,names=list({'IID'}))
  subjlist['IID'] = subjlist['IID'].apply(str)
  c, ia, ib = intersect_mtlb(subjlist['IID'],pheno['IID'])  
  pheno = pheno.iloc[ib]

pheno = pheno.reset_index(drop=True)
logger.info(str(pheno.shape[0]) + ' subjects remains after keep\n')

# Run the block correlation fit
# imp = preprocessing.Imputer(strategy='mean', axis=1)
logger.info('Processing genotype data: ' + geno_prefix)
(bim, fam, geno) = read_plink(geno_prefix)

# intersect data
c, ia, ib = intersect_mtlb(fam['iid'],pheno['IID'])
logger.info(str(len(ia)) + ' subjects found to have genotype data\n')

# Final sample assignment
pheno = pheno.iloc[ib]
pheno = pheno.reset_index(drop=True)
geno_ia = geno[:,ia]

# Function for null model
logger.info('Generating Null models\n')
cph = CoxPHFitter()
cph.fit(pheno[[T_name, event_name] + covname], T_name, event_col=event_name)
# res_surv = cph.compute_residuals(pheno[[T_name, event_name] + covname], 'deviance').sort_index()['deviance']
res_surv = cph.compute_residuals(pheno[[T_name, event_name] + covname], 'martingale').sort_index()['martingale']

# This is the most memory intensive part. Might need to change if we are dealing with biobank scale data

logger.info('Calculating Null covariance matrix\n')
mat = cph.predict_cumulative_hazard(pheno)
P = np.diff(mat,axis=-0)
for isubj in range(P.shape[1]):
  idx = np.abs(mat.index - pheno[T_name][isubj]).argmin()
  P[idx::,isubj] = 0
V = np.diag(pheno[event_name] - res_surv) - np.dot(P.transpose(),P)
X = pheno[covname]
C = V - np.matmul(np.matmul(np.matmul(V,X), np.linalg.inv(np.matmul(np.matmul(X.transpose(), V),X))), np.matmul(X.transpose(), V))



# auto chunk to reduce the query time
chunk_array = [bim.i.values[i:i + chunk_size] for i in xrange(0, len(bim.i.values), chunk_size)]  
nchunk = len(chunk_array)
chunk_ind = 1

#################################### Create HDF temporary file #########
# The idea is to use I/O to prevent memory overload
# Initialize the temporary files
tmp_f = outpath + '.tmp.hdf5'
try:
    os.remove(tmp_f)
except OSError:
    pass
logger.info('Initiating temporary file - ' + tmp_f)
f = h5py.File(tmp_f)
betavec = f.create_dataset('betavec',data=np.zeros([bim.shape[0],1]),chunks=(bim.shape[0]/nchunk, 1))
zvec = f.create_dataset('zvec',data=np.zeros([bim.shape[0],1]),chunks=(bim.shape[0]/nchunk, 1))
meanvec = f.create_dataset('meanvec',data=np.zeros([bim.shape[0],1]),chunks=(bim.shape[0]/nchunk, 1))
hetvec = f.create_dataset('hetvec',data=np.zeros([bim.shape[0],1]),chunks=(bim.shape[0]/nchunk, 1))

#####################################Calculation by Chunk ##############
#
# Begin chunk processing
for chunk1 in chunk_array:
  print('Processing Chunk - ' + str(chunk_ind) + ' of ' + str(nchunk))
  # Compute 
  geno_chunk = geno_ia[chunk1,:]
  g = da.apply_along_axis(SimpleImpute, 1, geno_chunk)
  with ProgressBar():
    results = da.compute(g, get=get)
  gtmp = np.stack(results[0][:,0])
  vtmp = np.diagonal(np.matmul(gtmp, np.matmul(gtmp, C).transpose()))
  beta_tmp = np.matmul(gtmp, res_surv)
  betavec[chunk1, 0] = beta_tmp
  zvec[chunk1, 0] = beta_tmp/np.sqrt(vtmp)
  meanvec[chunk1, 0] = np.stack(results[0][:,1])
  hetvec[chunk1, 0] = np.stack(results[0][:,2])
  chunk_ind += 1

##########################################################################
# Write into csv for later analysis

bim['allele_frq'] = np.array(meanvec)/2
bim['allele_std'] = np.array(hetvec)
bim['beta_surv'] = np.array(betavec)/ia.shape[0]
bim['z'] = np.array(zvec)

logger.info('Writing results to ' + outpath + '.beta.txt')
bim.to_csv(outpath + '.beta.txt', index=None, sep='\t')

# Initialize the temporary files
tmp_f = outpath + '.tmp.hdf5'
try:
    os.remove(tmp_f)
except OSError:
    pass

logger.info('\n Done! \n')
