from sklearn.metrics import accuracy_score
import logging
import sys

# read the outputs and create a dataframe
import os 
import pickle 
import pandas as pd 
import numpy as np 

from collections import defaultdict 
import copy 

def load_pkl_file(fpath):
    with open(fpath, 'rb') as handle:
        o = pickle.load(handle)
    return o 

def get_all_outs_for_exp(root_pfx):

    lst_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(root_pfx) for f in fn]
    lst_out_files = [f  for f in lst_files if f[-3:]=='pkl'] 
    lst_outs = []
    for fpath in lst_out_files:
        print(fpath)
        out = load_pkl_file(fpath)
        out_ = {} 
        params =fpath[len(root_pfx)+1:]
        params = dict([x.split('__') for x in params.split('/')[:-1] ])

        if(params['method']=='tbal'):
            out_['sel_auto_labeled_acc'] = out['counts']['auto_labeled_acc']
            out_['sel_coverage'] = out['counts']['coverage_1']
            out_['all_auto_labeled_acc'] = out['counts']['auto_labeled_acc']
            out_['all_coverage'] = out['counts']['coverage_1']
        else:
            out_['sel_auto_labeled_acc'] = out['sel_counts']['auto_labeled_acc']
            out_['sel_coverage'] = out['sel_counts']['coverage_1']
            out_['all_auto_labeled_acc'] = out['all_counts']['auto_labeled_acc']
            out_['all_coverage'] = out['all_counts']['coverage_1']

        out_.update(params)

        if(out_['sel_auto_labeled_acc']!=None):
            lst_outs.append(out_)
    return lst_outs 

def filter_outputs(lst_outs,param_f):
    filtered_outs = []
    for out in lst_outs:
        flag = True 
        for k in param_f.keys():
            flag = flag and (out[k]==str(param_f[k]))
        if(flag):
            filtered_outs.append(out)
    return filtered_outs

def filter_outputs_2(df,param_f):
    query = ' & '.join([ str(param)+ '==' + "'"+str(param_f[param])+"'" for param in param_f.keys()])
    return df.query(query)

def get_numbers_for_param(lst_outs,base_params,param, param_vals):
    out = defaultdict(list)

    for n in param_vals:
        
        #print(n)
        params = copy.deepcopy(base_params)

        params[param] = n
        df_1 = pd.DataFrame(lst_outs)
        
        params['method'] = 'active_learning'
        #filterd_outs = filter_outputs(lst_outs,params)
        df = filter_outputs_2(df_1,params)
        #df = pd.DataFrame(filterd_outs)
        #print(df['sel_auto_labeled_acc'].mean())
        #print(df['sel_coverage'].mean())
        out['max_num_train_pts'].append(n)

        out['AL_all_err_mean'].append(1- df['all_auto_labeled_acc'].mean())
        out['AL_all_err_std'].append(df['all_auto_labeled_acc'].std())

        out['AL_all_cov_mean'].append(df['all_coverage'].mean())
        out['AL_all_cov_std'].append(df['all_coverage'].std())

        out['AL_sel_err_mean'].append(1- df['sel_auto_labeled_acc'].mean())
        out['AL_sel_err_std'].append(df['sel_auto_labeled_acc'].std())

        out['AL_sel_cov_mean'].append(df['sel_coverage'].mean())
        out['AL_sel_cov_std'].append(df['sel_coverage'].std())


        params['method'] = 'passive_learning'
        #filterd_outs = filter_outputs(lst_outs,params)
        #df = pd.DataFrame(filterd_outs)
        df = filter_outputs_2(df_1,params)

        #print(df['sel_auto_labeled_acc'].mean())
        #print(df['sel_coverage'].mean())
        out['PL_all_err_mean'].append(1- df['all_auto_labeled_acc'].mean())
        out['PL_all_err_std'].append(df['all_auto_labeled_acc'].std())

        out['PL_all_cov_mean'].append(df['all_coverage'].mean())
        out['PL_all_cov_std'].append(df['all_coverage'].std())

        out['PL_sel_err_mean'].append(1- df['sel_auto_labeled_acc'].mean())
        out['PL_sel_err_std'].append(df['sel_auto_labeled_acc'].std())
        out['PL_sel_cov_mean'].append(df['sel_coverage'].mean())
        out['PL_sel_cov_std'].append(df['sel_coverage'].std())

        params['method'] = 'tbal'
        #filterd_outs = filter_outputs(lst_outs,params)

        #df = pd.DataFrame(filterd_outs)
        df = filter_outputs_2(df_1,params)
        #print(df['sel_auto_labeled_acc'].mean())
        #print(df['sel_coverage'].mean())
        out['ALBL_sel_err_mean'].append(1- df['sel_auto_labeled_acc'].mean())
        out['ALBL_sel_err_std'].append(df['sel_auto_labeled_acc'].std())
        out['ALBL_sel_cov_mean'].append(df['sel_coverage'].mean())
        out['ALBL_sel_cov_std'].append(df['sel_coverage'].std())

    for k in out.keys():
        out[k] = np.array(out[k])

    return out 

