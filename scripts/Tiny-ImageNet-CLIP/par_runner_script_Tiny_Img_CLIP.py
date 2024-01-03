import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

from multiprocessing import Process
from omegaconf import OmegaConf
from core.run_lib import *

root_dir = '../../'
conf_dir = f'{root_dir}configs/tiny-imagenet-CLIP/'

base_conf_file = '{}/tiny_imagenet_CLIP_base_conf.yaml'.format(conf_dir)
root_pfx = 'tiny_imagenet_CLIP_runs_2'
root_pfx = f'{root_dir}/outputs/{root_pfx}/'

if __name__ == "__main__":
    
    lst_seed_frac= [0.2]
    lst_query_batch_frac = [0.05]

    lst_methods = ['tbal','active_learning','passive_learning']

    # compute
    lst_devices = ['cuda:0','cuda:1']

    run_batch_size = 14
    T = 5 # number of experiment we want to try
    overwrite_flag = True 

    lst_C1       = [0, 0.25]
    #lst_eps      = [0.05,0.10]

    #lst_C1       = [0.25]
    lst_eps      = [0.10]
    
    lst_seeds = [i*7 for i in range(T)]
    params = {
            'C_1': lst_C1, 
            'max_num_train_pts': [], 
            'max_num_val_pts':[],
            'eps':lst_eps,
            'seed':lst_seeds,
            'method':lst_methods,
            'C':[2],
            'seed_frac':lst_seed_frac,
            'query_batch_frac':lst_query_batch_frac}

    #val = False 
    run_val = False
    run_nq  =  True 

    base_conf = OmegaConf.load(base_conf_file)
    base_conf['root_pfx']    = root_pfx

    lst_confs = []
    if(run_nq):
        #lst_n_q      = [1000,1500,2000,2500,3000,4000,5000,6000,7000,8000]
        #lst_n_q      = [1000,2000,3000,4000,5000,6000,7000,8000]
        lst_n_q      = [2000,4000,6000,8000,10000]
        lst_n_val    = [10000] 
        params['max_num_train_pts'] = lst_n_q
        params['max_num_val_pts'] =lst_n_val
        lst          = create_confs(base_conf,params)
        lst_confs.extend(lst)

    if(run_val):
        lst_n_q      = [6000]
        #lst_n_val    = [500,1000,2000,3000,4000,5000,6000,7000,8000,9000] 
        #lst_n_val    = [1000,2000,3000,4000,5000,6000,7000,8000,9000] 
        lst_n_val    = [2000,4000,6000,8000,10000] 
        params['max_num_train_pts'] = lst_n_q
        params['max_num_val_pts'] =lst_n_val

        lst = create_confs(base_conf,params)
        lst_confs.extend(lst)
    
    print(f'Total Confs to run {len(lst_confs)}')

    lst_confs = list(sorted(lst_confs, key=lambda x: x['method']))

    batched_par_run(lst_confs,batch_size=run_batch_size,lst_devices=lst_devices,overwrite=overwrite_flag)
        
