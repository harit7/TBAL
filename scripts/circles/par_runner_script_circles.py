import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

from multiprocessing import Process
from omegaconf import OmegaConf
from core.run_lib import *

root_dir = '../../'
conf_dir = f'{root_dir}configs/neurips23/circles/'

base_conf_file = '{}/circles_base_conf.yaml'.format(conf_dir)
root_pfx = 'circle_runs'
root_pfx = f'{root_dir}/outputs/{root_pfx}/'


if __name__ == "__main__":
    
    lst_seed_frac= [0.2]
    lst_query_batch_frac = [0.05]

    lst_methods = ['tbal','active_learning','passive_learning']

    # compute
    lst_devices = ['cpu']

    run_batch_size = 25
    #T = 25 # number of experiment we want to try
    T = 10
    overwrite_flag = False 
    lst_C1       = [0, 0.25]
    #lst_C1       = [0]
    lst_eps      = [0.01]
    
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

    
    run_val = False
    run_nq  = True 
    run_eps = False

    base_conf = OmegaConf.load(base_conf_file)
    base_conf['root_pfx']    = root_pfx

    lst_confs = []
    if(run_nq):
        
        lst_n_q      = [100, 200, 300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
        lst_n_val    = [2000] 
        params['max_num_train_pts'] = lst_n_q
        params['max_num_val_pts'] =lst_n_val
        lst          = create_confs(base_conf,params)
        lst_confs.extend(lst)

    if(run_val):
        lst_n_q      = [500]
        lst_n_val    = [100,400,800,1200,1600,2000,2400,2800,3200,3600,4000] 
        params['max_num_train_pts'] = lst_n_q
        params['max_num_val_pts'] =lst_n_val

        lst = create_confs(base_conf,params)
        lst_confs.extend(lst)

    if(run_eps):
        lst_n_q      = [500]
        lst_n_val    = [4000] 
        params['max_num_train_pts'] = lst_n_q
        params['max_num_val_pts'] =lst_n_val

        lst = create_confs(base_conf,params)
        lst_confs.extend(lst)
    
    print(f'Total Confs to run {len(lst_confs)}')

    lst_confs = list(sorted(lst_confs, key=lambda x: x['method']))

    batched_par_run(lst_confs,batch_size=run_batch_size,lst_devices=lst_devices,overwrite=overwrite_flag)
        
