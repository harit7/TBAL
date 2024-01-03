import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

from multiprocessing import Process
from omegaconf import OmegaConf
from core.run_lib import *

root_dir = '../../'
conf_dir = f'{root_dir}configs/IMDB/'

base_conf_file = '{}/IMDB_base_conf.yaml'.format(conf_dir)
root_pfx = 'IMDB_runs_2'
root_pfx = f'{root_dir}/outputs/{root_pfx}/'


if __name__ == "__main__":
    
    lst_seed_frac= [0.2]
    lst_query_batch_frac = [0.05]

    lst_methods = ['tbal','active_learning','passive_learning']

    # compute
    lst_devices = ['cuda:0','cuda:1']
    #lst_devices = ['cuda:0']

    run_batch_size = 24
    T = 10 # number of experiment we want to try
     
    lst_C1       = [0, 0.25]
    lst_eps      = [0.05]
    
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
    run_val = True
    run_nq  =  False 
    overwrite_flag = False

    base_conf = OmegaConf.load(base_conf_file)
    base_conf['root_pfx']    = root_pfx

    lst_confs = []
    if(run_nq):
        #lst_n_q      = [500,1000,1500,2000,2500,3000,3500,4000,4500]
        #lst_n_q      = [100,250,500,1000,1500,2000] #,2500,3000,3500,4000,4500]
        lst_n_q = [100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000]
        lst_n_val    = [1000,2000] 
        params['max_num_train_pts'] = lst_n_q
        params['max_num_val_pts'] =lst_n_val
        lst          = create_confs(base_conf,params)
        lst_confs.extend(lst)

    if(run_val):
        lst_n_q      = [200,500]
        lst_n_val    = [100,200,300,400,500,600,700,800,900,1000] 
        params['max_num_train_pts'] = lst_n_q
        params['max_num_val_pts'] =lst_n_val

        lst = create_confs(base_conf,params)
        lst_confs.extend(lst)
    
    print(f'Total Confs to run {len(lst_confs)}')

    lst_confs = list(sorted(lst_confs, key=lambda x: x['method']))

    batched_par_run(lst_confs,batch_size=run_batch_size,lst_devices=lst_devices,overwrite=overwrite_flag)
        
