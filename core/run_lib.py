import os
import copy
from multiprocessing import Process
import sys 
sys.path.append('../../')

from .conf_defaults import *
from core.passive_learning import *
from core.tbal import * 
from core.active_learning import * 
from core.auto_labeling import *

from datasets.dataset_utils import * 

from utils.counting_utils import *  
from utils.common_utils import * 
from utils.vis_utils import *
from utils.logging_utils import * 
from  datasets.data_manager import * 


def set_seed(seed):
    import random 
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

def run_active_learnling_auto_labeling(conf,logger):

    set_seed(conf['random_seed'])
    # get data
        
    logger.info('Loaded dataset {}'.format(conf['data_conf']['dataset']))
    
    dm = DataManager(conf,logger)
    len(dm.ds_std_train), len(dm.ds_std_val)

    act_learn = ActiveLearning(conf,dm,logger)

    out = act_learn.run_al_loop()

    auto_lbl_conf = conf['auto_lbl_conf']

    auto_lbl_conf['method_name']= 'all' 
    
    meta_df_cp = copy.deepcopy(dm.meta_df) 

    auto_labeler = AutoLabeling(conf,dm,act_learn.cur_clf,logger)
    out = auto_labeler.run()
    counts_all = dm.get_auto_labeling_counts()

    auto_lbl_conf['method_name']= 'selective' 
    
    dm.meta_df = meta_df_cp 

    auto_labeler = AutoLabeling(conf,dm,act_learn.cur_clf,logger)
    out = auto_labeler.run()
    counts_sel = dm.get_auto_labeling_counts()

    return {'all_counts':counts_all,'sel_counts':counts_sel}
    

def run_passive_labeling_auto_labeling(conf,logger):

    set_seed(conf['random_seed'])
    # get data
    
    logger.info('Loaded dataset {}'.format(conf['data_conf']['dataset']))
    
    dm = DataManager(conf,logger)
    len(dm.ds_std_train), len(dm.ds_std_val)

    pas_learn = PassiveLearning(conf,dm,logger)

    out = pas_learn.run()

    auto_lbl_conf = conf['auto_lbl_conf']

    auto_lbl_conf['method_name']= 'all' 
    
    meta_df_cp = copy.deepcopy(dm.meta_df) 

    auto_labeler = AutoLabeling(conf,dm,pas_learn.cur_clf,logger)
    out = auto_labeler.run()
    counts_all = dm.get_auto_labeling_counts()

    auto_lbl_conf['method_name']= 'selective' 
    
    dm.meta_df = meta_df_cp 

    auto_labeler = AutoLabeling(conf,dm,pas_learn.cur_clf,logger)
    out = auto_labeler.run()
    counts_sel = dm.get_auto_labeling_counts()

    return {'all_counts':counts_all,'sel_counts':counts_sel}

    #return #{'all_out':out_all,'all_counts':counts_all,'sel_out':out_sel,'sel_counts':counts_sel}

def run_tbal_conf(conf,logger,return_per_epoch_out=False):

    set_seed(conf['random_seed'])

    dm = DataManager(conf,logger)
    len(dm.ds_std_train), len(dm.ds_std_val)

    tbal = ThresholdBasedAutoLabeling(conf,dm,logger)

    tbal.init()

    lst_epoch_out = tbal.run_al_loop()
    
    logger.info('AL Loop Done')
    #test_err = al.get_test_error(al.cur_clf,test_set,conf['inference_conf'])
    out =  dm.get_auto_labeling_counts()

    #,"epoch_outs":lst_epoch_out
    if return_per_epoch_out:
        return {"counts":out ,"lst_epoch_out":lst_epoch_out}
    else:
        return {"counts":out }


def run_conf(conf,overwrite=True):

    if(not overwrite):
        if(os.path.exists(conf['out_file_path'])):
            print(f"path exists {conf['out_file_path']}")
            return 
    try:
        os.makedirs(conf['run_dir'])
    except OSError:
        pass

    set_defaults(conf)

    conf['inference_conf']['device'] = conf['device']

    #if('run_dir' in conf):
    #    print('run_dir ',conf['run_dir'])
    
    logger = get_logger(conf['log_file_path'],stdout_redirect=False,level=logging.DEBUG)
    
    if(conf['method']=='tbal'):
        out = run_tbal_conf(conf,logger)
    elif(conf['method']=='active_learning'):
        out = run_active_learnling_auto_labeling(conf,logger)
    elif(conf['method']=='passive_learning'):
        out = run_passive_labeling_auto_labeling(conf,logger)
    
    with open(conf['out_file_path'], 'wb') as out_file:
        pickle.dump(out, out_file, protocol=pickle.HIGHEST_PROTOCOL) 
    
    close_logger(logger)


def create_confs(conf,params):
    from itertools import product
    keys, values = zip(*params.items())
    lst_confs = []
    for bundle in product(*values):
        d = dict(zip(keys, bundle))
        conf = copy.deepcopy(conf)
        
        n_q = d['max_num_train_pts']
        seed_size        = int(n_q*d['seed_frac'])
        query_batch_size = int(n_q*d['query_batch_frac'])

        conf['method']      = d['method'] 
        conf["random_seed"] = d['seed']
        
        conf['train_pts_query_conf']['max_num_train_pts'] = d['max_num_train_pts']
        conf['train_pts_query_conf']['seed_train_size'] = seed_size
        conf['train_pts_query_conf']['query_batch_size'] = query_batch_size
        conf['val_pts_query_conf']['max_num_val_pts'] = d['max_num_val_pts']
        conf['auto_lbl_conf']['C_1'] = d['C_1']
        conf['auto_lbl_conf']['auto_label_err_threshold'] =d['eps']

        if(conf['method'] in ['tbal','active_learning']):
            conf['train_pts_query_conf']['margin_random_v2_constant'] = d['C']
            #conf['train_pts_query_conf']['seed_train_size'] = seed_size
            #conf['train_pts_query_conf']['query_batch_size'] = query_batch_size
            conf['stopping_criterion']= "max_num_train_pts"
        
        conf['run_dir'] = '/'.join([ f'{k}__{d[k]}' for k in sorted(d.keys())])
        
        conf['run_dir'] = f"{conf['root_pfx']}/{conf['run_dir']}"

        conf['log_file_path']  = '{}/{}.log'.format(conf['run_dir'],conf['method'])
        conf['out_file_path']  = '{}/{}.pkl'.format(conf['run_dir'],conf['method'])
        conf['conf_save_path'] = '{}/{}.yaml'.format(conf['run_dir'],conf['method'])

        lst_confs.append(conf)

    return lst_confs 


def run_conf_2(conf):
    logger = get_logger(conf['log_file_path'],stdout_redirect=False,level=logging.DEBUG)
    logger.info('Dry Run..')
    close_logger(logger)

def par_run(lst_confs,overwrite=True):
    lstP = []
    print(len(lst_confs))
    for conf in lst_confs:
        #print(conf)
        #conf = copy.deepcopy(conf) # ensure no shit happens
        p = Process(target = run_conf, args=(conf,overwrite))
        p.start()
        
        lstP.append(p)
    for p in lstP:
        p.join()

def assign_devices_to_confs(lst_confs,lst_devices = ['cpu']): 
    #round robin
    i = 0
    n = len(lst_confs)
    while(i<n):
        for dev in lst_devices:
            if(i<n):
                lst_confs[i]['device'] = dev
            i+=1
    
def exclude_existing_confs(lst_confs):
    lst_out_confs = []
    for conf in lst_confs:
        path = conf["out_file_path"]
        if os.path.exists(path):
            print(f"path exists {conf['out_file_path']}")
        else:
            lst_out_confs.append(conf)
    return lst_out_confs

def batched_par_run(lst_confs,batch_size=2, lst_devices=['cpu'],overwrite=True):
    
    if(not overwrite):
        lst_confs = exclude_existing_confs(lst_confs)
        n = len(lst_confs)
        print(f'NUM confs to run : {n}')

    assign_devices_to_confs(lst_confs,lst_devices)

    i=0
    n = len(lst_confs)
    while(i<n):
        print(f'running confs from {i} to {i+batch_size} ')
        #for conf in lst_confs[i:i+batch_size]:
        #    print(conf['device'])
        par_run(lst_confs[i:i+batch_size],overwrite)
        i+=batch_size 
    

def seq_run(lst_confs):
    for conf in lst_confs:
        run_conf(conf)

    

