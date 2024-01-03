import torch 
import numpy as np 
from .query_strategies import * 
from .conf_defaults import *
import models.clf_factory 
from .threshold_estimation import *


class AutoLabeling:
    
    def __init__(self,conf,dm,clf=None,logger=None):
        
        set_defaults(conf)
        
        self.conf = conf 
        self.cur_clf   =  clf 
        self.logger = logger 
        self.dm = dm # data manager

        self.meta_df = dm.meta_df 

        self.ds_unlbld = dm.ds_std_train
        self.ds_val    = dm.ds_std_val
        self.ds_test   = dm.ds_std_test 
        
        self.random_seed = conf['random_seed']

        self.auto_lbl_conf = conf['auto_lbl_conf']

        self.auto_label_err_threshold = self.auto_lbl_conf['auto_label_err_threshold']

        self.max_t = float('inf')
        self.num_classes = self.dm.num_classes
        self.lst_classes = np.arange(0,self.num_classes)
        self.margin_thresholds = [self.max_t]*self.num_classes
    
    def run(self,epoch=0):
        
        logger = self.logger 
        conf = self.conf

        epoch_out = {}
        auto_lbl_conf = self.auto_lbl_conf

        # Before doing anything first check if there are points to auto-label.
        
        cur_unlbld_idcs  = self.dm.get_current_unlabeled_idcs()
        cur_unlbld_idcs = np.array(cur_unlbld_idcs)
        n_u = len(cur_unlbld_idcs)
        if(n_u==0):
            logger.info('No unlabeled points left, exiting..')
            return {}
        
        method_name = auto_lbl_conf['method_name']

        logger.info(f'========================= Begin Auto-Labeling {method_name} ==========================')
        logger.debug('Auto Labeling Conf : {}'.format(auto_lbl_conf))
        logger.info('Number of unlabeled points : {}'.format(n_u))

        epoch_out['unlabeled_pts_idcs'] = cur_unlbld_idcs
        epoch_out['num_unlabeled'] = n_u 
        
        # load checkpoint from the path given in the auto-label config.
        # TODO: Want to have data manager state and model in the checkpoint

        if(self.cur_clf is None):
            # load from check point
            ckpt_load_path =auto_lbl_conf['ckpt_load_path']

            logger.info('Loading model checkpoint from :{}'.format(ckpt_load_path))
            self.load_state(ckpt_load_path)

        cur_unlbld_ds = self.dm.get_subset_dataset(cur_unlbld_idcs)

        #cur_val_ds,cur_val_idcs = self.dm.get_validation_data()
        cur_val_ds,cur_val_idcs = self.dm.get_current_validation_data()
        #unlbld_subset_ds = self.ds_unlbld.get_subset(unlbld_idcs)
        
        if(method_name=='all'):
            lst_auto_lbld_pts = self.auto_label_all(cur_unlbld_idcs,cur_unlbld_ds,epoch_out)
            
            epoch_out['val_idcs_to_rm'] = cur_val_idcs

        elif(method_name=='selective'):
            lst_auto_lbld_pts = self.selective_auto_label(cur_unlbld_idcs,cur_unlbld_ds,epoch_out)

        n_a = len(lst_auto_lbld_pts)
            
        epoch_out['auto_lbld_idx_lbl'] = lst_auto_lbld_pts
        epoch_out['num_auto_labeled'] = n_a

        # mark auto-labeled points
        self.dm.mark_auto_labeled(lst_auto_lbld_pts,round_id=epoch)

        logger.info('Num auto labeled points : {} '.format(n_a))

        val_idcs_to_rm = epoch_out['val_idcs_to_rm']
        #self.dm.remove_validation_points(val_idcs_to_rm,round_id=epoch)
        logger.info('Num validation pts to remove : {}'.format(len(val_idcs_to_rm)))
        
        logger.info('============================== Done Auto-Labeling ==============================')
        return epoch_out 
    
    def auto_label_all(self,cur_unlbld_idcs,cur_unlbld_ds,epoch_out):
        unlbld_inf_out = self.run_inference(self.cur_clf,cur_unlbld_ds,self.conf['inference_conf'])
        epoch_out['lst_t_i'] = [0]*self.num_classes

        scores = unlbld_inf_out[self.auto_lbl_conf['score_type']]
        y_hat  = unlbld_inf_out['labels']

        n_u = len(cur_unlbld_idcs)
        selected_idcs = cur_unlbld_idcs

        lst_auto_lbld_pts = [{'idx':cur_unlbld_idcs[i],'label':int(y_hat[i]),'confidence':float(scores[i])} for i in range(n_u)  ]
        epoch_out['unlbld_inf_out'] = unlbld_inf_out

        return lst_auto_lbld_pts

    def selective_auto_label(self,cur_unlbld_idcs,cur_unlbld_ds,epoch_out):

        cur_val_ds,cur_val_idcs = self.dm.get_current_validation_data()
        n_v = len(cur_val_ds) 
        epoch_out['cur_num_val'] = len(cur_val_ds) 
        err_threshold = self.auto_label_err_threshold
        logger = self.logger 
        logger.info('Using number of validation points : {}'.format(n_v))
        logger.info('Using Auto-Labeling Error Threshold = {}'.format(err_threshold))

        lst_t_val = []
        
        val_inf_out = self.run_inference(self.cur_clf,cur_val_ds,self.conf['inference_conf'])
        
        val_inf_out['true_labels']  = cur_val_ds.Y 
        epoch_out['val_inf_out'] = val_inf_out 

        logger.info('Determining Thresholds : Class Wise : {}'.format(self.auto_lbl_conf['class_wise']))
        
        lst_t_val, val_idcs_to_rm, val_err = determine_threshold(self.lst_classes,val_inf_out,
                                                                 self.auto_lbl_conf,cur_val_ds,
                                                                 cur_val_idcs,logger,err_threshold)
        
        logger.info('auto-labeling thresholds from val set: {}'.format(lst_t_val))
        
        epoch_out['val_idcs_to_rm'] = val_idcs_to_rm
        epoch_out['val_err'] = val_err 

        unlbld_inf_out = self.run_inference(self.cur_clf,cur_unlbld_ds,self.conf['inference_conf'])
        
        epoch_out['unlbld_inf_out'] = unlbld_inf_out
        epoch_out['lst_t_i_val'] = lst_t_val
        
        scores = unlbld_inf_out[self.auto_lbl_conf['score_type']]
        y_hat = unlbld_inf_out['labels']

        lst_t_val = np.array(lst_t_val) 
        lst_auto_lbld_pts = []
        n = len(cur_unlbld_idcs)
        
        # check if the score is bigger than the threshold for the predicted class.
        selected_idcs = [ i for i in range(n) if scores[i]>=lst_t_val[y_hat[i]] ]

        lst_auto_lbld_pts = [{'idx':cur_unlbld_idcs[i],'label':int(y_hat[i]),'confidence':float(scores[i])} for i in selected_idcs  ]
        return lst_auto_lbld_pts
    
    def save_state(self,path):
        torch.save({ 'model_state_dict': self.cur_clf.model.state_dict(),
                    'conf':self.conf,
                    'meta_df':self.meta_df 
                    }, path)
        
    def load_state(self,path):
        checkpoint = torch.load(path)
        self.cur_clf = models.clf_factory.get_classifier(self.conf['model_conf'],self.logger)

        self.cur_clf.model.load_state_dict(checkpoint['model_state_dict'])
    
    def run_inference(self,clf, ds, inference_conf):

        inf_out = clf.predict(ds,inference_conf)

        return inf_out