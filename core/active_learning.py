
import copy 
from .query_strategies import * 
from .conf_defaults import *
import models.clf_factory 
from .model_utils import *
from .query_strategies.query_strategies_factory import * 

class ActiveLearning:
    
    def __init__(self,conf, dm,logger=None):
                
        set_defaults(conf)

        self.conf   = conf 
        self.logger = logger
        self.dm     = dm 
        
        self.ds_unlbld     = dm.ds_std_train
        self.ds_std_val    = dm.ds_std_val
        self.ds_std_test   = dm.ds_std_test 
        
        self.N_v_std       = len(self.ds_std_val)


        self.random_seed        = conf['random_seed'] 
        
        self.stopping_criterion = conf['stopping_criterion']

        self.store_model_weights_in_mem = False 
        
        self.epoch   = 0
        self.cur_query_count = 0
        self.lst_classes = [0,1]
        self.num_classes = len(self.lst_classes)
        self.cur_train_err = 1.0
        self.cur_test_err = 1.0
        self.cur_val_err = 1.0
        
        self.logger = logger

        self.max_train_query = conf.train_pts_query_conf.max_num_train_pts

    
    def init(self):
        self.query_strategy = get_query_strategy(self.dm,None,self.conf,self.logger)
        # do any clustering etc. here.
    


    def query_seed_points(self,epoch_out):
        logger = self.logger 
        seed_train_size  = self.conf.train_pts_query_conf.seed_train_size

        logger.debug('Querying {} seed training points'.format(seed_train_size))
        q_idx = self.dm.select_seed_train_points(k=seed_train_size,method='randomly')
        
        epoch_out['seed_train_pts'] = q_idx 
        epoch_out['query_points']   = q_idx
        self.cur_query_count       += len(q_idx)

        epoch_out['true_labels']    = self.dm.get_true_labels(q_idx)

        logger.debug('Queried {} seed points for training'.format(len(q_idx)))

        #n_v                        = int(self.N_v_std * self.val_frac_for_auto_lbl)
        val_query_conf = self.conf.val_pts_query_conf 
        n_v            = val_query_conf.max_num_val_pts

        cur_val_idcs               = self.dm.query_validation_points(n_v,method='random')
        cur_val_ds,cur_val_idcs    = self.dm.get_current_validation_data()
        epoch_out['seed_val_pts']  = cur_val_idcs

        logger.debug('Validation Data Size :{}'.format(n_v))

    def query_training_batch(self,epoch_out):
        logger = self.logger 
        # query points to add in the training set.
        logger.debug('Querying next training batch')
        q_conf  = self.conf.train_pts_query_conf

        cur_avlbl_q_bgt = self.max_train_query - self.cur_query_count

        n_u             = self.dm.get_current_unlabeled_count()

        bs              = min(n_u, q_conf.query_batch_size, cur_avlbl_q_bgt)

        logger.debug(f'Query Batch Size = {bs}')

        self.query_strategy = get_query_strategy(self.dm,self.cur_clf,self.conf,self.logger)

        q_idx               = self.query_strategy.query_points(bs)
        
        self.dm.mark_queried(q_idx,round_id=self.epoch)

        logger.info(f'Queried {len(q_idx)} pts to add in training pool')

        epoch_out['query_points'] = q_idx
        epoch_out['true_labels']    = self.dm.get_true_labels(q_idx)
        
        self.cur_query_count += len(q_idx)
    
    def run_one_epoch(self):
        
        logger = self.logger 
        epoch  = self.epoch 
        conf   = self.conf

        epoch_out = {}
        logger.info('===========================================================================')
        logger.debug(f'========================= BEGIN EPOCH {epoch} ============================')
       # <<<<<<<<<<<<<<<<<<<<<<<<< BEGIN QUERYING POINTS BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<
        n_u = self.dm.get_current_unlabeled_count()
        logger.debug('Number of unalabeled points  :{}'.format(n_u))

        if(epoch==0):
            logger.info('Querying first batches of training and validation samples.')
            self.query_seed_points(epoch_out)
        else:
            self.query_training_batch(epoch_out)

        n_u = self.dm.get_current_unlabeled_count()
        cur_val_ds,cur_val_idcs  = self.dm.get_current_validation_data()

        logger.debug('Num Unlabeled Points After Querying :{}'.format(n_u))        
        
        #  >>>>>>>>>>>>>>>>>>>>>>>>>>> END QUERYING POINTS BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>
     
        
        #<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN TRAINING BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<
        logger.info('===========================================================================')
        logger.info('========================== Begin Model Training ===========================')
        
        # train new model
        cur_train_ds, cur_train_idcs = self.dm.get_current_training_data()

        # check if there is just one class in training data
        bad_ds_flag = len(set(cur_train_ds.Y))==1
        if(bad_ds_flag):
            logger.error('Bad training dataset with only one class!!!')
            logger.error('Going to the next epoch to get more training points.')
            # go to next epoch and query more points.
            return epoch_out 

        self.cur_clf = train_model(cur_train_ds,conf.model_conf, conf.training_conf, conf.inference_conf, 
                                   logger,cur_val_ds)
        cur_val_ds,cur_val_idcs    = self.dm.get_current_validation_data()

        train_err    = get_test_error(self.cur_clf, cur_train_ds     , conf.inference_conf)
        val_err      = get_test_error(self.cur_clf, cur_val_ds       , conf.inference_conf)
        test_err     = get_test_error(self.cur_clf, self.ds_std_test , conf.inference_conf)

        epoch_out['train_error'] = train_err
        epoch_out['val_error']   = val_err 
        epoch_out['test_err']    = test_err    

        self.cur_val_err = val_err
        
        logger.info(f'Training Error: {train_err:.2f} \t Validation Error : {val_err:.2f} \t Test Error : {test_err:.2f} ')

        logger.info('========================= End Model Training   =========================')
        logger.info('===========================================================================')
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> END TRAINING BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>   
        
        #<<<<<<<<<<<<<<<<<<<<<<<<< BEGIN SAVING STATE <<<<<<<<<<<<<<<<<<<<<<<<<
        if(self.conf.store_model_weights_in_mem):
            epoch_out['clf_weights'] = self.cur_clf.get_weights()
            epoch_out['cur_clf'] = copy.deepcopy(self.cur_clf)
                
        #>>>>>>>>>>>>>>>>>>>>>>>>>>> END SAVING STATE >>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        logger.debug('=============================== END Epoch {} ======================='.format(epoch))

        return epoch_out 
    

    def run_al_loop(self):
        ## Q: How to deterimine current error ... 
        # using a validation set and self.cur_err > error_threshold
        self.epoch = 0
        lst_epoch_out = []
        #print(self.cur_query_count)
        
        train_conf = self.conf['training_conf']
        
        prev_q = 0           

        while(not self.check_stopping_criterion() ):
            self.cur_query_count += prev_q
            epoch_out = self.run_one_epoch()
            #print(epoch_out)
            self.epoch+=1
            if('query_points' in epoch_out):
                prev_q = len(epoch_out['query_points'])
            else:
                prev_q = 0
            lst_epoch_out.append(epoch_out)
        
        out = {}
        out["lst_epoch_out"] = lst_epoch_out
        out["embeddings"] = []

        return lst_epoch_out
        

    def check_stopping_criterion(self):

        n_u = self.dm.get_current_unlabeled_count()
        labeled_all = n_u == 0
        self.logger.debug('Stop Criterion {}'.format(self.stopping_criterion))
        self.logger.debug('Unlabeled Count In check_stop_criterion {}'.format(n_u))
        self.logger.debug('cur_query_count= {} and max_query_count={}'.format(self.cur_query_count,self.max_train_query))
        
        
        if(labeled_all):
            # This conditions overrides all criterias. since no more unlabeled points left
            return labeled_all 

        err_th = self.cur_val_err 
        #err_th = err_th + 0.1 * np.sqrt(err_th*(1-err_th))

        # first check validation error condition.
        val_err_th = self.conf['val_err_threshold']
        self.logger.debug(f'err_th {err_th} and val_err_threshold {val_err_th}')

        if(err_th<= val_err_th):
            return True 
        
        if(self.stopping_criterion=='max_epochs'):
            return self.epoch >= self.conf['max_epochs']
        
        if(self.cur_query_count >= self.max_train_query):
            return  True
        
        err_th = self.cur_val_err 
        #err_th = err_th + 0.1 * np.sqrt(err_th*(1-err_th))

        # first check validation error condition.
        val_err_th = self.conf['val_err_threshold']
        self.logger.debug(f'err_th {err_th} and val_err_threshold {val_err_th}')

        if(err_th<= val_err_th):
            return True 
        
        '''
        if(self.stopping_criterion=='val_error'):
            #print('train err ',self.cur_train_err)
            #print('val err ',self.cur_val_err)
            err_th = self.cur_val_err 
            #err_th = err_th + 0.1 * np.sqrt(err_th*(1-err_th))
            return err_th<= self.conf['val_err_threshold'] 

        if(self.stopping_criterion=='max_query'):
            return self.cur_query_count >= self.conf['max_query']
        
        '''
         
        n_u = self.dm.get_current_unlabeled_count()
        if(n_u == 0):
            return True 
        
        # implement other stopping criterions.. 
        # query budget
        # error threshold

    def save_state(self,path):
        if(self.conf['model_conf']['lib']=='pytorch'):
            model_state_dict = self.cur_clf.model.state_dict()
        else:
            model_state_dict =None 

        torch.save({ 'model_state_dict': model_state_dict,
                        'conf':self.conf,
                        'meta_df':self.meta_df 
                    }, path)
        
    def load_state(self,path):
        checkpoint = torch.load(path)
        if(self.conf['model_conf']['lib']=='pytorch'):
            self.cur_clf = models.clf_factory.get_classifier(self.conf['model_conf'],self.logger)
            self.cur_clf.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.meta_df = checkpoint['meta_df']    
    
            
