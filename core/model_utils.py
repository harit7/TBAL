
from sklearn.metrics import accuracy_score
import models.clf_factory 
from datasets.dataset_utils import randomly_split_dataset
import copy 
import torch


def train_model(train_dataset,model_conf,training_conf,infernce_conf,logger,cur_val_ds=None):
    # create a new model for training. make it part of config...
    # in some cases we might use the previous model and retrain.
    #clf.fit(train_dataset, training_conf,self.ds_val)
    clf = models.clf_factory.get_classifier(model_conf,logger)

        
    # in some cases we might use the previous model and retrain.
    S = training_conf['num_trials']
    #if(S==1):
    
    clf.fit(train_dataset, training_conf,cur_val_ds)

    if(S>1):
        min_train_err = 1.0
        min_train_err_clf = clf
        inf_conf = infernce_conf
        for t in range(S):
            training_conf['seed']=t
            ts_sub,_ = randomly_split_dataset(train_dataset,0.5,random_state=t)
            try:
                clf.fit(ts_sub, training_conf,cur_val_ds)
                train_err = get_test_error(clf,train_dataset,inf_conf)
                #print(f'train_err {train_err}')
                if(train_err<min_train_err):
                    min_train_err = train_err 
                    min_train_err_clf = copy.deepcopy(clf)
            except:
                logger.debug('Error in Training...')
                pass
            
        clf = min_train_err_clf
                
    return clf 


def get_test_error(clf,test_ds,inference_conf):
    inf_out = clf.predict(test_ds, inference_conf) 
    test_err = 1-accuracy_score(inf_out['labels'],test_ds.Y)
        
    return test_err 

def run_inference(ds,clf,inference_conf,logger):
    inf_out = clf.predict(ds, inference_conf) 
    return inf_out 

