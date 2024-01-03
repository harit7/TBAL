
from sklearn.metrics import accuracy_score

def get_test_error(self,clf,test_ds,inference_conf):
    inf_out = clf.predict(test_ds, inference_conf) 
    test_err = 1-accuracy_score(inf_out['labels'],test_ds.Y)
        
    return test_err 


def get_counts(self,train_set,test_set):
    
    logger = self.logger 
    logger.info("------------------ Begin Get Counts ---------------")

    df = self.meta_df
    df['true_label'] = train_set.Y 

    a = df.query('true_label==label')['idx'].count()
    n = len(df)
    out = {}

    #self.logger.info("counting human_labeled_count")

    b = df.query('is_auto_labeled==True & true_label == label')['idx'].count()

    out['human_labeled_count'] = df.query('human_labeled==True')['idx'].count()

    n_q = out['human_labeled_count']

    out['human_labeled_count_train'] = df.query('human_labeled==True & is_val==False')['idx'].count()

    #out['human_labeled_count_val'] = df.query('human_labeled==True & is_val==True')['idx'].count()

    n_v_end = self.get_current_validation_count() #out['human_labeled_count_val']
    n_u_end = self.get_current_unlabeled_count()

    auto_labeled_count = df.query('is_auto_labeled==True')['idx'].count()


    out['num_auto_labeled'] = auto_labeled_count

    if(auto_labeled_count>0):
        out['auto_labeled_acc'] = b/auto_labeled_count
    else:
        out['auto_labeled_acc'] = None

    out['overall_acc'] = a/n

    out['coverage_1'] = auto_labeled_count/n
    out['coverage_2'] = auto_labeled_count/(n-n_q)
    
    logger.info('Auto Labeling Accuracy : {}'.format(out['auto_labeled_acc']))
    logger.info('Auto Labeling Coverage 1 : {}'.format(out['coverage_1']))
    logger.info('Auto Labeling Coverage 2 : {}'.format(out['coverage_2']))

    logger.info('Training pts count : {}'.format(n_q))
    #logger.info('Validation pts count at the begining : {}'.format())
    logger.info('Validation pts count at the end : {}'.format(n_v_end))
    logger.info('Total pool pts count :{}'.format(n))
    logger.info('Unlabeled pts count at the begining : {}'.format(n-n_q))
    logger.info('Unlabeled pts count at the end : {}'.format(n_u_end))
    

    inference_conf = self.conf['inference_conf']
    
    cur_train_ds = self.get_training_data()

    out['train_accuracy'] = 1 - self.get_test_error(self.cur_clf,cur_train_ds,inference_conf)
    out['pool_accuracy'] = 1 - self.get_test_error(self.cur_clf,train_set,inference_conf)
    out['test_accuracy'] = 1 - self.get_test_error(self.cur_clf,test_set,inference_conf)

    logger.info('Accuracy on the data used for training  : {}'.format(out['train_accuracy']))
    logger.info('Accuracy on entire pool  : {}'.format(out['pool_accuracy']))
    logger.info('Test Accuracy : {}'.format(out['test_accuracy']))
    
    self.logger.info("------------------ End Get Counts ---------------")

    return out
