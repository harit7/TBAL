import numpy as np 

class RandomSamplingStrategy():
    def __init__(self,dm,clf,conf,logger):
        self.dm = dm 
        self.clf = clf 
        self.logger = logger
        self.conf = conf 
        pass 
        
    def query_points(self,batch_size,inf_out=None):
        #np.random.seed(random_seed)
        cur_unlbld_idcs  = self.dm.get_current_unlabeled_idcs()
        
        selected_idcs    = np.random.choice(cur_unlbld_idcs,batch_size,replace=False)
        
        selected_idcs    = selected_idcs.astype(int)

        return selected_idcs
