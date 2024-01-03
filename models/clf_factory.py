import sys

sys.path.append('../../')

from models.torch.pytorch_clf import *
from models.sklearn.sklearn_clf import *

def get_classifier(model_conf,logger=None):
    # conf should have, lib = sklearn or torch
    # and model_conf, train_conf, inference_conf
    logger = logger 
    if(model_conf['lib']=='pytorch'):
        clf = PyTorchClassifier(model_conf,logger=logger)
    elif(model_conf['lib']=='sklearn'):
        clf = SkLearnClassifier(model_conf,logger=logger)
    else:
        logger.info('invalid lib')
    
    return clf
    
    