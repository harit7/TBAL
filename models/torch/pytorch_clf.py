
from ..abstract_clf import AbstractClassifier
from .logistic_regression import  PyTorchLogisticRegression

from .model_training import *
from .clf_inference import *
from .clf_get_embedding import *
import torch
from .lenet import *
from .linear_model import LinearModel

from .cifar_small_net import CifarSmallNet
from .resnet import ResNet18
from .cifar_medium_net import CifarMediumNet
from .scaling_model import TemperatureScalingModel
from .two_layer_nets import *



from .resnet_v2 import *

class PyTorchClassifier(AbstractClassifier):

    def __init__(self,model_conf,logger):
        
        self.model_conf = model_conf 
        self.logger = logger 
        
        logger.info(model_conf)
        model_name = model_conf['model_name']

        if('num_classes' in model_conf):
            self.num_classes = model_conf['num_classes']

        if(model_conf['model_name']=='binary_logistic_regression'):
            assert model_conf['num_classes']==2 
            self.model= PyTorchLogisticRegression(model_conf,logger)
        
        if(model_conf['model_name']=='lenet'):
            self.model = LeNet5(model_conf['num_classes'])
        
        if(model_conf['model_name']=='linear_model'):
            self.model = LinearModel(model_conf)
        
        if(model_conf['model_name']=='cifar_small_net'):
            self.model = CifarSmallNet(model_conf['num_classes'])
        
        if(model_conf['model_name']=='resnet18'):
            self.model = ResNet18(model_conf['num_classes'])
        
        if(model_conf['model_name']=='resnet18_v2'):
            self.model =  resnet18_v2(n_classes=model_conf['num_classes'])

        if(model_conf['model_name']=='cifar_med_net'):
            self.model = CifarMediumNet(model_conf['num_classes'])
        
        if(model_conf['model_name']=='temp_scaling'):
            self.model = TemperatureScalingModel(model_conf)
        
        if(model_name=='two_layer_net'):
            self.model = TwoLayerNet(model_conf)
        
        if(model_name == "mlp"):
            from .mlp import MLP
            self.model = MLP(model_conf)
            
        if(model_name=='ViT'):
            from .cifar_vit_small import ViT
            self.model = ViT(model_conf)
        
        if(model_name== 'TextClassificationModel'): 
            from .text_embed_mlp import TextClassificationModel
            self.model = TextClassificationModel(model_conf) 

        if(model_name == 'text_clf_mlp_head'):
            from .text_clf_mlp_head import TextClassifierMLPHead
            self.model = TextClassifierMLPHead(model_conf)
            
    def fit(self,dataset,training_conf,val_set=None):
        model_training = ModelTraining(self.logger)
        out = model_training.train(self.model,dataset,training_conf,val_set=val_set)
        return out 


    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        return clf_inference.predict(self.model,dataset,inference_conf)

    def get_weights(self):
        w = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().cpu()
        return w

    def get_grad_embedding(self,dataset, inference_conf):
        clf_embedding = ClassifierEmbedding(self.logger)
        return clf_embedding.get_gard_embedding(self.model,dataset,inference_conf)

    def get_embedding(self,dataset, inference_conf):
        clf_embedding = ClassifierEmbedding(self.logger)
        return clf_embedding.get_embedding(self.model,dataset,inference_conf)