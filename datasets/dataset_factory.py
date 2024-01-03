

def load_dataset(conf):
    data_conf = conf['data_conf']
    dataset_name = data_conf['dataset']
    
    if(dataset_name == 'general_synthetic'):
        from datasets.synthetic_general import GeneralSynthetic    
        return GeneralSynthetic(data_conf)
    
    elif(dataset_name == 'synth_moon'):
        from datasets.synthetic_moon import SyntheticMoon
        
    elif(dataset_name == 'synth_concenteric_circles'):
        from datasets.concentric_circles import ConcentricCircles
        return ConcentricCircles(data_conf)
    
    elif(dataset_name == 'mnist'):
        from datasets.mnist import MNISTData
        return MNISTData(data_conf)

    elif(dataset_name == 'mnist_sklearn'):
        from datasets.mnist_sklearn import MNISTData_sklearn
        return MNISTData_sklearn(data_conf)

    elif(dataset_name == 'cifar10'):
        from datasets.cifar10 import Cifar10Data
        return Cifar10Data(data_conf)
    
    elif(dataset_name == 'cub_birds'):
        from datasets.cub_birds import CUB_BirdsData
        return CUB_BirdsData(data_conf)
    
    elif(dataset_name == 'tiny_imagenet_200'):
        from .tiny_imagenet import TinyImageNet200
        return TinyImageNet200(data_conf)
    
    elif(dataset_name == 'tiny_imagenet_200_CLIP'):
        from .tiny_imagenet_clip import TinyImageNet200CLIP
        return TinyImageNet200CLIP(data_conf)
    
    elif(dataset_name == 'unif_unit_ball'):
        from datasets.uniform_unit_ball import UniformUnitBallDataset
        return UniformUnitBallDataset(data_conf)
    
    elif(dataset_name == 'xor_balls'):
        from datasets.xor_balls import XORBallsDataset
        return XORBallsDataset(data_conf)
    
    elif(dataset_name == 'AG_NEWS'):
        from datasets.text_torch import TextTorch
        return TextTorch(data_conf,dataset_name)
    
    elif(dataset_name == 'IMDB'):
        from datasets.text_torch_emb import TextTorchEmb
        return TextTorchEmb(data_conf,dataset_name)
    
    else:
        print('Datset {} Not Defined'.format(dataset_name))
        return None

