
def get_query_strategy(dm,clf,conf,logger):

    query_conf = conf['train_pts_query_conf']
    strat_name = query_conf['query_strategy_name']

    if(strat_name == 'random_sampling'):
        from .random_sampling import RandomSamplingStrategy
        return RandomSamplingStrategy(dm,clf,conf,logger)

    if(strat_name == 'entropy_sampling'):
        from .entropy_sampling import EntropySamplingStrategy
        return EntropySamplingStrategy(dm,clf,conf,logger)
    
    elif(strat_name == 'margin_sampling'):
        from .margin_sampling import MarginSamplingStrategy
        return MarginSamplingStrategy(dm,clf,conf,logger)
    
    elif(strat_name == 'margin_random_v2'):
        from .margin_random_v2 import MarginRandomV2
        return MarginRandomV2(dm,clf,conf,logger)
    
    elif(strat_name == 'uncertainty_sampling'):
        from .uncertainty_sampling import UncertaintySamplingStrategy
        return UncertaintySamplingStrategy(dm,clf,conf,logger)
    
    else:
        print(f'Query Strategy {strat_name} Not Defined')
        return None

