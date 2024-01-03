import numpy as np
import torch 
from collections import defaultdict 

# for badge sampling
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats
# ----------------

# for cluster margin sampling
from sklearn.cluster import AgglomerativeClustering
# -----------------

def random_sampling(n,batch_size,random_seed=0):
    np.random.seed(random_seed)
    return np.random.choice(np.arange(0,n,1),batch_size,replace=False)

def entropy_sampling(inf_out,batch_size=1,random_seed=0):
    # get samples based on entropy score
    confidence_scores = inf_out["probs"]
    if type(confidence_scores) == np.ndarray or type(confidence_scores) == list: 
        probs = torch.Tensor(list(confidence_scores))
    else:
        probs = confidence_scores
    log_probs = torch.log(probs)
    U = (probs*log_probs).sum(1)
    sample_size = min(batch_size,len(confidence_scores))
    inx = U.sort()[1][:sample_size]
    return inx.numpy()

def margin_sampling(inf_out,batch_size=1,random_seed=0):

    confidence_scores = inf_out['probs']
    if type(confidence_scores) == np.ndarray or type(confidence_scores) == list:    
        probs = torch.Tensor(np.array(confidence_scores))
    else:
        probs = confidence_scores
    probs_sorted, _ = probs.sort(descending=True)
    U = probs_sorted[:, 0] - probs_sorted[:,1]
    sample_size = min(batch_size,len(confidence_scores))
    inx = U.sort()[1].numpy()[:sample_size]
    return inx

def margin_random_sampling(inf_out,margin_thresholds,batch_size=1,random_seed=0):
    
    np.random.seed(random_seed)
    y_hat = inf_out['labels']
    #margin_scores = inf_out['margin_score']
    scores = inf_out['abs_logit']
    
    margin_thresholds = np.array(margin_thresholds)

    idcs_in_margin = []
    for i,y in enumerate(y_hat):
        if(scores[i]<= margin_thresholds[y]):
            idcs_in_margin.append(i)

    #print(margin_scores,margin_threshold)
    #idcs_in_margin = np.where(margin_scores<=margin_threshold)[0]
    
    sample_size = min(batch_size,len(idcs_in_margin))

    return np.random.choice(idcs_in_margin,sample_size,replace=False)

def badge_sampling(grad_embedding,num_classes,batch_size=1,random_seed=0):
    return badge_sampling_helper_clusters(grad_embedding, num_classes)

def badge_sampling_helper_clusters(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

def uncertainty_sampling(margin_scores,batch_size=1,random_seed=0):
    np.random.seed(random_seed)
    sample_size = min(batch_size,len(margin_scores))
    
    #print(np.argsort(margin_scores)[:10])
    #print(np.argsort(margin_scores)[:sample_size])

    return np.argsort(margin_scores)[:sample_size]

# https://github.com/cure-lab/deep-active-learning/blob/main/query_strategies/batch_active_learning_at_scale.py
def cluster_margin_sampling(embedding,margin_sampling_idx,num_classes,batch_size,random_seed):
    # step1: do the clustering
    # todo: what should be the cluster size??? be default, 20???
    HAC_list = AgglomerativeClustering(n_clusters=20, linkage = 'average').fit(embedding)

    # step3: pass the margin sampling indices (passed as a parameter), and use Round Robin to decides the final indices
    # todo: is the parameter the class number??
    rr_idx = cluster_margin_sampling_helper_round_robin(margin_sampling_idx, HAC_list.labels_, batch_size,len(embedding))
    #print("RR idx:",rr_idx,len(rr_idx))
    
    return rr_idx
def cluster_margin_sampling_helper_round_robin(unlabeled_index, hac_list, k , all_sample_size):
        # k is the number of samples alg wants to query
        cluster_list = []
        for i in range(all_sample_size):
            cluster = []
            cluster_list.append(cluster)
        for real_idx in unlabeled_index:
            i = hac_list[real_idx]
            cluster_list[i].append(real_idx)
        cluster_list.sort(key=lambda x:len(x))
        index_select = []
        cluster_index = 0
        while k > 0:
            if len(cluster_list[cluster_index]) > 0:
                index_select.append(cluster_list[cluster_index].pop(0)) 
                k -= 1
            if cluster_index < len(cluster_list) - 1:
                cluster_index += 1
            else:
                cluster_index = 0

        return index_select

def hac_uncertainty_sampling(margin_scores, cluster_assignments, batch_size=1,random_seed=0):
    np.random.seed(random_seed)
    cluster_count = defaultdict(int)
    cluster_map = defaultdict(list)
    
    idx_ = np.array(range(len(margin_scores)))

    for i in idx_:
        cluster_map[cluster_assignments[i]].append(i)

    cluster_id_len = [(c_id,len(cluster_map[c_id])) for c_id in cluster_map.keys()]
    cluster_id_len.sort(key=lambda x: x[1])
    out = []
    
    while(len(out)<batch_size):
        for c_idx,count in cluster_id_len:
            pts_idx_in_cluster = cluster_map[c_idx]
            if(len(pts_idx_in_cluster)>0):
                sampled_idx = np.random.choice(pts_idx_in_cluster,1)[0]
                out.append(sampled_idx)
                cluster_map[c_idx].remove(sampled_idx)
                cluster_count[c_idx] +=1
                if(len(out)>=batch_size):
                    break
    return out 


def select_points_for_human_labeling(query_strategy, batch_size, margin_thresholds=None, 
                                            random_seed=0,cluster_assignments=None):

    logger = self.logger 
    
    logger.debug('Using {} Query Strategy'.format(query_strategy))

    unlbld_idcs = self.get_current_unlabeled_pool()
    unlbld_idcs = np.array(unlbld_idcs)
    n_u = len(unlbld_idcs)

    #if(n_u <= batch_size):
    #    return unlbld_idcs,
    #

    unlbld_subset_ds = self.ds_unlbld.get_subset(unlbld_idcs)

    #inf_out = self.cur_clf.predict(unlbld_subset_ds,self.conf['inference_conf'])
    inf_out = self.run_inference(unlbld_subset_ds,self.cur_clf,self.conf['inference_conf'],self.cur_calibrator)


    #margin_scores = inf_out['margin_score']
    #print(inf_out.keys())
    abs_logits = inf_out['abs_logit'] 
    margin_scores = abs_logits
    #print(margin_scores.shape)
    
    #print(margin_scores.shape, len(margin_scores))

    scores = margin_scores 
    n = len(margin_scores)
    
    logger.debug('len(scores) = {} , batch_size ={}'.format(len(scores), batch_size))

    if(query_strategy=='random'):
        batch = random_sampling(n,batch_size,random_seed)
    elif(query_strategy=='uncertainty'): 
        batch = uncertainty_sampling(scores,batch_size,random_seed)
        
    elif(query_strategy=='margin_random'):
        batch = margin_random_sampling(inf_out,margin_thresholds,batch_size,random_seed)
    elif(query_strategy=='hac_uncertainty'):
        batch =  hac_uncertainty_sampling(margin_scores,cluster_assignments,batch_size,random_seed)

    elif(query_strategy=='margin_random_v2'):
        # get margin scores for: some unlabeled samples (e.g. 10 times bigger than query size)
        C = self.active_learning_conf['margin_random_v2_constant']
        
        #score_type = self.act_learn_conf['score_type']

        margin_sampling_idx = margin_sampling(inf_out,C*batch_size,random_seed)
        sample_size = min(batch_size,len(margin_scores))
        batch = np.random.choice(margin_sampling_idx,sample_size,replace=False)

    else:
        batch = None 
    
    #print(batch,margin_scores[batch])
    if(batch is None or len(batch)==0):
        self.logger.error('empty batch....')
        return [],0
    
    if(batch is not None):
        if(len(margin_scores)>0):
            margin = max(margin_scores[batch])
        else:
            margin = None 
    
    q_idx = unlbld_idcs[batch] 

    return q_idx, margin