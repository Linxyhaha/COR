import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# from tensorboardX import SummaryWriter
from scipy import sparse
import models
import random
import data_utils
import evaluate_util
import os

parser = argparse.ArgumentParser(description='PyTorch COR Inference')
parser.add_argument('--model_name', type=str, default='COR',
                    help='model name')
parser.add_argument('--dataset', type=str, default='synthetic',
                    help='dataset name')
parser.add_argument('--data_path', type=str, default='../data/',
                    help='directory of all datasets')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument('--CI', type=int, default=1,
                    help='whether use counterfactual inference in ood settings')
parser.add_argument("--topN", default='[10, 20, 50, 100]',  
                    help="the recommended item num")
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU id')
parser.add_argument('--ckpt', type=str, default=None,
                    help='pre-trained best iid model path')

args = parser.parse_args()
print(args)

random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# device = torch.device("cuda:"+args.gpu if args.cuda else "cpu")
device = torch.device("cuda:0" if args.cuda else "cpu")
print(f'using device {device}')

###############################################################################
# Load data
###############################################################################
data_path = args.data_path + args.dataset + '/'

# load iid data
iid_test_dataset = 'iid'
train_path = data_path + '{}/training_list.npy'.format(iid_test_dataset)
valid_path = data_path + '{}/validation_dict.npy'.format(iid_test_dataset)
test_path = data_path + '{}/testing_dict.npy'.format(iid_test_dataset)
if args.dataset=='synthetic':
    user_feat_path = data_path + '{}/user_preference.npy'.format(iid_test_dataset)
else:
    user_feat_path = data_path + '{}/user_feature.npy'.format(iid_test_dataset)
item_feat_path = data_path + '{}/item_feature.npy'.format(iid_test_dataset)

train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \
                                            data_utils.data_load(train_path, valid_path, test_path, args.dataset)
user_feature, item_feature = data_utils.feature_load(user_feat_path, item_feat_path)
if args.dataset=='synthetic':
    user_feature = user_feature[:,0:1]
    item_feature = torch.FloatTensor(item_feature[:,1:9]).to(device)
else:
    item_feature = torch.FloatTensor(item_feature).to(device)

# load ood data
ood_test_dataset = 'ood'
ood_train_path = data_path + '{}/training_list.npy'.format(ood_test_dataset)
ood_valid_path = data_path + '{}/validation_dict.npy'.format(ood_test_dataset)
ood_test_path = data_path + '{}/testing_dict.npy'.format(ood_test_dataset)
if args.dataset=='synthetic':
    ood_user_feat_path = data_path + '{}/user_preference.npy'.format(ood_test_dataset)
else:
    ood_user_feat_path = data_path + '{}/user_feature.npy'.format(ood_test_dataset)

ood_item_feat_path = data_path + '{}/item_feature.npy'.format(ood_test_dataset)
ood_train_data, ood_valid_x_data, ood_valid_y_data, ood_test_x_data, ood_test_y_data, ood_n_users, ood_n_items = \
                                    data_utils.data_load(ood_train_path, ood_valid_path, ood_test_path, args.dataset)
ood_user_feature, ood_item_feature = data_utils.feature_load(ood_user_feat_path, ood_item_feat_path)
if args.dataset=='synthetic':
    ood_user_feature = ood_user_feature[:,0:1]


mask_test =  train_data + valid_y_data

N = train_data.shape[0]
idxlist = list(range(N))

###############################################################################
# load the model
###############################################################################

model = torch.load(args.ckpt)
criterion = models.loss_function

###############################################################################
# Inference 
###############################################################################

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def evaluate(data_tr, data_te, his_mask, user_feat, topN, CI=0):
    
    assert data_tr.shape[0] == data_te.shape[0] == user_feat.shape[0]

    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i,:].nonzero()[1].tolist())
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]  
            his_item = his_mask[e_idxlist[start_idx:end_idx]]          
            user_f = torch.FloatTensor(user_feat[e_idxlist[start_idx:end_idx]]).to(device)
            data_tensor = naive_sparse2tensor(data).to(device)
            Z2_reuse_batch=None

            recon_batch, mu, logvar, _ = model(data_tensor, user_f, Z2_reuse_batch, CI)

            loss = criterion(recon_batch, data_tensor, mu, logvar)
            total_loss += loss.item()
            
            # Exclude examples from training set
            recon_batch[his_item.nonzero()] = -np.inf

            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)
    
    return total_loss, test_results
        

print('Inference COR on iid and ood data')

print('---'*18)
print('Given IID Interaction')
print("Performance on iid")
valid_loss, valid_results = evaluate(valid_x_data, valid_y_data, valid_x_data, user_feature, eval(args.topN))
test_loss, test_results = evaluate(test_x_data, test_y_data, mask_test, user_feature, eval(args.topN))
evaluate_util.print_results(None, valid_results, test_results)

print("Performance on ood")
valid_loss, valid_results = evaluate(valid_x_data, ood_valid_y_data, mask_test, ood_user_feature, eval(args.topN),args.CI)
test_loss, test_results = evaluate(test_x_data, ood_test_y_data, mask_test, ood_user_feature, eval(args.topN),args.CI)
evaluate_util.print_results(None, valid_results, test_results)
print('---'*18)
