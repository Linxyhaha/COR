import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ipdb

# from tensorboardX import SummaryWriter
from scipy import sparse
import models
import random
import data_utils
import evaluate_util
import os

parser = argparse.ArgumentParser(description='PyTorch COR')
parser.add_argument('--model_name', type=str, default='COR',
                    help='model name')
parser.add_argument('--dataset', type=str, default='synthetic',
                    help='dataset name')
parser.add_argument('--data_path', type=str, default='../data/',
                    help='directory of all datasets')
parser.add_argument('--log_name', type=str, default='',
                    help='log/model special name')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument("--mlp_dims", default='[100, 20]',  
                    help="the dims of the mlp encoder")
parser.add_argument("--mlp_p1_1_dims", default='[100, 200]',  
                    help="the dims of the mlp p1-1")
parser.add_argument("--mlp_p1_2_dims", default='[1]',  
                    help="the dims of the mlp p1-2")
parser.add_argument("--mlp_p2_dims", default='[]',  
                    help="the dims of the mlp p2")
parser.add_argument("--mlp_p3_dims", default='[10]',  
                    help="the dims of the mlp p3")
parser.add_argument("--Z1_hidden_size", type=int, default=8,
                    help="hidden size of Z1")
parser.add_argument('--E2_hidden_size', type=int, default=20,
                    help='hidden size of E2')
parser.add_argument('--Z2_hidden_size', type=int, default=20,
                    help='hidden size of Z2')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--sample_freq', type=int, default=1,
                    help='sample frequency for Z1/Z2')
parser.add_argument('--CI', type=int, default=1,
                    help='whether use counterfactual inference in ood settings')
parser.add_argument('--bn', type=int, default=1,
                    help='batch norm')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout')
parser.add_argument('--regs', type=float, default=0,
                    help='regs')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument("--topN", default='[10, 20, 50, 100]',  
                    help="the recommended item num")
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=str, default='1',
                    help='GPU id')
parser.add_argument("--ood_test", default=True,
                    help="whether test ood data during iid training")
parser.add_argument('--save_path', type=str, default='./models/',
                    help='path to save the final model')
parser.add_argument('--act_function', type=str, default='tanh',
                    help='activation function')
parser.add_argument('--ood_finetune',action='store_true',
                    help='fine-tuning on ood data')
parser.add_argument('--ckpt', type=str, default=None,
                    help='pre-trained model directory')
parser.add_argument('--X',type=int,default=10,
                    help='use X percent of ood data for fine-tuning')
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
device = torch.device("cuda:0" if args.cuda else "cpu")
print(f'using device {device}')

###############################################################################
# Load data
###############################################################################

data_path = args.data_path + args.dataset + '/'

if args.ood_finetune:
    train_dataset = 'ood'
    ood_test_dataset = 'iid'
else:
    train_dataset = 'iid'
    ood_test_dataset = 'ood'

# iid data
train_path = data_path + '{}/training_list.npy'.format(train_dataset)
valid_path = data_path + '{}/validation_dict.npy'.format(train_dataset)
test_path = data_path + '{}/testing_dict.npy'.format(train_dataset)

if args.ood_finetune:
    train_path = data_path + '{}/training_list_{}%.npy'.format('X_ood',args.X)
    if args.X == 0:
        train_path = data_path + '{}/training_list.npy'.format(ood_test_dataset)
if args.dataset=='synthetic':
    user_feat_path = data_path + '{}/user_preference.npy'.format(train_dataset)
else:
    user_feat_path = data_path + '{}/user_feature.npy'.format(train_dataset)
item_feat_path = data_path + '{}/item_feature.npy'.format(train_dataset)

train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \
                                            data_utils.data_load(train_path, valid_path, test_path, args.dataset)
user_feature, item_feature = data_utils.feature_load(user_feat_path, item_feat_path)
if args.dataset=='synthetic':
    user_feature = user_feature[:,0:1]
    item_feature = torch.FloatTensor(item_feature[:,1:9]).to(device)
else:
    item_feature = torch.FloatTensor(item_feature).to(device)

# ood data for testing on ood data
if args.ood_test:
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
N = train_data.shape[0]
idxlist = list(range(N))

# mask of interacted items for validation and testing
mask_val = train_data
mask_test =  train_data + valid_y_data
if args.ood_finetune:
    mask_val = train_data + ood_train_data + ood_valid_y_data 
    mask_test = train_data+ valid_y_data + ood_train_data + ood_valid_y_data 

###############################################################################
# Build the model
###############################################################################
if args.ood_finetune:
    model = torch.load(args.ckpt)
    ckpt_structure = args.ckpt.split('_')
else: 
    E1_size = user_feature.shape[1]
    Z1_size = args.Z1_hidden_size
    mlp_q_dims = [n_items+user_feature.shape[1]] + eval(args.mlp_dims) + [args.E2_hidden_size]

    # used for COR
    mlp_p1_dims = [E1_size + args.E2_hidden_size] + eval(args.mlp_p1_1_dims) + [Z1_size]

    # used for COR_G
    mlp_p1_1_dims = [1] + eval(args.mlp_p1_1_dims)
    mlp_p1_2_dims = [mlp_p1_1_dims[-1]] + eval(args.mlp_p1_2_dims)

    mlp_p2_dims = [args.E2_hidden_size] + eval(args.mlp_p2_dims) + [args.Z2_hidden_size]
    mlp_p3_dims = [Z1_size + args.Z2_hidden_size] + eval(args.mlp_p3_dims) +  [n_items] # need to delete

    # predefined causal graph
    adj = np.concatenate((np.array([[0.0]*E1_size + [1.0]*args.E2_hidden_size,
                                    [0.0]*E1_size + [1.0]*args.E2_hidden_size]), 
                        np.ones([6, E1_size+args.E2_hidden_size])), axis=0)
    adj = torch.FloatTensor(adj).to(device)

    if args.model_name == 'COR':
        model = models.COR(mlp_q_dims, mlp_p1_dims, mlp_p2_dims, mlp_p3_dims, \
                                                item_feature, adj, E1_size, args.dropout, args.bn, args.sample_freq, args.regs, args.act_function).to(device)
    elif args.model_name == 'COR_G':
        model = models.COR_G(mlp_q_dims, mlp_p1_1_dims, mlp_p1_2_dims, mlp_p2_dims, mlp_p3_dims, \
                                                item_feature, adj, E1_size, args.dropout, args.bn, args.sample_freq, args.regs, args.act_function).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models.loss_function

###############################################################################
# Training code
###############################################################################

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def adjust_lr(e):
    if args.dataset=='meituan':
        if e>90: # Decay 
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * args.lr
    elif args.dataset=='yelp':
        if e>60: # Decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * args.lr
    else:
        pass

def train():
    # Turn on training mode
    model.train()
    global update_count
    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        user_f = torch.FloatTensor(user_feature[idxlist[start_idx:end_idx]]).to(device)
        data = naive_sparse2tensor(data).to(device)
        Z2_reuse_batch=None
        if args.ood_finetune:
            Z2_reuse_batch = Z2_reuse[idxlist[start_idx:end_idx]] 

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()

        recon_batch, mu, logvar, reg_loss = model(data, user_f, Z2_reuse_batch)

        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss = loss + reg_loss
        loss.backward()
        optimizer.step()
        update_count += 1


def evaluate(data_tr, data_te, his_mask, user_feat, topN, CI=0):
    
    assert data_tr.shape[0] == data_te.shape[0] == user_feat.shape[0]
    
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
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
            user_f = torch.FloatTensor(user_feat[e_idxlist[start_idx:end_idx]]).to(device)
            data_tensor = naive_sparse2tensor(data).to(device)
            his_data = his_mask[e_idxlist[start_idx:end_idx]]
            Z2_reuse_batch = None
            if args.ood_finetune and args.X!=0:
                Z2_reuse_batch = Z2_reuse[e_idxlist[start_idx:end_idx]] 
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar, reg_loss = model(data_tensor, user_f, Z2_reuse_batch, CI)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            loss = loss + reg_loss
            total_loss += loss.item()
            
            # Exclude examples from training set
            recon_batch[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    total_loss /= len(range(0, e_N, args.batch_size))
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)
    return total_loss, test_results

if args.ood_finetune:
    with torch.no_grad():
        Z2_reuse = model.reuse_Z2(naive_sparse2tensor(ood_train_data).to(device),\
                            torch.FloatTensor(user_feature).to(device)) # ood user feature
    if args.X==0:
        print(f"Performance on {ood_test_dataset}")
        _, test_results = evaluate(ood_test_x_data, ood_test_y_data, ood_test_x_data+ood_valid_y_data, ood_user_feature, eval(args.topN))
        evaluate_util.print_results(None, None, test_results)

        print(f"Performance on {train_dataset}")
        _, test_results = evaluate(ood_test_x_data, test_y_data, ood_test_x_data+ood_valid_y_data, user_feature, eval(args.topN),1)
        evaluate_util.print_results(None, None, test_results)
        print('-'*18)
        print("Exiting from training by using 0 % of OOD data")
        os._exit(0)

best_recall = -np.inf
best_ood_recall = -np.inf
best_epoch = 0
best_ood_epoch = 0
best_valid_results = None
best_test_results = None
best_ood_test_results = None
update_count = 0

# recall@10 for best model selection when K=0, recall@50 when K=2
K = 0 if args.dataset == 'synthetic' else 2
evaluate_interval = 1 if args.dataset=='yelp' or args.ood_finetune else 5

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        adjust_lr(epoch)
        train()
        if epoch % evaluate_interval == 0:

            valid_loss, valid_results = evaluate(valid_x_data, valid_y_data, mask_val, user_feature, eval(args.topN))
            test_loss, test_results = evaluate(test_x_data, test_y_data, mask_test, user_feature, eval(args.topN))

            print('---'*18)
            print("Runing Epoch {:03d} ".format(epoch) + 'valid loss {:.4f}'.format(valid_loss) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-epoch_start_time)))
            evaluate_util.print_results(None, valid_results, test_results)
            print('---'*18)
            
            if args.ood_test:
                ood_val_loss, ood_valid_results = evaluate(valid_x_data, ood_valid_y_data, mask_test,\
                                                           ood_user_feature, eval(args.topN), args.CI)
                ood_test_loss, ood_test_results = evaluate(test_x_data, ood_test_y_data, mask_test, \
                                                           ood_user_feature, eval(args.topN), args.CI)
                print(f'{ood_test_dataset} testing')
                if args.ood_finetune:
                    ood_valid_results=None
                evaluate_util.print_results(None, ood_valid_results, ood_test_results)
                print('---'*18)

            # Save the model if recall is the best we've seen so far.
            if valid_results[1][K] > best_recall: # recall@10 for selection
                best_recall, best_epoch = valid_results[1][K], epoch
                best_test_results = test_results
                best_valid_results = valid_results
                if args.ood_test:
                    best_ood_test_results = ood_test_results
                if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                if not args.ckpt is None: # save finetuning model
                    torch.save(model, '{}{}_{}_{}_{}_{}_{}_{}_{}_{}_{}lr_{}wd_{}bs_{}anneal_{}cap_{}CI_{}drop_{}_{}_{}_{}bn_{}freq_{}reg_{}_{}.pth'.format(
                            args.save_path, args.model_name, args.dataset, train_dataset ,ckpt_structure[-20], ckpt_structure[-19], ckpt_structure[-18], \
                            ckpt_structure[-17], ckpt_structure[-16], ckpt_structure[-15], args.lr, args.wd,\
                            args.batch_size, args.total_anneal_steps, args.anneal_cap, args.CI, \
                            args.dropout, ckpt_structure[-7], ckpt_structure[-6], ckpt_structure[-5], args.bn, args.sample_freq, args.regs, ckpt_structure[-2], args.log_name))                   
                else:
                    torch.save(model, '{}{}_{}_{}_{}q_{}p11_{}p12_{}p2_{}p3_{}lr_{}wd_{}bs_{}anneal_{}cap_{}CI_{}drop_{}Z1_{}E2_{}Z2_{}bn_{}freq_{}reg_{}_{}.pth'.format(
                            args.save_path, args.model_name, args.dataset, train_dataset , args.mlp_dims, args.mlp_p1_1_dims, \
                            args.mlp_p1_2_dims, args.mlp_p2_dims, args.mlp_p3_dims, args.lr, args.wd,\
                            args.batch_size, args.total_anneal_steps, args.anneal_cap, args.CI, \
                            args.dropout, args.Z1_hidden_size, args.E2_hidden_size, args.Z2_hidden_size, args.bn, args.sample_freq, args.regs, args.act_function, args.log_name))
except KeyboardInterrupt:
    print('-'*18)
    print('Exiting from training early')

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_util.print_results(None, best_valid_results, best_test_results)
print('==='*18)
if args.ood_test:
    print(f"End. {ood_test_dataset} Performance")
    evaluate_util.print_results(None, None, best_ood_test_results)
    print('==='*18)



