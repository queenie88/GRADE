import numpy as np
import random
import torch,os
os.sys.path.append("DefRec_and_PCM")
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from PointDA.data.dataloader import ScanNet, ModelNet, ShapeNet, label_to_idx
from PointDA.Models import PointNet, DGCNN, Pct
from PointDA.Models_trs import DGCNN_trs3_1 as DGCNN_trs
from PointDA.Models_trs import DGCNN_trs3 as DGCNN_trs0
from PointDA.Models_trs import DGCNN_trs4 as DGCNN_trs2
from utils import pc_utils
from DefRec_and_PCM_1 import DefRec, PCM,serialization
import os.path as osp
import ipdb
NWORKERS=4
MAX_LOSS = 9 * (10**9)

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='Dgcnn',  help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments_test', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=150, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn','pct','dgcnn_trs','dgcnn_trs0',
                                                                   'dgcnn_trs2'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
parser.add_argument('--apply_PCM', type=str2bool, default=False, help='Using mixup in source')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--pre_model', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--pre_model2', type=str, default='', metavar='N',
                    help='Pretrained model2 path')
parser.add_argument('--alpha', type=float, default=0.999)
parser.add_argument('--DefRec_SCALER', type=float, default=0.001,help='for defrec loss')
parser.add_argument('--ce_soft_weight', type=float, default=0.2,help='for defrec loss')

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
# device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
# if torch.cuda.is_available():
#     torch.cuda.set_device(int(args.gpus[0]))
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


### soft loss
class SoftEntropy(nn.Module):
	def __init__(self):
		super(SoftEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		loss = (- nn.functional.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
		return loss
# ==================
##### discrepancy loss ##
def discrepancy(out1, out2):
    """discrepancy loss"""
    out = torch.mean(torch.abs(nn.functional.softmax(out1, dim=-1) - nn.functional.softmax(out2, dim=-1)))
    return out
# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

# src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

# src_trainset = data_func[src_dataset](io, args.dataroot, 'train',multi=True)
trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train',multi=True)
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')

# Creating data indices for training and validation splits:
# src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
# src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
#                                sampler=src_train_sampler, drop_last=True)
# src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
#                              sampler=src_valid_sampler)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                                sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                                  sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
if args.model == 'pointnet':
    model = PointNet(args)
    model_ema = PointNet(args)
    model2 = PointNet(args)
    model_ema2 = PointNet(args)
elif args.model == 'dgcnn':
    model = DGCNN(args)
    model_ema = DGCNN(args)
    model2 = DGCNN(args)
    model_ema2 = DGCNN(args)
elif args.model == 'pct':
    model = Pct(args)
    model_ema = Pct(args)
    model2 = Pct(args)
    model_ema2 = Pct(args)
elif args.model == 'dgcnn_trs':
    model = DGCNN_trs(args)
    model_ema = DGCNN_trs(args)
    model2 = DGCNN_trs(args)
    model_ema2 = DGCNN_trs(args)
elif args.model == 'dgcnn_trs0':
    model = DGCNN_trs0(args)
    model_ema = DGCNN_trs0(args)
    model2 = DGCNN_trs0(args)
    model_ema2 = DGCNN_trs0(args)
elif args.model == 'dgcnn_trs2':
    model = DGCNN_trs2(args)
    model_ema = DGCNN_trs2(args)
    model2 = DGCNN_trs2(args)
    model_ema2 = DGCNN_trs2(args)
else:
    raise Exception("Not implemented")

model = model.to(device)
model_ema =model_ema.to(device)
model2 = model2.to(device)
model_ema2 =model_ema2.to(device)

if args.pre_model !='' :
    initial_weights = serialization.load_checkpoint(args.pre_model)
    serialization.copy_state_dict(initial_weights, model)
    serialization.copy_state_dict(initial_weights, model_ema)
    model_ema.C.mlp3.weight.data.copy_(model.C.mlp3.weight.data)

if args.pre_model2 !='' :
    initial_weights = serialization.load_checkpoint(args.pre_model2)
    serialization.copy_state_dict(initial_weights, model2)
    serialization.copy_state_dict(initial_weights, model_ema2)
    model_ema2.C.mlp3.weight.data.copy_(model2.C.mlp3.weight.data)
# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    model2 = nn.DataParallel(model2)
    model_ema2 = nn.DataParallel(model_ema2)

for param in model_ema.parameters():
    param.detach_()

for param in model_ema2.parameters():
    param.detach_()

best_model = copy.deepcopy(model_ema)

# ==================
# Optimizer
# ==================
params =[]
if args.optimizer == "SGD":
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr,"momentum": args.momentum, "weight_decay": args.wd}]
    for key, value in model2.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "momentum": args.momentum, "weight_decay": args.wd}]
else:
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.wd}]
    for key, value in model2.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.wd}]
opt = optim.SGD(params) if args.optimizer == "SGD" \
    else optim.Adam(params)
scheduler = CosineAnnealingLR(opt, args.epochs)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
criterion2=  SoftEntropy()
criterion3 = nn.MSELoss()
# lookup table of regions means
lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

#=================
# update ema model
#=================
def _update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):

    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        if partition=="Val":
            for data,_, labels in test_loader:
                data, labels = data.to(device), labels.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                if args.model == 'pct':
                    logits = model(data)
                else:
                    logits = model(data, activate_DefRec=False)
                loss = criterion(logits["cls"], labels)
                print_losses['cls'] += loss.item() * batch_size

                # evaluation metrics
                preds = logits["cls"].max(dim=1)[1]
                test_true.append(labels.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                count += batch_size
                batch_idx += 1
        else:
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                if args.model == 'pct':
                    logits = model(data)
                else:
                    logits = model(data, activate_DefRec=False)
                loss = criterion(logits["cls"], labels)
                print_losses['cls'] += loss.item() * batch_size

                # evaluation metrics
                preds = logits["cls"].max(dim=1)[1]
                test_true.append(labels.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                count += batch_size
                batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat

# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
best_model = io.save_model(model_ema)

for epoch in range(args.epochs):
    model.train()
    model2.train()
    model_ema.train()
    model_ema2.train()

    # init data structures for saving epoch stats
    cls_type = 'mixup' if args.apply_PCM else 'cls'
    src_print_losses = {"total": 0.0, cls_type: 0.0}
    if args.DefRec_on_src:
        src_print_losses['DefRec'] = 0.0
    trgt_print_losses = {'DefRec': 0.0}
    src_count = trgt_count = 0.0

    batch_idx = 1
    for ki,data in enumerate(trgt_train_loader):
        opt.zero_grad()

        #### target data ####
        if data is not None:
            trgt_data, trgt_data2,trgt_label = data[0].to(device), data[1].to(device),data[2].to(device).squeeze()
            trgt_data = trgt_data.permute(0, 2, 1)
            trgt_data2 = trgt_data2.permute(0, 2, 1)
            batch_size = trgt_data.size()[0]
            trgt_data_orig = trgt_data.clone()
            trgt_data_orig2 = trgt_data2.clone()
            device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")

            # trgt_data, trgt_mask = DefRec.deform_input(trgt_data, lookup, args.DefRec_dist, device)
            trgt_logits = model(trgt_data, activate_DefRec=True)
            trgt_logits_ema = model_ema(trgt_data, activate_DefRec=True)
            preds1 = trgt_logits_ema["cls"].max(dim=1)[1]

            # trgt_data2, trgt_mask2 = DefRec.deform_input(trgt_data2, lookup, args.DefRec_dist, device)
            trgt_logits2 = model2(trgt_data2, activate_DefRec=True)
            trgt_logits_ema2 = model_ema2(trgt_data2, activate_DefRec=True)
            preds2 = trgt_logits_ema2["cls"].max(dim=1)[1]

            loss_C1 = criterion(trgt_logits["cls"], preds1)
            loss_C2 = criterion(trgt_logits2["cls"], preds2)

            loss_CS1 = criterion2(trgt_logits["cls"], trgt_logits_ema2["cls"])
            loss_CS2 = criterion2(trgt_logits2["cls"], trgt_logits_ema["cls"])

            loss = DefRec.calc_loss4(args, trgt_logits, trgt_data_orig )
            loss2 = DefRec.calc_loss4(args, trgt_logits2, trgt_data_orig2)

            tr_loss_adv = criterion3(trgt_logits['DefRec'], trgt_logits_ema['DefRec'])
            tr_loss_adv2 = criterion3(trgt_logits2['DefRec'], trgt_logits_ema2['DefRec'])

            # tr_loss_adv = criterion3(trgt_logits['DefRec'], trgt_logits_ema2['DefRec'])
            # tr_loss_adv2 = criterion3(trgt_logits2['DefRec'], trgt_logits_ema['DefRec'])
            # tr_loss_adv = - 1 * discrepancy(trgt_logits['DefRec'], trgt_logits_ema['DefRec'])
            loss = (loss_C1+loss_C2)*(1-args.ce_soft_weight)+(loss_C1+loss_C2)*args.ce_soft_weight+\
                   loss+loss2+tr_loss_adv+tr_loss_adv2
            trgt_print_losses['DefRec'] += loss.item() * batch_size
            loss.backward()
            trgt_count += batch_size

        _update_ema_variables(model, model_ema, args.alpha, epoch * len(trgt_train_loader) + batch_idx)
        _update_ema_variables(model2, model_ema2, args.alpha, epoch * len(trgt_train_loader) + batch_idx)
        opt.step()
        batch_idx += 1

    scheduler.step()


    # print progress
    # src_print_losses = {k: v * 1.0 / src_count for (k, v) in src_print_losses.items()}
    # src_acc = io.print_progress("Source", "Trn", epoch, src_print_losses)
    trgt_print_losses = {k: v * 1.0 / trgt_count for (k, v) in trgt_print_losses.items()}
    trgt_acc = io.print_progress("Target", "Trn", epoch, trgt_print_losses)

    #===================
    # Validation
    #===================
    trgt_val_acc, trgt_val_loss, trgt_conf_mat = test(trgt_val_loader, model_ema, "Target", "Val", epoch)
    trgt_val_acc2, trgt_val_loss2, trgt_conf_mat2 = test(trgt_val_loader, model_ema2, "Target", "Val", epoch)

    # save model according to best source model (since we don't have target labels)
    if trgt_val_acc > trgt_val_acc2:
        if  trgt_val_acc > trgt_best_val_acc:
            trgt_best_val_acc = trgt_val_acc
            trgt_best_val_loss = trgt_val_loss
            best_val_epoch = epoch
            best_epoch_conf_mat = trgt_conf_mat
            best_model = io.save_model(model_ema)
    else:
        if  trgt_val_acc2 > trgt_best_val_acc:
            trgt_best_val_acc = trgt_val_acc2
            trgt_best_val_loss = trgt_val_loss2
            best_val_epoch = epoch
            best_epoch_conf_mat = trgt_conf_mat2
            best_model = io.save_model(model_ema2)

io.cprint("Best model was found at epoch %d, "
          "target validation accuracy: %.4f, target validation loss: %.4f"
          % (best_val_epoch, trgt_best_val_acc, trgt_best_val_loss))
io.cprint("Best validtion model confusion matrix:")
io.cprint('\n' + str(best_epoch_conf_mat))

#===================
# Test
#===================
model = best_model
trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_best_val_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))