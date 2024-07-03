import numpy as np
import random
import torch,os
# os.sys.path.append("DefRec_and_PCM")
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import argparse
import copy
import utils.log
from torchsummary import summary
from PointSegDA.data.dataloader import datareader
from PointSegDA.Models import DGCNN_DefRec
from PointSegDA.Models_trs import DGCNN_DefRec_tr as DGCNN_DefRec_tr
from PointSegDA.Models_trs import DGCNN_DefRec_tr2 as DGCNN_DefRec_tr2
from utils import pc_utils
from sklearn.metrics import jaccard_score
from DefRec_and_PCM_1 import DefRec, PCM, serialization
import ipdb
from PointSegDA.data import IterLoader
######
##---mutual mean teacher with target
######
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
parser.add_argument('--exp_name', type=str, default='DefRec_PCM_t',  help='Name of the experiment')
parser.add_argument('--dataroot', type=str, default='./data/PointSegDAdataset', help='data path')
parser.add_argument('--out_path', type=str, default='./experiments_test', help='log folder path')
parser.add_argument('--src_dataset', type=str, default='adobe', choices=['adobe', 'faust', 'mit', 'scape'])
parser.add_argument('--trgt_dataset', type=str, default='faust', choices=['adobe', 'faust', 'mit', 'scape'])
parser.add_argument('--model', type=str, default='dgcnn', choices=['dgcnn','dgcnn_trs','dgcnn_trs2'], help='Model to use')
parser.add_argument('--epochs', type=int, default=80, help='number of episode to train')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_radius', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--radius', type=float, default=0.3, help='radius of the ball for reconstruction')
parser.add_argument('--min_pts', type=int, default=20, help='minimum number of points per region')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--noise_std', type=float, default=0.1, help='learning rate')
parser.add_argument('--DefRec_weight', type=float, default=0.05, help='weight of the DefRec loss')
parser.add_argument('--soft_weight', type=float, default=0.05, help='weight of the DefRec loss')
parser.add_argument('--lamda1', type=float, default=0.2, help='weight of the DefRec loss')
parser.add_argument('--lamda2', type=float, default=0.1, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--pre_model', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--pre_model2', type=str, default='', metavar='N',
                    help='Pretrained model path2')
parser.add_argument('--iters', type=int, default=50, help='data load  iter numbers')
parser.add_argument('--alpha', type=float, default=0.999)

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
trgt_trainset = datareader(args.dataroot, dataset=args.trgt_dataset, partition='train', domain='target',multi=True)
trgt_valset = datareader(args.dataroot, dataset=args.trgt_dataset, partition='val', domain='target')
trgt_testset = datareader(args.dataroot, dataset=args.trgt_dataset, partition='test', domain='target')

# dataloaders for source and target
batch_size = min(len(trgt_trainset), args.batch_size)

trgt_train_loader = IterLoader(DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=batch_size,
                               shuffle=True, drop_last=True),length=args.iters)
trgt_val_loader = DataLoader(trgt_valset, num_workers=NWORKERS, batch_size=args.test_batch_size)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
num_classes = 8
if args.model == 'dgcnn':
    model = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)
    model_ema = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)

    model2 = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)
    model_ema2 = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)
elif args.model == 'dgcnn_trs':
    model = DGCNN_DefRec_tr(args, in_size=3, num_classes=num_classes)
    model_ema = DGCNN_DefRec_tr(args, in_size=3, num_classes=num_classes)

    model2 = DGCNN_DefRec_tr(args, in_size=3, num_classes=num_classes)
    model_ema2 = DGCNN_DefRec_tr(args, in_size=3, num_classes=num_classes)
elif args.model == 'dgcnn_trs2':
    model = DGCNN_DefRec_tr2(args, in_size=3, num_classes=num_classes)
    model_ema = DGCNN_DefRec_tr2(args, in_size=3, num_classes=num_classes)

    model2 = DGCNN_DefRec_tr2(args, in_size=3, num_classes=num_classes)
    model_ema2 = DGCNN_DefRec_tr2(args, in_size=3, num_classes=num_classes)

# model = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)
# model_ema = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)
#
# model2 = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)
# model_ema2 = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)

summary(model, input_size=(3, 2048), device='cpu')
summary(model_ema, input_size=(3, 2048), device='cpu')
summary(model2, input_size=(3, 2048), device='cpu')
summary(model_ema2, input_size=(3, 2048), device='cpu')

model = model.to(device)
model_ema =model_ema.to(device)
model2 = model2.to(device)
model_ema2 =model_ema2.to(device)

if args.pre_model !='' :
    initial_weights = serialization.load_checkpoint(args.pre_model)
    serialization.copy_state_dict(initial_weights, model)
    serialization.copy_state_dict(initial_weights, model_ema)
    model_ema.seg.conv4.weight.data.copy_(model.seg.conv4.weight.data)
if args.pre_model2 !='' :
    initial_weights = serialization.load_checkpoint(args.pre_model2)
    serialization.copy_state_dict(initial_weights, model2)
    serialization.copy_state_dict(initial_weights, model_ema2)
    model_ema2.seg.conv4.weight.data.copy_(model2.seg.conv4.weight.data)

for param in model.parameters():
    param.requires_grad = False

for param in model2.parameters():
    param.requires_grad = False

for param in model.seg.parameters():
    param.requires_grad = True
for param in model2.seg.parameters():
    param.requires_grad = True
for param in model.DefRec.parameters():
    param.requires_grad = True
for param in model2.DefRec.parameters():
    param.requires_grad = True
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
t_max = args.epochs
scheduler = CosineAnnealingLR(opt, T_max=t_max, eta_min=0.0)

# ==================
# Loss and Metrics
# ==================
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
sample_criterion = nn.CrossEntropyLoss(reduction='none')  # to get the loss per shape
criterion2=  SoftEntropy()
criterion3 = nn.MSELoss()

def seg_metrics(labels, preds):
    batch_size = labels.shape[0]
    mIOU = accuracy = 0
    for b in range(batch_size):
        y_true = labels[b, :].detach().cpu().numpy()
        y_pred = preds[b, :].detach().cpu().numpy()
        # IOU per class and average
        mIOU += jaccard_score(y_true, y_pred, average='macro')
        accuracy += np.mean(y_true == y_pred)
    return mIOU, accuracy

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
def test(test_loader, model=None):

    # Run on cpu or gpu
    seg_loss = mIOU = accuracy = 0.0
    batch_idx = num_samples = 0

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            data, labels = data[0].to(device), data[1].to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.shape[0]

            logits = model(data, make_seg=True, activate_DefRec=False)
            loss = criterion(logits["seg"].permute(0, 2, 1), labels)
            seg_loss += loss.item() * batch_size

            # evaluation metrics
            preds = logits["seg"].max(dim=2)[1]
            batch_mIOU, batch_seg_acc = seg_metrics(labels, preds)
            mIOU += batch_mIOU
            accuracy += batch_seg_acc

            num_samples += batch_size
            batch_idx += 1

    seg_loss /= num_samples
    mIOU /= num_samples
    accuracy /= num_samples
    model.train()
    model_ema.train()
    return seg_loss, mIOU, accuracy


# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_mIOU = trgt_best_val_mIOU = 0.0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
epoch = step = 0
lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

for epoch in range(args.epochs):
    trgt_train_loader.new_epoch()
    train_iters = len(trgt_train_loader)
    model.train()
    model_ema.train()
    model2.train()
    model_ema2.train()
    # init data structures for saving epoch stats
    # Run on cpu or gpu
    src_seg_loss = src_mIOU = src_accuracy = 0.0
    trgt_rec_loss = total_loss = 0.0
    batch_idx = src_count = trgt_count = 0

    for k in range(train_iters):
        step += 1
        opt.zero_grad()
        trgt_batch_loss = src_batch_loss = batch_mIOU = batch_seg_acc = 0.0
        data_t = trgt_train_loader.next()

        #### target data ####
        if data_t is not None:
            trgt_data,trgt_data2, trgt_labels = data_t[0].to(device), data_t[1].to(device),data_t[2].to(device)
            trgt_data = trgt_data.permute(0, 2, 1)
            trgt_data2 = trgt_data2.permute(0, 2, 1)
            batch_size = trgt_data.shape[0]
            trgt_data_orig = trgt_data.clone()
            trgt_data_orig2 = trgt_data2.clone()

            trgt_data, trgt_mask = DefRec.deform_input(trgt_data, lookup, args.DefRec_dist, device=device)
            trgt_logits = model(trgt_data, make_seg=True, activate_DefRec=True)
            trgt_logits_ema = model_ema(trgt_data, make_seg=True, activate_DefRec=True)
            preds1 = trgt_logits_ema['seg'].max(dim=2)[1]

            trgt_data2, trgt_mask2 = DefRec.deform_input(trgt_data2, lookup, args.DefRec_dist, device=device)
            trgt_logits2 = model2(trgt_data2, make_seg=True, activate_DefRec=True)
            trgt_logits_ema2 = model_ema2(trgt_data2, make_seg=True, activate_DefRec=True)
            preds2 = trgt_logits_ema2['seg'].max(dim=2)[1]

            loss_C1 = criterion(trgt_logits['seg'].permute(0, 2, 1), preds1)
            loss_C2 = criterion(trgt_logits2['seg'].permute(0, 2, 1), preds2)

            loss_CS1 = criterion3(trgt_logits["seg"], trgt_logits_ema["seg"])
            loss_CS2 = criterion3(trgt_logits2["seg"], trgt_logits_ema2["seg"])

            loss = DefRec.calc_loss(args, trgt_logits, trgt_data_orig, trgt_mask)
            loss2 = DefRec.calc_loss(args, trgt_logits2, trgt_data_orig2, trgt_mask2)
            tr_loss_adv = args.lamda2 * criterion3(trgt_logits['DefRec'], trgt_logits_ema['DefRec'])
            tr_loss_adv2 = args.lamda2 * criterion3(trgt_logits2['DefRec'], trgt_logits_ema2['DefRec'])
            # tr_loss_adv = - 1 * discrepancy(trgt_logits['DefRec'], trgt_logits_ema['DefRec'])
            loss = (loss_C1+loss_C2)*(1-args.soft_weight)+(loss_CS1+loss_CS2)*args.soft_weight+\
                   loss+loss2+tr_loss_adv+tr_loss_adv2
            trgt_batch_loss = loss.item()
            trgt_rec_loss += loss.item() * batch_size
            total_loss += loss.item() * batch_size
            loss.backward()

            trgt_count += batch_size

        _update_ema_variables(model, model_ema, args.alpha, step)
        _update_ema_variables(model2, model_ema2, args.alpha, step)
        opt.step()

    scheduler.step(epoch=epoch)
    # _update_ema_variables(model, model_ema, args.alpha,epoch * len(src_train_loader) + batch_idx)

    # print progress
    trgt_rec_loss /= trgt_count

    #===================
    # Validation
    #===================
    trgt_val_loss, trgt_val_miou, trgt_val_acc = test(trgt_val_loader,model_ema)
    trgt_val_loss2, trgt_val_miou2, trgt_val_acc2 = test(trgt_val_loader,model_ema2)

    # save model according to best source model (since we don't have target labels)
    if trgt_val_acc > trgt_val_acc2:
        if trgt_val_acc > trgt_best_val_acc:
            trgt_best_val_mIOU = trgt_val_miou
            trgt_best_val_acc = trgt_val_acc
            trgt_best_val_loss = trgt_val_loss
            best_val_epoch = epoch
            best_model = copy.deepcopy(model_ema)
    else:
        if trgt_val_acc2 > trgt_best_val_acc:
            trgt_best_val_mIOU = trgt_val_miou2
            trgt_best_val_acc = trgt_val_acc2
            trgt_best_val_loss = trgt_val_loss2
            best_val_epoch = epoch
            best_model = copy.deepcopy(model_ema2)

    io.cprint(f"Epoch: {epoch}, "
              f"Target train rec loss: {trgt_rec_loss:.5f}, ")

    io.cprint(f"Epoch: {epoch}, "
              f"Target val seg loss: {trgt_val_loss:.5f}, "
              f"Target val seg mIOU: {trgt_val_miou:.5f}, "
              f"Target val seg accuracy: {trgt_val_acc:.5f}")

    io.cprint(f"Epoch: {epoch}, "
              f"Target val seg loss2: {trgt_val_loss2:.5f}, "
              f"Target val seg mIOU2: {trgt_val_miou2:.5f}, "
              f"Target val seg accuracy2: {trgt_val_acc2:.5f}")

io.cprint("Best model was found at epoch %d\n"
          "target val seg loss: %.4f, target val seg mIOU: %.4f, target val seg accuracy: %.4f\n"
          % (best_val_epoch,
             trgt_best_val_loss, trgt_best_val_mIOU, trgt_best_val_acc))

#===================
# Test
#===================
model = best_model
trgt_test_loss, trgt_test_miou, trgt_test_acc = test(trgt_test_loader,model)
io.cprint("target test seg loss: %.4f, target test seg mIOU: %.4f, target test seg accuracy: %.4f"
          % (trgt_test_loss, trgt_test_miou, trgt_test_acc))