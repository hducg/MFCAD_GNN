import os
import os.path as osp
this_dir = osp.dirname(osp.realpath(__file__))
import sys
sys.path.append(osp.join(this_dir, '..'))
import time
import numpy as np
import logging
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import mean_iou

from data_loader import FaceNet, RandomTransform
import net
import utils
from torch_utils import save_checkpoint

# =============================================================================
# arguments 
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', 
                    default=osp.join(this_dir, '../experiments/gnn-ipt/'), 
                    help="Directory containing the data list, loss, log and checkpoint files")
args = parser.parse_args()
root = osp.join(this_dir, '..', 'data')

# =============================================================================
# set logging
# =============================================================================
exp_path = args.exp_dir
if not osp.exists(exp_path):
    os.mkdir(exp_path)
utils.set_logger(osp.join(exp_path, 'train.log'))


# =============================================================================
# load parameters
# =============================================================================
json_path = os.path.join(args.exp_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

loss_train = []
def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    loss_list = []
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_list.append(loss.item())
        correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 100 == 0:
            logging.info('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
                i + 1, len(train_loader), total_loss / 100,
                correct_nodes / total_nodes))
            total_loss = correct_nodes = total_nodes = 0
    loss_train.append(sum(loss_list)/len(loss_list))

        
loss_valid = []
best = 0
def test(loader):
    model.eval()
    correct_nodes = total_nodes = 0
    ious = []
    loss_list = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss_list.append(loss.item())
        pred = out.max(dim=1)[1]
        correct_nodes += pred.eq(data.y).sum().item()
        ious += [mean_iou(pred, data.y, params.nClassesTotal, data.batch)]
        total_nodes += data.num_nodes
    loss_valid.append(sum(loss_list)/len(loss_list))
    acc = correct_nodes / total_nodes
    save_checkpoint({'state_dict':model.state_dict()}, acc>best, exp_path)
    return acc, torch.cat(ious, dim=0).mean().item()


# =============================================================================
# load dataset
# =============================================================================
transform = RandomTransform(10.0)
train_dataset = FaceNet(root, osp.join(exp_path, 'train_list'), transform)
test_dataset = FaceNet(root, osp.join(exp_path, 'valid_list'), transform)
train_loader = DataLoader(
    train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
test_loader = DataLoader(
    test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
logging.info(str(len(train_dataset)) + ' training data; ' + 
             str(len(test_dataset)) + ' testing data')


# =============================================================================
# net, optimizer, hyperparameters
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net.Net(params.nClassesTotal).to(device)
logging.info('#parameters {}'.format(sum([x.nelement() for x in model.parameters()])))
optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
       

# =============================================================================
# training
# =============================================================================
for epoch in range(1, 1 + params.num_epochs):
    start = time.time()
    train()
    end = time.time()
    logging.info('training time {:.4f}'.format(end - start))
    acc, iou = test(test_loader)        
    logging.info('Epoch: {:02d}, Acc: {:.4f}, IoU: {:.4f}'.format(epoch, acc, iou))
    scheduler.step()

np.save(osp.join(exp_path, 'train_loss'), loss_train)
np.save(osp.join(exp_path, 'val_loss'), loss_valid)
