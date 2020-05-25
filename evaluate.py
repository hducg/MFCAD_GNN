import os.path as osp
import os
this_dir = osp.dirname(osp.realpath(__file__))
import sys
sys.path.append(osp.join(this_dir, '..'))
from torch_utils import load_checkpoint
import pickle
import argparse
import logging

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import mean_iou

from data_loader import load_dataset, RandomTransform  
import net
import utils
    
def test(loader):
    model.eval()
    correct_nodes = total_nodes = 0
    ious = []
    loss_list = []
    for i, data in enumerate(loader):
        path = path_list[i]
        
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss_list.append(loss.item())
        pred = out.max(dim=1)[1]               
        path = osp.join(args.out_dir, path.split(os.sep)[-1].replace('graph', 'face_gnn'))
        with open(path, 'wb') as file:
            pickle.dump(pred.cpu().numpy(), file)
            
        correct_nodes += pred.eq(data.y).sum().item()
        ious += [mean_iou(pred, data.y, params.nClassesTotal, data.batch)]
        total_nodes += data.num_nodes
    acc = correct_nodes / total_nodes
    return acc, torch.cat(ious, dim=0).mean().item()


# =============================================================================
# arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', 
                    default=osp.join(osp.dirname(osp.realpath(__file__)), 
                                     '../experiments/test/'), 
                    help="Directory containing the step files")
parser.add_argument('--exp_dir', 
                    default=osp.join(osp.dirname(osp.realpath(__file__)), 
                                     '../experiments/gnn-ipt'), 
                    help="Directory containing the step files")
args = parser.parse_args()

# =============================================================================
# logging
# =============================================================================
utils.set_logger(os.path.join(args.exp_dir, 'evaluate.log'))

# =============================================================================
# hyperparameters
# =============================================================================
json_path = os.path.join(args.exp_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

# =============================================================================
# dataset
# =============================================================================
transform = RandomTransform(10.0)
data_path = osp.join(args.exp_dir, 'test_list')
assert osp.isfile(data_path), "No test list file found at {}".format(data_path)
test_dataset, path_list = load_dataset(data_path, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
logging.info('{} testing data'.format(len(test_dataset)))

# =============================================================================
# model
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net.Net(params.nClassesTotal).to(device)
      
load_checkpoint(osp.join(args.exp_dir, 'best.pth.tar'), model)
        
acc, iou = test(test_loader)        
logging.info('Acc: {:.4f}, IoU: {:.4f}'.format(acc, iou))

