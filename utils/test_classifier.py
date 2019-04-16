# example:
# python3 test_classifier.py --model <pthfilepath> --pcd <ptsfilepath> [--nclasses (default: 7)]

import argparse
import numpy as np
import torch
from pointnet.model import PointNetCls


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='model path')
parser.add_argument('--nclasses', type=int, default=7, help='number of classes')
parser.add_argument('--pcd', type=str, help='sample point cloud')


opt = parser.parse_args()
print(opt)

classifier = PointNetCls(k=opt.nclasses)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

points = np.loadtxt(opt.pcd).astype(np.float32)
points = torch.from_numpy(points).unsqueeze(0)
points = points.transpose(2, 1)
points = points.cuda()
pred, _, _ = classifier(points)
pred = torch.exp(pred)
pred = pred.cpu().data.numpy().squeeze()

print("Winner:", np.argmax(pred))
print("Output:", pred)

