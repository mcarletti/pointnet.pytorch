# example:
# python3 test_classifier.py --model <pthfilepath> --pcd <ptsfilepath> [--nclasses (default: 7)]

import argparse
import os
import glob
import numpy as np
import torch
from pointnet.model import PointNetCls
from lib.utils.io import load


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='model path')
parser.add_argument('--nclasses', type=int, default=7, help='number of classes')
parser.add_argument('--pcd', type=str, help='sample point cloud')
parser.add_argument('--target', type=int, default=-1, help='sample point cloud')


opt = parser.parse_args()
print(opt)

classifier = PointNetCls(k=opt.nclasses)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

if os.path.isdir(opt.pcd):
    # filenames = [fn for i,fn in enumerate(sorted(glob.glob(os.path.join(opt.pcd, "obj_*.pts")))) if i in [3,4,7,10,12,13,14]]
    filenames = [fn for i,fn in enumerate(sorted(glob.glob(os.path.join(opt.pcd, "obj_*.ply")))) if i in [3,4,7,10,12,13,14]]
    targets = [0,1,2,3,4,5,6]
else:
    filenames = [opt.pcd]
    targets = [opt.target]

assert len(filenames) > 0

ntests = 1000

for i,fn in enumerate(filenames):

    # points_orig = np.loadtxt(fn).astype(np.float32)
    points_orig = load(fn)["vertices"].astype(np.float32)

    accuracy = 0

    for j in range(ntests):

        points = points_orig - np.min(points_orig,axis=0)
        points = points / np.max(points)
        points = 2 * points - 1
        theta = np.random.uniform(0,np.pi*2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        points[:,[0,2]] = points[:,[0,2]].dot(rotation_matrix) # random rotation
        points += np.random.normal(0, 0.02, size=points.shape) # random jitter

        ids = np.random.choice(np.arange(points.shape[0]), 2500, replace=False)
        points = points[ids]
        points = points - np.min(points,axis=0)
        points = points / np.max(points)
        points = 2 * points - 1

        points = torch.from_numpy(points).unsqueeze(0)
        points = points.transpose(2, 1)
        points = points.cuda()
        pred, _, _ = classifier(points)
        pred = torch.exp(pred)
        pred = pred.cpu().data.numpy().squeeze()

        # print("Winner:", np.argmax(pred))
        # print("Output:", pred)

        if np.argmax(pred) == targets[i]:
            accuracy += 1./ntests

    print(fn, "Accuracy: {: 3d}".format(int(accuracy * 100 + 0.5)))

