import os
import glob
import numpy as np

def split_dataset(root, ext=None, shuffle=False, split=0.7):
    """
    Load the dataset folder structure and generate
    the split of the dataset (train, valid, test).

    Parameters
    ----------
    root : string
        Contains a set of directories, one per class
    ext : string (default: None)
        Extension of the files; if None, it is assumed the class folder contains only image files
    shuffle : boolean (default: False)
        Shuffle the per-class file list
    split : float (default: 0.7)
        Trainset size
    """

    regex = os.path.join(root, "*")
    cpaths = sorted([x for x in glob.glob(regex) if os.path.isdir(x)])

    data = []
    train = []
    valid = []
    test = []

    if ext is None:
        ext = ""

    for i,c in enumerate(cpaths):
        fpaths = sorted([x for x in glob.glob(os.path.join(c,"*" + ext))])
        for j,x in enumerate(fpaths):
            x = x[len(root):]
            fpaths[j] = x if x[0] != "/" else x[1:]
        nSamples = len(fpaths)
        assert nSamples > 0
        for x in fpaths:
            data.append([x,i])
        if shuffle:
            idx = np.random.permutation(nSamples)
            fpaths = np.asarray(fpaths)[idx].tolist()
            trSize = int(nSamples * split)
            vlSize = (nSamples - trSize) // 2
            tsSize = (nSamples - trSize - vlSize)
            for x in fpaths[:trSize]:
                train.append([x,i])
            for x in fpaths[trSize:-tsSize]:
                valid.append([x,i])
            for x in fpaths[-tsSize:]:
                test.append([x,i])
        else:
            idx = []
            for j in range(nSamples):
                k = round(j/(1 - split))
                if k < nSamples:
                    idx.append(k)
                else:
                    break
            trIdx = [j for j in np.arange(nSamples) if j not in idx]
            vlIdx = idx[0::2]
            tsIdx = idx[1::2]
            fpaths = np.asarray(fpaths)
            for x in fpaths[trIdx]:
                train.append([x,i])
            for x in fpaths[vlIdx]:
                valid.append([x,i])
            for x in fpaths[tsIdx]:
                test.append([x,i])

    data = sorted(data)
    train = sorted(train)
    valid = sorted(valid)
    test = sorted(test)

    np.savetxt(os.path.join(root, "data.txt"), data, fmt="%s")
    np.savetxt(os.path.join(root, "train.txt"), train, fmt="%s")
    np.savetxt(os.path.join(root, "valid.txt"), valid, fmt="%s")
    np.savetxt(os.path.join(root, "test.txt"), test, fmt="%s")

if __name__ == "__main__":

    root = "/media/Data/userdata/mcarletti/pointnet/pts"
    filenames = sorted(glob.glob(os.path.join(root, "tmp/pts_fixed+augmented+vandal+linemod+partial", "*.pts")))

    npts = 2500
    destfolder = os.path.join(root, "pointclouds.partial")
    nptc = float("inf")
    valid_fnames = []

    print("Preparing dataset (n.files: {})".format(len(filenames)))
    for i,fn in enumerate(filenames):
        if i % 100 == 0:
            print("{:6.2f}".format(100 * (i+1) / len(filenames)), fn)
        data = np.loadtxt(fn)
        N = data.shape[0]
        if N > 10000:
            valid_fnames.append(fn)
            nptc = int(np.minimum((N // npts), nptc))
    
    print("n.pts:", nptc)
    assert nptc > 0

    filenames = valid_fnames

    for i,fn in enumerate(filenames):
        data = np.loadtxt(fn)
        N = data.shape[0]
        if i % 50 == 0:
            print("{:6.2f}".format(100 * (i+1) / len(filenames)), data.shape, fn)
        for i in range(nptc):
            ids = np.random.choice(range(N), npts, False)
            pts = data[ids]
            # cn = os.path.basename(fn).split(".")[0][:-3]
            cn = os.path.basename(fn).split("_")[0]
            out = os.path.join(destfolder, cn)
            os.makedirs(out, exist_ok=True)
            n = len(glob.glob(os.path.join(out, "*.pts")))
            np.savetxt(os.path.join(out, "{:03d}.pts".format(n)), pts)

    split_dataset(destfolder, ".pts")
