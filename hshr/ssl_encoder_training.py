import argparse
import copy
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from FEATURES.DATABASE.CONST import EXPERIMENTS
from utils.data_utils import check_dir, get_files_type
from utils.evaluate import Evaluator
from self_supervision.call import get_moco
from utils.feature import min_max_binarized
from utils.model.base_model import HashEncoder, AttenHashEncoder
import numpy as np


class PairCenterDataset(Dataset):

    def __init__(self, cluster_dir, tmp, single=False, subtypes: list = None) -> None:
        super().__init__()
        cluster_list_0 = get_files_type(cluster_dir, 'clu_0.npy', tmp)
        cluster_list_0 = [x[1:] for x in cluster_list_0]
        cluster_list_0.sort()
        cluster_list_1 = get_files_type(cluster_dir, 'clu_1.npy', tmp)
        cluster_list_1 = [x[1:] for x in cluster_list_1]
        set1 = set(cluster_list_1)
        self.centers = []
        for f0 in cluster_list_0:
            f1 = f0.replace('clu_0.npy', 'clu_1.npy')
            if f1 not in set1:
                continue
            path = os.path.dirname(f0)
            if subtypes is not None and path.split("/")[-2] not in subtypes:
                continue

            f0, f1 = os.path.join(cluster_dir, f0), os.path.join(cluster_dir, f1)
            c0 = np.load(f0)
            if single:
                self.centers.append((c0, path))
            else:
                c1 = np.load(f1)
                self.centers.append((c0, c1, path))

    def __getitem__(self, idx):
        return self.centers[idx]

    def __len__(self) -> int:
        return len(self.centers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess raw WSIs")
    parser.add_argument("--RESULT_DIR", type=str, required=True, help="A path to save your preprocessed results.")
    parser.add_argument("--TMP", type=str, required=True, help="The path to save some necessary tmp files.")
    parser.add_argument("--MODEL_DIR", type=str, required=True, help="The path of ssl hash encoder model.")
    parser.add_argument("--DATASETS", type=str, nargs='+', required=False, help="A list of datasets.")
    args = parser.parse_args()

    feature_in = 512
    feature_out = 1024
    depth = 1
    lr = 0.003
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 128  # 128
    num_cluster = 20
    gamma = 0.99
    num_epoch = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().cuda(True)
    train_dataset = PairCenterDataset(args.RESULT_DIR, args.TMP)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    base_model = lambda: AttenHashEncoder(feature_in, feature_out, depth)
    model = get_moco(base_model(), base_model(), device, feature_out)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    evaluator = Evaluator()
    exps = []
    for name in args.DATASETS:
        valid_dataset = PairCenterDataset(args.RESULT_DIR, args.TMP, False, EXPERIMENTS[name])
        cfs = []
        cf_paths = []
        for c1, _, path in valid_dataset.centers:
            cfs.append(c1)
            cf_paths.append(path)
        cfs = np.array(cfs)
        exps.append((cfs, cf_paths, name))

    print('without hash encoder:')
    for cfs, cf_paths, name in exps:
        raw = min_max_binarized(cfs)
        evaluator.reset()
        evaluator.add_patches(raw, cf_paths)
        acc, ave = evaluator.eval()
        print(name, ave, acc)

    for epoch in range(num_epoch):
        print('*' * 5, 'epoch: ', epoch, '*' * 5)
        loss_sum = 0
        loss_count = 0
        pre_model = copy.deepcopy(model)
        for x0, x1, _ in train_dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            output, target = model(x0, x1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            loss_count += 1
        scheduler.step()
        loss_ave = loss_sum / loss_count
        print("loss: ", loss_ave)

        if epoch % 10 == 0 or epoch == num_epoch - 1:
            with torch.no_grad():
                for cfs, cf_paths, name in exps:
                    evaluator.reset()
                    cfs = torch.from_numpy(np.array(cfs)).to(device)
                    raw = cfs

                    h, w = pre_model.encoder_q(raw, no_pooling=True, weight=True)
                    evaluator.add_patches(h.cpu().detach().numpy(), cf_paths)
                    evaluator.add_weight(w.cpu().detach().numpy())

                    acc, ave = evaluator.eval()
                    print(name, ave, acc)

    label = 'ssl_att'
    torch.save(model.encoder_q.state_dict(), check_dir(os.path.join(args.MODEL_DIR, label, 'model_best.pth')))
