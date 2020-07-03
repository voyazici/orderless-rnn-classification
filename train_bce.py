import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from scipy.stats import logistic
from sklearn.metrics import precision_recall_fscore_support
from dataset import COCOMultiLabel
from model import Net, convert_weights
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import sys
import datetime

def adjust_learning_rate(optimizer, shrink_factor):
    print "DECAYING learning rate."
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print "The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'])

def my_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                              epoch * len(train_loader) + batch_idx)

def test(args, model, device, test_loader, threshold):
    model.eval()
    outputs = None
    labels = None
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_arr = output.data.cpu().numpy()
            output_arr = logistic.cdf(output_arr)
            junk = output_arr.copy()
            output_arr[output_arr >= threshold] = 1
            output_arr[output_arr < threshold] = 0
            if labels is None:
                labels = target.data.cpu().numpy()
                outputs = output_arr
            else:
                labels = np.concatenate((labels, target.data.cpu().numpy()),
                                        axis=0)
                outputs = np.concatenate((outputs, output_arr),
                                         axis=0)

    prec, recall, _, _ = precision_recall_fscore_support(outputs,
                                                         labels,
                                                         average='macro')
    f1 = 2 * prec * recall / (prec + recall)
    print('\nMACRO prec: {:.2f}, recall: {:.2f}, f1: {:.2f}\n'.format(
        100*recall, 100*prec, 100*f1))
    prec, recall, f1, _ = precision_recall_fscore_support(outputs,
                                                          labels,
                                                          average='micro')
    print('\nMICRO prec: {:.2f}, recall: {:.2f}, f1: {:.2f}\n'.format(
        100*recall, 100*prec, 100*f1))

    return f1

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('-epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('-log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-threshold', type=float, default=0.5,
                        help='threshold for the evaluation (default: 0.5)')
    parser.add_argument('-image_path', help='path for the training and validation folders')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-snapshot', default=None)
    parser.add_argument('-resume', type=int, default=None)
    parser.add_argument('-test_model', action='store_true')
    parser.add_argument('-save_path')
    args = parser.parse_args()

    assert args.image_path is not None

    device = "cuda"
    save_path = args.save_path
    if not args.test_model:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        log_path = os.path.join(save_path, 'logs')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        writer = SummaryWriter(log_dir=log_path)

    train_dataset = COCOMultiLabel(train=True,
                                   classification=True,
                                   image_path=args.image_path)
    test_dataset = COCOMultiLabel(train=False,
                                  classification=True,
                                  image_path=args.image_path)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=False,
                              collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=my_collate)

    model = Net().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.snapshot:
        if isinstance(model, nn.DataParallel):
            model.load_state_dict(torch.load(args.snapshot))
        else:
            model.load_state_dict(convert_weights(torch.load(args.snapshot)))
        if args.test_model == False:
            assert args.resume is not None
            resume = args.resume
            print "Resuming at", resume
        else:
            resume = 0
    else:
        resume = 1
    highest_f1 = 0
    epochs_without_imp = 0

    if args.test_model == False:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum)

    for epoch in range(resume, args.epochs + 1):
        if args.test_model == False:
            train(args, model, device, train_loader,
                  optimizer, epoch, writer)
            f1 = test(args, model, device, test_loader, args.threshold)
            writer.add_scalar('f1', f1 * 100, epoch)
            torch.save(model.state_dict(), args.save_path + "/checkpoint.pt")
            if f1 > highest_f1:
                torch.save(model.state_dict(), args.save_path + "/BEST_checkpoint.pt")
                print "Now the highest f1 is %.2f%%, it was %.2f%%" % (
                    100*f1, 100*highest_f1)
                highest_f1 = f1
            else:
                epochs_without_imp += 1
                print "Highest f1 is still %.2f%%, epochs without imp. %d" % (
                    100 * highest_f1, epochs_without_imp)
                if epochs_without_imp == 3:
                    adjust_learning_rate(optimizer, 0.1)
                    epochs_without_imp = 0
        else:
            f1 = test(args, model, device, test_loader, args.threshold)
            break

if __name__ == '__main__':
    main()
