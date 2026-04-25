import os
import sys
import argparse
import time
from datetime import datetime

# NOTE:
# CUDA_VISIBLE_DEVICES must be set before importing torch.
# When set to "1", physical GPU #1 is remapped as logical cuda:0 inside this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

import datasets
import models
import conf
from training_utils import *

# Original code from https://github.com/weiaicunzai/pytorch-cifar100 <- refer to this repo for comments

def setup_wandb():
    if not args.wandb:
        return

    if wandb is None:
        raise ImportError("wandb is not installed. Please install it or run without -wandb.")

    run_name = (
        args.wandb_run_name
        if args.wandb_run_name
        else f"{args.net}-{args.dataset}-{conf.TIME_NOW}"
    )
    project_name = (
        args.wandb_project
        if args.wandb_project
        else f"pretrain-{args.dataset}"
    )

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "net": args.net,
            "dataset": args.dataset,
            "classes": args.classes,
            "batch_size": args.b,
            "warm": args.warm,
            "lr": args.lr,
            "epochs": EPOCHS,
            "milestones": MILESTONES,
            "gpu": args.gpu,
        },
    )
    wandb.watch(net, log="all", log_freq=max(len(trainloader), 1))


def log_metrics(metrics):
    if args.wandb:
        wandb.log(metrics)


def train(epoch):
    start = time.time()
    net.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    train_iterator = tqdm(
        trainloader,
        desc=f"Train Epoch {epoch}/{EPOCHS}",
        leave=False,
        disable=not args.tqdm,
    )

    for batch_index, (images, _, labels) in enumerate(train_iterator):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(images)
        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * batch_size
        running_correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        current_loss = running_loss / total_samples
        current_acc = running_correct / total_samples
        current_lr = optimizer.param_groups[0]['lr']

        if args.tqdm:
            train_iterator.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_acc:.4f}",
                lr=f"{current_lr:.6f}",
            )
        else:
            print(
                'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {loss:0.4f}\tAcc: {acc:0.4f}\tLR: {lr:0.6f}'.format(
                    epoch=epoch,
                    trained_samples=batch_index * args.b + batch_size,
                    total_samples=len(trainloader.dataset),
                    loss=current_loss,
                    acc=current_acc,
                    lr=current_lr,
                )
            )

        if args.wandb:
            log_metrics(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_acc": preds.eq(labels).float().mean().item(),
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                    "train/step": (epoch - 1) * len(trainloader) + batch_index,
                }
            )

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    print(
        "Train summary - Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            epoch,
            epoch_loss,
            epoch_acc,
            finish - start,
        )
    )

    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))

    log_metrics(
        {
            "train/epoch_loss": epoch_loss,
            "train/epoch_acc": epoch_acc,
            "train/epoch_time": finish - start,
            "train/epoch": epoch,
        }
    )

    return {"loss": epoch_loss, "acc": epoch_acc, "time": finish - start}


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0
    total_samples = 0

    eval_iterator = tqdm(
        testloader,
        desc=f"Eval Epoch {epoch}/{EPOCHS}",
        leave=False,
        disable=not args.tqdm,
    )

    for images, _, labels in eval_iterator:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        batch_size = len(images)
        test_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total_samples += batch_size

        if args.tqdm:
            eval_iterator.set_postfix(
                loss=f"{test_loss / total_samples:.4f}",
                acc=f"{correct / total_samples:.4f}",
            )

    finish = time.time()
    avg_test_loss = test_loss / total_samples
    accuracy = correct / total_samples

    if args.gpu:
        print("GPU INFO.....")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        current_device = torch.cuda.current_device()
        print(f"CURRENT CUDA DEVICE (logical): {current_device}")
        print(f"CURRENT CUDA DEVICE NAME: {torch.cuda.get_device_name(current_device)}")
        print(torch.cuda.memory_summary(), end="")
    print("Evaluating Network.....")
    print(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            epoch,
            avg_test_loss,
            accuracy,
            finish - start,
        )
    )
    print()

    log_metrics(
        {
            "eval/loss": avg_test_loss,
            "eval/acc": accuracy,
            "eval/time": finish - start,
            "eval/epoch": epoch,
        }
    )

    return accuracy


parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, required=True, help="net type")
parser.add_argument("-dataset", type=str, required=True, help="dataset to train on")
parser.add_argument("-classes", type=int, required=True, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("-tqdm", action="store_true", default=False, help="show tqdm progress bars")
parser.add_argument("-wandb", action="store_true", default=False, help="log metrics to wandb")
parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name")
args = parser.parse_args()


MILESTONES = (
    getattr(conf, f"{args.dataset}_MILESTONES")
    if args.net != "ViT"
    else getattr(conf, f"{args.dataset}_ViT_MILESTONES")
)
EPOCHS = (
    getattr(conf, f"{args.dataset}_EPOCHS")
    if args.net != "ViT"
    else getattr(conf, f"{args.dataset}_ViT_EPOCHS")
)
# get network
net = getattr(models, args.net)(num_classes=args.classes)
if args.gpu:
    net = net.cuda()

# dataloaders
root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
img_size = 224 if args.net == "ViT" else 32

trainset = getattr(datasets, args.dataset)(
    root=root, download=True, train=True, unlearning=False, img_size=img_size
)
testset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=False, img_size=img_size
)

trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
testloader = DataLoader(testset, batch_size=args.b, shuffle=False)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=MILESTONES, gamma=0.2
)  # learning rate decay
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

setup_wandb()

checkpoint_path = os.path.join(conf.CHECKPOINT_PATH, args.net, conf.TIME_NOW)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, "{net}-{dataset}-{epoch}-{type}.pth")

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    if epoch > args.warm:
        train_scheduler.step(epoch)

    train(epoch)
    acc = eval_training(epoch)

    # start to save best performance model after learning rate decay to 0.01
    if best_acc < acc:  # and epoch > MILESTONES[1]
        weights_path = checkpoint_path.format(
            net=args.net, dataset=args.dataset, epoch=epoch, type="best"
        )
        print("saving weights file to {}".format(weights_path))
        torch.save(net.state_dict(), weights_path)
        best_acc = acc
        continue

    # if not epoch % conf.SAVE_EPOCH:
    #     weights_path = checkpoint_path.format(net=args.net, dataset=args.dataset, epoch=epoch, type='regular')
    #     print('saving weights file to {}'.format(weights_path))
    #     torch.save(net.state_dict(), weights_path)

if args.wandb:
    wandb.finish()
