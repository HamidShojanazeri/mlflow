from __future__ import print_function

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from fastprogress.fastprogress import master_bar, progress_bar
from pathlib import Path
from utils import *
import argparse
import mlflow
import mlflow.pytorch
import os
import random
import tempfile
import torch.nn.functional as F
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from functools import partial
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params

class Callback():
    _order = 0
    def set_runner(self, run): self.run = run
    def __getattr__(self, k): return getattr(self.run, k)
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

    def begin_fit(self):
        self.run.n_epochs = 0
        self.run.n_iter = 0
    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs+=1./self.iters
        self.run.n_iter+=1
    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True
    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

class autolog():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop, self.cbs = False, [MLFlowTracking_Model_Params(),MLFLowTrackingMetrics()] + cbs

        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False
    @property
    def opt(self): return self.learn.opt
    @property
    def model(self): return self.learn.model
    @property
    def loss_func(self): return self.learn.loss_func
    @property
    def data(self): return self.learn.data

    def one_batch(self, xb, yb):
        try:
            self.xb, self.yb = xb, yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in progress_bar(dl, leave=False): self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')
    def fit(self, epochs, learn):
        self.epochs, self.learn, self.loss = epochs, learn, torch.tensor(0.)
        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data["train_data"])
                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data["valid_data"])
                self('after_epoch')
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None
    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res
        return res

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func,self.data = model, opt, loss_func, data


def get_data(train_dl, valid_dl):
    data = {"train_data":train_dl,"valid_data": valid_dl}
    return data


def get_learner(model,optimizer, loss_fn,train_idx, valid_idx):
    data = get_data(train_idx, valid_idx)
    learn = Learner(model,optimizer, loss_fn,data=data)
    return learn

class AverageMetrics():
    def __init__(self, metrics, in_train): self.metrics, self.in_train = listify(metrics), in_train
    def reset(self):
        self.total_loss, self.count = 0., 0
        self.tot_mets = [0.]*len(self.metrics)
    @property
    def all_stats(self):
        return [self.total_loss.item()] + self.tot_mets
    @property
    def avg_stats(self):
        return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ''
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
        
    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.total_loss+=run.loss*bn
        self.count+=bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i]+=m(run.pred, run.yb)*bn

class MLFLowTrackingMetrics(Callback):
    def __init__(self):
        self.train_stats, self.valid_stats = AverageMetrics([accuracy], True), AverageMetrics([accuracy], False)
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)
        for item in self.train_stats.avg_stats:
            mlflow.log_metric("train_loss", self.train_stats.avg_stats[0],self.epoch)
            mlflow.log_metric("train_accuracy", self.train_stats.avg_stats[1].item(),self.epoch)
        for item in self.valid_stats.avg_stats:
            mlflow.log_metric("valid_loss", self.valid_stats.avg_stats[0],self.epoch)
            mlflow.log_metric("valid_accuracy", self.valid_stats.avg_stats[1].item(),self.epoch)


class MLFlowTracking_Model_Params(Callback):
    def begin_fit(self):

        mlflow.log_param('epochs', self.epochs)


    def after_fit(self):
        for param_group in self.opt.param_groups:
            self.lr=param_group['lr']
        mlflow.log_param('lr', self.lr)
        mlflow.pytorch.log_model(self.learn.model, "models")
