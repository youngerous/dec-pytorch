import os
from collections import OrderedDict
from typing import Iterable, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from utils import SoftClusterAssignment


def cluster_acc(y_true, y_pred):
    """
    This is code from original repository.
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)

    accuracy = 0
    for idx in range(len(ind[0]) - 1):
        i = ind[0][idx]
        j = ind[1][idx]
        accuracy += w[i, j]
    accuracy = accuracy * 1.0 / y_pred.size
    return accuracy


class SAE(pl.LightningModule):
    """
    Stacked AutoEncoder for pretraining the initial parameter and cluster centroid.

    :param dimensions: [input_dim, hidden_dim_1, ..., hidden_dim_N]
    :param activation: Non-linear activation function both in encoder and decoder
    :param final_activation: Non-linear activation function in final layer
    :param dropout: Dropout rate in each layer

    :param batch_size: Size of minibatch
    :param lr: Learning rate
    :param lr_decay: Learning rate decay ratio
    :param lr_decay_step: Learning rate decay frequency
    :param weight_decay: Weight decay ratio
    """

    def __init__(
        self,
        dimensions: Iterable[int],
        activation: Optional[nn.Module] = nn.ReLU(),
        final_activation: Optional[nn.Module] = None,
        dropout: Optional[float] = 0.0,
        batch_size: int = 256,
        lr: float = 0.1,
        lr_decay: float = 0.1,
        lr_decay_step: int = 20000,
        weight_decay: float = 0.0,
    ):
        super(SAE, self).__init__()
        self.criterion = nn.MSELoss()

        # stack encoder layers
        encoder_layers = self._add_linear_layer_stack(
            dimensions[:-1], activation, dropout
        )
        encoder_layers.extend(
            self._add_linear_layer_stack(
                [dimensions[-2], dimensions[-1]], final_activation, dropout=None,
            )
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # stack decoder layers
        decoder_layers = self._add_linear_layer_stack(
            list(reversed(dimensions[1:])), activation, dropout
        )
        decoder_layers.extend(
            self._add_linear_layer_stack(
                [dimensions[1], dimensions[0]], final_activation, dropout=None,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # initialize parameter
        self.encoder.apply(self._init_weight)
        self.decoder.apply(self._init_weight)

        self.save_hyperparameters()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)

    def prepare_data(self) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)

        # Hyperparameter Tuning by cross-validation on a validation set is not an option in unsupervised clustering.
        # So train and test set are combined.
        self.dset = ConcatDataset([train_dset, test_dset])

    def _add_linear_layer_stack(
        self,
        dims: Iterable[int],
        activation: Optional[nn.Module],
        dropout: Optional[float],
    ) -> List[nn.Module]:
        def single_unit(in_dim: int, out_dim: int) -> List[nn.Module]:
            unit = [nn.Linear(in_dim, out_dim)]
            if activation is not None:
                unit.append(activation)
            if dropout is not None:
                unit.append(nn.Dropout(0.2))
            return nn.Sequential(*unit)

        return [single_unit(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]

    def _init_weight(self, layer):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # follow paper setting
            nn.init.constant_(layer.bias, 0)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        scheduler = StepLR(
            optimizer, self.hparams.lr_decay_step, gamma=self.hparams.lr_decay
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> dict:
        data, _ = batch
        flatten = data.reshape(self.hparams.batch_size, -1)
        reconstruction = self(flatten)
        loss = self.criterion(reconstruction, flatten)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> dict:
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        log = {"avg_train_loss": train_loss_mean}
        return {"log": log, "train_loss": train_loss_mean}


class DEC(pl.LightningModule):
    """
    Deep Embedded Clustering

    :param encoder: Finetuned Encoder of Stacked AutoEncoder
    :param num_cluster: Number of cluster
    :param hidden_dim: Dimension of final encoder vector
    :param alpha: Freedom value of t-distribution
    :param batch_size: Batch size
    :param lr_dec: Learning rate
    :param tol: Threshold to stop training
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_cluster: int = 10,
        hidden_dim: int = 10,
        alpha: float = 1.0,
        batch_size: int = 256,
        lr_dec: float = 0.01,
        tol: float = 1e-3,
    ):
        super(DEC, self).__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.assignment = SoftClusterAssignment(num_cluster, hidden_dim, alpha)
        self.criterion = F.kl_div
        self.kmeans = KMeans(self.hparams.num_cluster, n_init=20)
        self.init = True

    def forward(self, batch):
        return self.assignment(self.encoder(batch))

    def prepare_data(self) -> None:
        transform = transforms.Compose([transforms.ToTensor(),])

        # Hyperparameter Tuning by cross-validation on a validation set is not an option in unsupervised clustering.
        # So train and test set are combined.
        train_dset = MNIST(os.getcwd(), train=True, transform=transform, download=True)
        test_dset = MNIST(os.getcwd(), train=False, transform=transform, download=True)
        self.dset = ConcatDataset([train_dset, test_dset])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=False, drop_last=True
        )

    def configure_optimizers(self) -> torch.optim:
        return optim.SGD(self.parameters(), lr=self.hparams.lr_dec, momentum=0.9)

    def training_step(self, batch, batch_idx) -> dict:
        if self.init:
            init_info = self._initialize_centroid()
            self.assignment = SoftClusterAssignment(
                self.hparams.num_cluster,
                self.hparams.hidden_dim,
                self.hparams.alpha,
                init_info["centroid"],
            )
            print(f"Initial acc: {init_info['accuracy']}")
            self.init = False

        data, target = batch
        q = self(data.reshape(self.hparams.batch_size, -1))
        p = self._get_target_distribution(q).detach()

        loss = self.criterion(q.log(), p)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> dict:
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        log = {"avg_train_loss": train_loss_mean}
        return {"avg_train_loss": train_loss_mean, "log": log}

    def validation_step(self, batch, batch_idx) -> dict:
        data, target = batch
        embedded = self(data.reshape(self.hparams.batch_size, -1))
        pred = torch.cat([embedded]).max(1)[1]

        accuracy = cluster_acc(target.cpu().numpy(), pred.cpu().numpy())
        log = {"accuracy": accuracy}
        return {"accuracy": accuracy, "log": log}

    def validation_epoch_end(self, outputs) -> dict:
        mean_acc = torch.stack([torch.tensor(x["accuracy"]) for x in outputs]).mean()
        log = {"accuracy": mean_acc}
        return {"log": log}

    def _initialize_centroid(self) -> dict:
        print("Set Initial Centroid...")
        dloader = DataLoader(
            self.dset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True
        )
        label, feature = [], []

        for batch in dloader:
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            label.append(target)
            feature.append(
                self.encoder(data.reshape(self.hparams.batch_size, -1)).detach().cpu()
            )
        label = torch.cat(label)
        pred = self.kmeans.fit_predict(torch.cat(feature).numpy())
        accuracy = cluster_acc(label.cpu().numpy(), pred)

        return {
            "accuracy": accuracy,
            "centroid": torch.tensor(
                self.kmeans.cluster_centers_, requires_grad=True
            ).cuda(),
        }

    def _get_target_distribution(self, q):
        numerator = (q ** 2) / torch.sum(q, 0)
        p = (numerator.t() / torch.sum(numerator, 1)).t()
        return p
