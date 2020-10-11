import argparse
import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

from model import DEC, SAE


def main(hparams):
    pl.seed_everything(hparams.seed)

    if hparams.mode == "sae":
        pass
    # Pretrain Stacked AutoEncoder
    pprint.pprint("#########################################")
    pprint.pprint("# Start Pretraining Stacked AutoEncoder #")
    pprint.pprint("#########################################")
    sae = SAE([28 * 28, 500, 500, 2000, 10], activation=nn.ReLU(), dropout=0.2)
    trainer_sae_pt = pl.Trainer(
        gpus=hparams.gpus,
        max_epochs=hparams.epoch_pretrain,
        logger=TensorBoardLogger(save_dir="./logs", name="sae_pt"),
        early_stop_callback=True,
        deterministic=True,
        distributed_backend=hparams.distributed_backend,
    )

    trainer_sae_pt.fit(sae)

    # Finetune Stacked AutoEncoder without Dropout
    pprint.pprint("########################################")
    pprint.pprint("# Start Finetuning Stacked AutoEncoder #")
    pprint.pprint("########################################")
    sae.hparams.dropout = 0.0
    trainer_sae_ft = pl.Trainer(
        gpus=hparams.gpus,
        max_epochs=hparams.epoch_finetune,
        logger=TensorBoardLogger(save_dir="./logs", name="sae_ft"),
        early_stop_callback=True,
        deterministic=True,
        distributed_backend=hparams.distributed_backend,
    )
    trainer_sae_ft.fit(sae)

    # sae_pth = "./sae_ft/version_0/checkpoints/epoch=421.ckpt"
    # sae = SAE.load_from_checkpoint(sae_pth)

    # Train Deep Embedded Clustering(DEC)
    pprint.pprint("###########################################")
    pprint.pprint("# Start Training Deep Embedded Clustering #")
    pprint.pprint("###########################################")

    dec = DEC(sae.encoder, num_cluster=10, hidden_dim=10)
    trainer_dec = pl.Trainer(
        gpus=hparams.gpus,
        max_epochs=hparams.epoch_dec,
        logger=TensorBoardLogger(save_dir="./logs", name="dec"),
        deterministic=True,
        distributed_backend=hparams.distributed_backend,
    )
    trainer_dec.fit(dec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="The number of gpus")
    parser.add_argument(
        "--distributed_backend", type=str, default="dp", help="GPU distributing options"
    )
    parser.add_argument(
        "--seed", type=int, default=711, help="Random seed for reproducibility"
    )

    # settings for SAE
    parser.add_argument(
        "--epoch_pretrain", type=int, default=300, help="Max epoch to pretrain"
    )
    parser.add_argument(
        "--epoch_finetune", type=int, default=500, help="Max epoch to finetune"
    )

    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.1, help="Learning rate decay ratio"
    )
    parser.add_argument(
        "--lr_decay_step",
        type=float,
        default=20000,
        help="Learning rate decay frequency",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay ratio"
    )

    # settings for DEC
    parser.add_argument(
        "--lr_dec", type=float, default=0.01, help="Learning rate in DEC"
    )
    parser.add_argument(
        "--epoch_dec", type=int, default=1000, help="Training epoch for DEC"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.000001,
        help="Stopping threshold(Not used in this implementation)",
    )

    hparams = parser.parse_args()
    main(hparams)
