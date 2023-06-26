import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from ase.db import connect

def clean_files():
    if os.path.exists('./split.npz'):
        os.remove('./split.npz')
        os.remove('./splitting.lock')

def data_split_fun(path, train, val):
    db = connect(path)
    db_len = len(db)
    if train == 0:
        train_data = 1
    else:
        train_data = int(db_len*train)
    if val == 0:
        val_data = 1
    else:
        val_data = int(db_len*val)
    return train_data, val_data


def get_data(data_split, dataCutoff, path):
    qm9data = QM9(
        path,
        batch_size=data_split[0],
        num_train=data_split[1],
        num_val=data_split[2],
        transforms=[
            trn.ASENeighborList(cutoff=dataCutoff),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
            trn.CastTo32()
        ],
        property_units={QM9.U0: 'eV'},
        num_workers=8,
        pin_memory=True,  # set to false when not using a GPU
        load_properties=[QM9.U0],  # only load U0 property
    )
    return qm9data


def model_startup(trainingCutoff, n_atom_basis, lr, m_epochs, log, n_inter):
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_atom_basis, cutoff=trainingCutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=n_inter,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(trainingCutoff)
    )
    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_U0],
        postprocessors=[trn.CastTo64(),
                        trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)]
    )

    output_U0 = spk.task.ModelOutput(
        name=QM9.U0,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError(),
            "MSE": torchmetrics.MeanSquaredError(),
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": lr}
    )

    logger = pl.loggers.TensorBoardLogger(save_dir='./qm9tut')
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join('./qm9tut', "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    if log:
        wandb_logger = WandbLogger(log_model="all")
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=wandb_logger,
            default_root_dir='./qm9tut',
            max_epochs=m_epochs,
            accelerator="gpu",
            devices=1,
        )
    else:
        trainer = pl.Trainer(
            callbacks=callbacks,
            default_root_dir='./qm9tut',
            max_epochs=m_epochs,
            accelerator="gpu",
            devices=1,
        )

    return trainer, task


def train_model(trainer, task, data):
    return trainer.fit(task, datamodule=data)


def validate_model(trainer, task, data):
    return trainer.validate(task, datamodule=data)


def main():
    run = wandb.init()

    # CLEANING DATA SPLIT:
    clean_files()

    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    lr = wandb.config.lr

    training_cutoff = wandb.config.trainingCutoff
    data_cutoff = wandb.config.dataCutoff
    n_atom_basis = wandb.config.n_atom_basis
    n_inter = wandb.config.n_inter

    # DATA FOR TRAINING:
    t, v = data_split_fun('qm9.db', 0.8, 0.2)
    data_split = [512, t, v]  # B_size, train, val
    qm9data_train = get_data(data_split, data_cutoff, 'qm9.db')
    qm9data_train.prepare_data()
    qm9data_train.setup()

    trainer, task = model_startup(training_cutoff, n_atom_basis, lr, 20, True, n_inter)

    train_model(trainer, task, qm9data_train)
    print(trainer.validate(task, datamodule=qm9data_train))

    t, v = data_split_fun('./Databases/Splited/over_18_atom.db', 0.2, 0.8)
    data_split = [512, t, v]
    qm9data_val = get_data(data_split, data_cutoff, './Databases/Splited/over_18_atom.db')
    qm9data_val.setup()

    output = trainer.validate(task, datamodule=qm9data_val)
    return output

def sweepFunc():
    sweep_configuration = {
        'method': 'random',
        'name': 'Loss',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [512]},
            'epochs': {'values': [25]},
            'lr': {'values': [0.001]},
            'trainingCutoff': {'distribution': 'q_uniform', 'min': 3, 'max': 9, 'q': 1},
            'dataCutoff': {'distribution': 'q_uniform', 'min': 3, 'max': 9, 'q': 1},
            'n_atom_basis': {'distribution': 'q_uniform', 'min': 24, 'max': 40, 'q': 2},
            'n_inter': {'distribution': 'q_uniform', 'min': 3, 'max': 15, 'q': 2}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project='sweep'
    )
    wandb.agent(sweep_id, function=main, count=40)

if __name__ == '__main__':
    sweepFunc()


