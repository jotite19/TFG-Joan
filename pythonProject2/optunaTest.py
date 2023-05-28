import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning import Trainer
import optuna

def get_data(data_split, dataCutoff):
    qm9data = QM9(
        './qm9.db',
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

def train_model(trainingCutoff, dataCutoff, n_atom_basis, lr, m_epochs, data_split):
    qm9data = get_data(data_split, dataCutoff)
    qm9data.prepare_data()
    qm9data.setup()

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=trainingCutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
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
            "RMSE": torchmetrics.MeanSquaredError(),
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

    wandb_logger = WandbLogger(log_model="all")
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir='./qm9tut',
        max_epochs=m_epochs,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(task, datamodule=qm9data)

    return trainer

def evaluate_model(trainer, data_split, cutoff, task):
    qm9data = get_data(data_split, cutoff)
    qm9data.prepare_data()
    qm9data.setup()

    return trainer.validate(task, datamodule=qm9data)

def main():
    # Define hyperparameters to search over
    run = wandb.init()

    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    lr = wandb.config.lr

    trainingCutoff = wandb.config.trainingCutoff
    dataCutoff = wandb.config.dataCutoff
    n_atom_basis = wandb.config.n_atom_basis

    m_epochs = epochs
    data_split = [batch_size, 80000, 20000]  # B_size, train, val

    train_model(trainingCutoff, dataCutoff, n_atom_basis, lr, m_epochs, data_split)

if __name__ == '__main__':

    sweep_configuration = {
        'method': 'bayes',
        'name': 'Sampling',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [512]},
            'epochs': {'values': [22]},
            'lr': {'values': [0.001]},
            'trainingCutoff': {'distribution': 'q_uniform', 'min': 3, 'max': 9, 'q': 1},
            'dataCutoff': {'distribution': 'q_uniform', 'min': 3, 'max': 9, 'q': 1},
            'n_atom_basis': {'distribution': 'q_uniform', 'min': 24, 'max': 40, 'q': 2}
        }
    }

    sweep_configuration_lr = {
        'method': 'grid',
        'name': 'Sampling',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [512]},
            'epochs': {'values': [60, 70, 80]},
            'lr': {'values': [0.0001]},
            'trainingCutoff': {'values': [5]},
            'dataCutoff': {'values': [5]},
            'n_atom_basis': {'values': [30]}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration_lr,
        project='sweep'
    )
    wandb.agent(sweep_id, function=main, count=40)
