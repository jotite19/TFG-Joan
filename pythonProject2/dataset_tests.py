import os
from ase.db import connect
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

from databaseTest import create_new_db

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
    original_db = connect('qm9.db')
    new_db = connect(path)
    new_db._metadata = original_db.metadata

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


def model_startup(trainingCutoff, dataCutoff, n_atom_basis, lr, m_epochs, log):
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


def main_sweep():
    # Define hyperparameters to search over
    run = wandb.init()

    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    lr = wandb.config.lr

    trainingCutoff = wandb.config.trainingCutoff
    dataCutoff = wandb.config.dataCutoff
    n_atom_basis = wandb.config.n_atom_basis
    m_epochs = epochs

    # DATA FOR TRAINING:
    data_split = [batch_size, 11000, 1000]  # B_size, train, val
    qm9data = get_data(data_split, dataCutoff, './qm9.db')
    qm9data.prepare_data()
    qm9data.setup()

    trainer, task = model_startup(trainingCutoff, dataCutoff, n_atom_basis, lr, m_epochs, True)

    train_model(trainer, task, qm9data)

    # CLEANING DATA:
    os.remove('/pythonProject2/split.npz')
    os.remove('/pythonProject2/splitting.lock')

    # DATA FOR VALIDATION:
    data_split = [10, 1000, 1000]
    qm9data = get_data(data_split, dataCutoff, './filtered.db')
    qm9data.setup()

    trainer, task = model_startup(trainingCutoff, dataCutoff, n_atom_basis, lr, m_epochs, False)

    print(validate_model(trainer, task, qm9data))

    # CLEANING DATA:
    os.remove('/pythonProject2/split.npz')
    os.remove('/pythonProject2/splitting.lock')


def sweep_func():
    sweep_configuration = {
        'method': 'random',
        'name': 'Plus16',
        'metric': {

            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [10]},
            'epochs': {'values': [5]},
            'lr': {'values': [0.001]},
            'trainingCutoff': {'values': [5]},
            'dataCutoff': {'values': [5]},
            'n_atom_basis': {'values': [30]}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project='sweep'
    )
    wandb.agent(sweep_id, function=main, count=10)


def main(training_path, validate_path):

    original_db = connect('qm9.db')
    new_db = connect(training_path)
    new_db.metadata = original_db.metadata

    original_db = connect('qm9.db')
    new_db = connect(validate_path)
    new_db.metadata = original_db.metadata

    # CLEANING DATA SPLIT:
    if os.path.exists('./split.npz'):
        os.remove('./split.npz')
        os.remove('./splitting.lock')

    m_epochs = 20
    lr = 0.001
    training_cutoff = 5
    data_cutoff = 5
    n_atom_basis = 38

    # DATA FOR TRAINING:
    t, v = data_split_fun(training_path, 0.8, 0.2)
    data_split = [512, t, v]  # B_size, train, val
    qm9data_train = get_data(data_split, data_cutoff, training_path)
    qm9data_train.prepare_data()
    qm9data_train.setup()

    trainer, task = model_startup(training_cutoff, data_cutoff, n_atom_basis, lr, m_epochs, True)

    train_model(trainer, task, qm9data_train)
    print(trainer.validate(task, datamodule=qm9data_train))

    # CLEANING DATA SPLIT:
    os.remove('./split.npz')
    os.remove('./splitting.lock')

    # DATA FOR VALIDATION:
    t, v = data_split_fun(validate_path, 0, 1)
    data_split = [512, t, v]
    qm9data_val = get_data(data_split, data_cutoff, validate_path)
    qm9data_val.setup()

    print(trainer.validate(task, datamodule=qm9data_val))

    # CLEANING DATA SPLIT:
    os.remove('./split.npz')
    os.remove('./splitting.lock')


if __name__ == '__main__':
    create_new_db('qm9.db', 'under_18_atom.db', 101974)

    training_path = 'under_18_atom.db'
    validate_path = 'qm9.db'

    main(training_path, validate_path)
    main(training_path, validate_path)




