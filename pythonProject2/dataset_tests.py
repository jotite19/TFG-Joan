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

from database_representations import plot_dictionary_data


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


def main(training_path, validate_paths, folds):

    original_db = connect('qm9.db')
    new_db = connect(training_path)
    new_db.metadata = original_db.metadata

    # CLEANING DATA SPLIT:
    clean_files()

    m_epochs = 25
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

    clean_files()

    val_results = {}
    for path in validate_paths:
        val_results[path] = 0
    for i in range(folds):
        for path in validate_paths:

            # GIVE METADATA
            new_db = connect(path)
            new_db.metadata = original_db.metadata

            # DATA FOR VALIDATION:
            t, v = data_split_fun(path, 0.2, 0.8)
            data_split = [512, t, v]
            qm9data_val = get_data(data_split, data_cutoff, path)
            qm9data_val.setup()

            output = trainer.validate(task, datamodule=qm9data_val)

            print()
            print("VALIDATION MADE WITH: ", path)
            print(output)

            val_results[path] += output[0]['val_loss']

            run_path = "./Outputs/" + test_name + ".txt"
            # text = "Iteration" + str(i) + ": "
            text = "Iteration5: "

            with open(run_path, "a") as file:
                file.write(text + ": " + str(output[0]) + "\n")

            clean_files()

    for key in val_results:
        val_results[key] = val_results[key]/folds

    return val_results


if __name__ == '__main__':

    fold = 1
    folder_path = './Databases/nAtoms'
    val_paths = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        item_path = item_path.replace('\\', '/')
        val_paths.append(item_path)

    train_path = 'over_12_under_24.db'
    test_name = 'loss_factor_v2'
    val_loss_dic = main(train_path, val_paths, fold)
    # plot_dictionary_data(val_loss_dic, './Plots/' + test_name)














