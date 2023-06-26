print("Starting...")
import warnings
warnings.filterwarnings("ignore")

import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import optuna


print("Necessary imports done.")

qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

print("Folder for dataset created.")
    
def ImportData(dbPath, bSize, nTrain, nVal):
    params = [trn.ASENeighborList(cutoff=5.),
              trn.RemoveOffsets(QM9.U0, remove_mean=True, 
              remove_atomrefs=True),
              trn.CastTo32()]
    
    qm9data = QM9(dbPath, 
        batch_size = bSize,
        num_train = nTrain, 
        num_val = nVal,
        transforms = params,
        property_units={QM9.U0: 'eV'},
        num_workers=1,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=True, # set to false when not using a GPU
        load_properties=[QM9.U0], # only load U0 property
    )
    qm9data.prepare_data()
    qm9data.setup()
    return qm9data

def settingModel(cutoff, n_atom_basis):
    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, 
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_U0],
        postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)]
    )
    return nnpot
    
def settingOutput():
    output_U0 = spk.task.ModelOutput(
        name=QM9.U0,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    return output_U0
    
def definingTask(nnpot, output_U0):
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )
    return task
    
def trainingModel(task, datamodule, trainer, maxEpochs):
    logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
    callbacks = [
        spk.train.ModelCheckpoint(
            monitor="MAE",
            filename="{epoch}-{MAE:.2f}",
            model_path = os.path.join(qm9tut, "best_inference_model"),
            save_top_k=1,
            mode="min",
            dirpath=qm9tut,  # Specify the directory path to save the best model
        )
    ]
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=qm9tut,
        max_epochs=maxEpochs,
    )
    trainer.fit(task, datamodule=datamodule)


def objective(trial):
    cutoff = trial.suggest_float("cutoff", 3.0, 6.0)
    n_atom_basis = trial.suggest_int("n_atom_basis", 20, 50)
    
    print("Importing data from the QM9 dataset")
    # Import data from the QM9 dataset
    dbPath = './qm9tut/qm9.db'
    qm9data = ImportData(dbPath, 100, 80000, 20000) # dbPath, bSize, nTrain, nVal

    print("Setting up model")
    # Setup model
    nnpot = settingModel(cutoff, n_atom_basis) # cutoff, n_atom_basis

    # Setup output
    output_U0 = settingOutput()

    # Define task
    task = definingTask(nnpot, output_U0)

    print("Training model...")
    trainer = Trainer(logger=False)
    trainingModel(task, qm9data, trainer, 5)  # Pass the `trainer` object as an argument

    # Return the validation loss as the objective value for optimization
    return trainer.callback_metrics["val_loss"].item()


# Create an Optuna study
study = optuna.create_study(direction="minimize")

# Start the hyperparameter search
study.optimize(objective, n_trials=10)

# Print the best hyperparameters and objective value
print("Best hyperparameters:", study.best_params)
print("Best objective value:", study.best_value)

