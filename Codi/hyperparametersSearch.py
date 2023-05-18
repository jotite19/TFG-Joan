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


qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)
%rm split.npz

def main(cutoff,n_atom_basis,lr):  
	qm9data = QM9(
	    './qm9.db', 
	    batch_size=10,
	    num_train=8000,
	    num_val=2000,
	    transforms=[
		trn.ASENeighborList(cutoff=5.),
		trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
		trn.CastTo32()
	    ],
	    property_units={QM9.U0: 'eV'},
	    num_workers=1,
	    #split_file=os.path.join(qm9tut, "split.npz"),
	    pin_memory=False, # set to false, when not using a GPU
	    load_properties=[QM9.U0], #only load U0 property
	)
	
	qm9data.prepare_data()
	qm9data.setup()    

	pairwise_distance = spk.atomistic.PairwiseDistances() 
	radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
	schnet = spk.representation.SchNet(
	    n_atom_basis=n_atom_basis, n_interactions=3,
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

	output_U0 = spk.task.ModelOutput(
	    name=QM9.U0,
	    loss_fn=torch.nn.MSELoss(),
	    loss_weight=1.,
	    metrics={
		"MAE": torchmetrics.MeanAbsoluteError()
	    }
	)

	task = spk.task.AtomisticTask(
	    model=nnpot,
	    outputs=[output_U0],
	    optimizer_cls=torch.optim.AdamW,
	    optimizer_args={"lr": float(lr)}
	)

	logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
	callbacks = [
	    spk.train.ModelCheckpoint(
		model_path=os.path.join(qm9tut, "best_inference_model"),
		save_top_k=1,
		monitor="val_loss"
	    )
	]
	
	wandb_logger = WandbLogger()
	trainer = Trainer(logger=wandb_logger)


	trainer = pl.Trainer(
	    callbacks=callbacks,
	    logger= wandb_logger,
	    default_root_dir=qm9tut,
	    max_epochs=20,
	)
	trainer.fit(task, datamodule=qm9data)
	return trainer.validate(task, datamodule=qm9data)
	

def objective(trial):
    # Define hyperparameters to search over
    
    lr = trial.suggest_float(name='lr', low=1e-5, high=1e-3)
    #lr = 1e-5
    
    cutoff = int(trial.suggest_float(name='cutoff', low=3, high=8))
    #cutoff = 5
    
    n_atom_basis = int(trial.suggest_loguniform('n_atom_basis', 20, 50))
    #n_atom_basis = 20	
    
    val_output = main(cutoff,n_atom_basis,lr)
    
    return val_output[0]["val_loss"]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

best_params = study.best_params
print('Best parameters:', best_params)
