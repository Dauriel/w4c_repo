# Weather4cast 2023 Starter Kit
#
# This Starter Kit builds on and extends the Weather4cast 2022 Starter Kit,
# the original license for which is included below.
#
# In line with the provisions of this license, all changes and additional
# code are also released unde the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 

# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import datetime
import os
import torch
import torch.nn.functional as F

from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData

class DataModule(L.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params     
        self.training_params = training_params
        if mode in ['train']:
            print("Loading TRAINING/VALIDATION dataset -- as test")
            self.train_ds = RainData('training', **self.params)
            self.val_ds = RainData('validation', **self.params)
            print(f"Training dataset size: {len(self.train_ds)}")
        if mode in ['val']:
            print("Loading VALIDATION dataset -- as test")
            self.val_ds = RainData('validation', **self.params)  
        if mode in ['predict']:    
            print("Loading PREDICTION/TEST dataset -- as test")
            self.test_ds = RainData('test', **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset, 
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, 
                        pin_memory=pin, prefetch_factor=2,
                        persistent_workers=False)
        return dl
    
    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)
    
    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def load_model(Model, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['experiment'], **params['dataset'], **params['train']} 
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params['model'], p)            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, UNet_params=params['model'], params=p)
    return model
    
def upsample(y_hat, scale_factor=2):
    y_hat = y_hat.squeeze()
    bs, ts, width, height = y_hat.shape
    height = int(height * scale_factor)
    width = int(width * scale_factor)
    tar_size = (height, width)
    y_hat = F.interpolate(y_hat, size=tar_size, mode='bilinear')
    y_hat = y_hat.unsqueeze(dim=1)
    return y_hat
    
class EnsembleModel(L.LightningModule):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list):
        super().__init__()            
        self.models = torch.nn.ModuleList(models)

    def forward(self, batch):
        x, y, metadata = batch
        result = None
        for model in self.models:
            y_hat = model(x)
            y_hat = upsample(y_hat)
            if result is None:
                result = y_hat
            else:
                result += y_hat
        result /= torch.tensor(len(self.models))
        return result
        
    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata = batch
        result = None
        for model in self.models:
            y_hat = model(x)
            y_hat = upsample(y_hat)
            if result is None:
                result = y_hat
            else:
                result += y_hat
        result /= torch.tensor(len(self.models))
        return result

def get_trainer(gpus,params):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs=params['train']['max_epochs'];
    print("Trainig for",max_epochs,"epochs");
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3, save_last=True,
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')
    
    # date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    # version = params['experiment']['name']
    # version = version + '_' + date_time

    #SET LOGGER
    if params['experiment']['logging']: 
        # logger = TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],name=params['experiment']['sub_folder'], version=version, log_graph=True)
        logger = WandbLogger(project="w4c2023", log_model="all", save_dir=params['experiment']['experiment_folder'])
    else: 
        logger = False

    if params['train']['early_stopping']: 
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else: 
        callback_funcs = [checkpoint_callback]

    trainer = L.Trainer(max_epochs=max_epochs,
                         gradient_clip_val=params['model']['gradient_clip_val'],
                         gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                         callbacks=callback_funcs,
                         logger=logger,
                         profiler='simple',
                         precision=params['experiment']['precision'],
                        )

    return trainer

def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
    scores = torch.concat(scores)   
    tensor_to_submission_file(scores,predict_params)

def do_test(trainer, model, test_data):
    scores = trainer.test(model, dataloaders=test_data)
    
def train(params, gpus, mode, checkpoint_path, model="UNetModel"): 
    """ main training/evaluation method
    """
    # ------------
    # data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)

    # ------------
    # Add your models here
    # ------------
    if model == "SmatUnet":
        from models.smatunet_lightning_w4c23 import UNet_Lightning as UNetModel
    elif model == "SmatSimVP":
        from models.smatsimvp_lightning_w4c23 import UNet_Lightning as UNetModel
    elif model == "SimVP":
        from models.simvp_lightning_w4c23 import UNet_Lightning as UNetModel
    else:
        from models.unet_lightning_w4c23 import UNet_Lightning as UNetModel

    
    checkpoints_dict = {
        'boxi_0015': {
            2019: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0015y2019-best.ckpt',
            2020: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0015y2020-best.ckpt'
        },
        'boxi_0034': {
            2019: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0034y2019.ckpt',
            2020: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0034y2020-best.ckpt'
        },
        'boxi_0076': {
            2019: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0076y2019-best.ckpt',
            2020: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0076y2020-best.ckpt'
        },
        'roxi_0004': {
            2019: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0004y2019-best.ckpt',
            2020: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0004y2020-best.ckpt'
        },
        'roxi_0005': {
            2019: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0005y2019-best.ckpt',
            2020: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0005y2020-best.ckpt'
        },
        'roxi_0006': {
            2019: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0006y2019-best.ckpt',
            2020: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0006y2020-best.ckpt'
        },
        'roxi_0007': {
            2019: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2019/r0007y2019.ckpt',
            2020: '/kaggle/input/w4c-smatunet-checkpoints/checkpoints/2020/r0007y2020-best.ckpt'
        }
    }

    checkpoints_list=[checkpoints_dict["boxi_0015"][2019], checkpoints_dict["boxi_0034"][2019], checkpoints_dict["boxi_0076"][2019], checkpoints_dict["roxi_0004"][2019], checkpoints_dict["roxi_0005"][2019], checkpoints_dict["roxi_0006"][2019], checkpoints_dict["roxi_0007"][2019], 
                      checkpoints_dict["boxi_0015"][2020], checkpoints_dict["boxi_0034"][2020], checkpoints_dict["boxi_0076"][2020], checkpoints_dict["roxi_0004"][2020], checkpoints_dict["roxi_0005"][2020], checkpoints_dict["roxi_0006"][2020], checkpoints_dict["roxi_0007"][2020]]
    
    # Loading models using the updated checkpoint paths
    models = [load_model(UNetModel, params, path) for path in checkpoints_list]
    model = EnsembleModel(models)
  
    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params)
    get_cuda_memory_usage(gpus)

    if mode == 'predict':
    # ------------
    # PREDICT
    # ------------
        print("--------------------")
        print("--- PREDICT MODE ---")
        print("--------------------")
        print("REGIONS!:: ", params["dataset"]["regions"], params["predict"]["region_to_predict"])
        if params["predict"]["region_to_predict"] not in params["dataset"]["regions"]:
            print("EXITING... \"regions\" and \"regions to predict\" must indicate the same region name in your config file.")
        else:
            do_predict(trainer, model, params["predict"], data.test_dataloader())
    
    get_cuda_memory_usage(gpus)

def update_params_based_on_args(options):
    config_p = os.path.join('models/configurations',options.config_path)
    params = load_config(config_p)
    
    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name
    return params
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='./configurations/config_basline.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1, 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='', 
                         help="Set the name of the experiment")
    parser.add_argument("-mdl", "--model", type=str, required=False, default='UnetModel', 
                        help="choose model: Benchmark unet (default) / SmatUnet / other")

    return parser

def main():
    parser = set_parser()
    options = parser.parse_args()

    params = update_params_based_on_args(options)
    train(params, options.gpus, options.mode, options.checkpoint, options.model)

if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU
    python train.py --gpus 2 --mode train --config_path config_baseline.yaml --name baseline_train

    2) train from scratch on four GPUs
    python train.py --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --name baseline_train
    
    3) fine tune a model from a checkpoint on one GPU
    python train.py --gpus 1 --mode train  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
    
    4) evaluate a trained model from a checkpoint on two GPUs
    python train.py --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate

    5) generate predictions (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode predict  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    """
