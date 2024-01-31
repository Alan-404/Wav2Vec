import os
import torch
from torch.utils.data import DataLoader, random_split

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers.wandb import WandbLogger

import fire

from module import Wav2VecModule
from processing.processor import Wav2VecProcessor
from dataset import ConformerDataset

from dotenv import load_dotenv
from typing import Optional, Union, List, Tuple

load_dotenv()

def train(train_path: str,
          batch_size: int = 1,
          num_epochs: int = 1,
          saved_checkpoint: str = "./checkpoints",
          vocab_path: str = "./vocabulary/dictionary.json",
          num_train: Optional[int] = None,
          n_layers: int = 12,
          d_model: int = 768,
          heads: int = 8,
          conv_dims: Union[List[str], Tuple[str]] = (512, 512, 512, 512, 512, 512, 512, 512),
          kernel_sizes: Union[List[str], Tuple[str]] = (10, 3, 3, 3, 3, 2, 2, 2),
          strides: Union[List[str], Tuple[str]] = (5, 2, 2, 2, 2, 2, 2, 2),
          dropout_rate: float = 0.1,
          sampling_rate: int = 16000,
          pad_token: str = "<pad>", 
          unk_token: str = "<unk>", 
          word_delim_token: str = "|",
          checkpoint: str = None,
          use_validation: bool = False,
          val_path: Optional[str] = None,
          val_size: float = 0.1,
          early_stopping_patience: int = 3,
          num_val: Optional[int] = None,
          num_workers: int = 1,
          project_name: str = "wav2vec_speech_to_text"):
    
    assert os.path.exists(vocab_path)

    processor = Wav2VecProcessor(
        vocab_path=vocab_path,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delim_token=word_delim_token,
        sampling_rate=sampling_rate
    )

    if checkpoint is None:
        module = Wav2VecModule(
            pad_token=processor.pad_token,
            metric_fx=processor.decode_batch,
            token_size=len(processor.dictionary),
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            conv_dims=conv_dims,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dropout_rate=dropout_rate
        )
    else:
        module = Wav2VecModule.load_from_checkpoint(checkpoint)

    def get_batch(batch) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signals, transcripts = zip(*batch)
        signals, signal_lengths = processor(signals, return_length=True)

        tokens, token_lengths = processor.tokenize(transcripts)

        return signals, tokens, signal_lengths, token_lengths
    
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=saved_checkpoint, filename="{epoch}", save_on_train_epoch_end=True, save_last=True))

    if use_validation:
        callbacks.append(EarlyStopping(monitor='val_score', verbose=True, mode='min', patience=early_stopping_patience))

    dataset = ConformerDataset(train_path, processor=processor, num_examples=num_train)
    
    if use_validation:
        if val_batch_size is None:
            val_batch_size = batch_size
        if val_path is not None:
            val_dataset = ConformerDataset(val_path, processor=processor, num_examples=num_val)
        else:
            if type(val_size) == int:
                data_lengths = [dataset.__len__() - val_size, val_size]
            else:
                data_lengths = [1 - val_size, val_size]
            dataset, val_dataset = random_split(dataset, lengths=data_lengths, generator=torch.Generator().manual_seed(41))
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, collate_fn=lambda batch: get_batch(batch), num_workers=num_workers)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: get_batch(batch), num_workers=num_workers)

    strategy = 'auto'
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(process_group_backend='gloo')

    logger = WandbLogger(
        project=project_name,
        name=os.environ.get("WANDB_USERNAME"),
        save_dir=os.environ.get("WANDB_SAVE_DIR"),
    )

    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, precision='16-mixed', strategy=strategy, logger=logger)
    
    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=val_dataloader if use_validation else None, ckpt_path=checkpoint)

if __name__ == '__main__':
    fire.Fire(train)