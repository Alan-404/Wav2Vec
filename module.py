import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import lightning as L
from torchmetrics.text import WordErrorRate

from model.wav2vec import Wav2Vec
from typing import List, Union, Tuple, Optional, Callable

import statistics

class Wav2VecModule(L.LightningModule):
    def __init__(self,
                 pad_token: int,
                 metric_fx: Callable[[torch.Tensor, bool], str],
                 token_size: int,
                 n_layers: int = 17,
                 d_model: int = 768,
                 heads: int = 12,
                 conv_dims: Union[List[str], Tuple[str]] = (512, 512, 512, 512, 512, 512, 512), 
                 kernel_sizes: Union[List[str], Tuple[str]] = (10, 3, 3, 3, 3, 2, 2), 
                 strides: Union[List[str], Tuple[str]] = (5, 2, 2, 2, 2, 2, 2),
                 dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.metric_fx = metric_fx
        self.model = Wav2Vec(token_size=token_size, n_layers=n_layers, d_model=d_model, heads=heads, conv_dims=conv_dims, kernel_sizes=kernel_sizes, strides=strides, dropout_rate=dropout_rate)

        self.train_loss = []
        self.val_loss = []
        self.val_score = []

        self.criterion = ConformerCriterion(blank_id=pad_token)
        self.metric = ConformerMetric()

        self.save_hyperparameters(ignore=['pad_token', 'metric_fx'])

    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]
        labels = batch[1]

        input_lengths = batch[2]
        target_lengths = batch[3]

        outputs, input_lengths = self.model(inputs, input_lengths)

        loss = self.criterion.ctc_loss(outputs, labels, input_lengths, target_lengths)

        self.train_loss.append(loss.item())

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]
        labels = batch[1]

        input_lengths = batch[2]
        target_lengths = batch[3]

        outputs, input_lengths = self.model(inputs, input_lengths)

        loss = self.criterion.ctc_loss(outputs, labels, input_lengths, target_lengths)
        score = self.metric.wer_score(self.metric_fx(outputs.cpu().numpy()), self.metric_fx(labels.cpu().numpy(), False))

        self.val_loss.append(loss.item())
        self.val_score.append(score.item())
        
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=3e-5, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]

    def on_train_epoch_end(self):
        loss = statistics.mean(self.train_loss)
        print(f"Train Loss: {(loss):.4f}")
        print(f"Current Learning Rate: {self.optimizers().param_groups[0]['lr']}")

        self.log("train_loss", loss, rank_zero_only=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], rank_zero_only=True)
        
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        print(f"Validation Loss: {(statistics.mean(self.val_loss)):.4f}")
        print(f"Validation Score: {(statistics.mean(self.val_score)):.4f}")

        self.val_loss.clear()
        self.val_score.clear()

class ConformerCriterion:
    def __init__(self, blank_id: int) -> None:
        self.ctc_criterion = nn.CTCLoss(
            blank=blank_id,
            zero_infinity=True,
            reduction='mean'
        )

    def ctc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return self.ctc_criterion(outputs.log_softmax(dim=-1).transpose(0,1), targets, input_lengths, target_lengths)

class ConformerMetric:
    def __init__(self) -> None:
        self.wer_metric = WordErrorRate()

    def wer_score(self, pred: Union[List[str], str], label: Union[List[str], str]) -> torch.Tensor:
        return self.wer_metric(pred, label)