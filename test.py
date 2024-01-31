import os
import torch
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import torchsummary

from dataset import Wav2VecTestDataset

import fire

from processing.processor import Wav2VecProcessor
from model.wav2vec import Wav2Vec

from typing import Tuple
from module import ConformerMetric
from common import map_weights
from typing import Optional, Union, List, Tuple
def test(result_folder: str,
         test_path: str,
         vocab_path: str,
         arpa_path: str,
         checkpoint: str,
         sampling_rate: int = 16000,
         pad_token: str = "<pad>",
         unk_token: str = "<unk>",
         word_delim_token: str = "|",
         n_layers: int = 12,
          d_model: int = 768,
          heads: int = 12,
          conv_dims: Union[List[str], Tuple[str]] = (512, 512, 512, 512, 512, 512, 512),
          kernel_sizes: Union[List[str], Tuple[str]] = (10, 3, 3, 3, 3, 2, 2),
          strides: Union[List[str], Tuple[str]] = (5, 2, 2, 2, 2, 2, 2),
          dropout_rate: float = 0.0,
         batch_size: int  = 1,
         num_examples: int = None,
         saved_name: str = None):
    if os.path.exists(result_folder) == False:
        os.mkdir(result_folder)

    # Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Processor Setup
    processor = Wav2VecProcessor(
        vocab_path=vocab_path,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delim_token=word_delim_token,
        sampling_rate=sampling_rate,
        lm_path=arpa_path
    )

    # Model Setup
    model = Wav2Vec(token_size=len(processor.dictionary), n_layers=n_layers, d_model=d_model, heads=heads, conv_dims=conv_dims, kernel_sizes=kernel_sizes, strides=strides, dropout_rate=dropout_rate).to(device)

    checkpoint = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(map_weights(checkpoint['state_dict']))
    else:
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    metric = ConformerMetric()

    def get_batch(signals: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        signals, signal_lengths = processor(signals, return_length=True)
        return signals, signal_lengths
    
    dataset = Wav2VecTestDataset(test_path, processor, num_examples=num_examples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=get_batch)

    labels = dataset.prompts['text'].to_list()
    preds = []

    def test_step(_: Engine, batch: Tuple[torch.Tensor]):
        inputs = batch[0].to(device)
        input_lengths = batch[1].to(device)
        
        with torch.no_grad():
            outputs, output_lengths = model(inputs, input_lengths)
        
        outputs = outputs.cpu().numpy()
        output_lengths = output_lengths.type(torch.int).cpu().numpy()

        for index, logit in enumerate(outputs):
            pred = processor.decode_beam_search(logit[:output_lengths[index], :])
            preds.append(pred)

    tester = Engine(test_step)
    ProgressBar().attach(tester)

    @tester.on(Events.STARTED)
    def _ (_: Engine):
        torchsummary.summary(model)
        model.eval()

    @tester.on(Events.COMPLETED)
    def _ (_: Engine):
        print(f"WER score: {metric.wer_score(preds, labels)}")

    tester.run(dataloader, max_epochs=1)

    if saved_name is not None:
        saved_filename = saved_name
    else:
        test_name = os.path.basename(test_path)
        saved_filename = f"result_{test_name}"

    df = dataset.prompts
    df['pred'] = preds

    df.to_csv(f"{result_folder}/{saved_filename}", sep="\t", index=False)

if __name__ == '__main__':
    fire.Fire(test)