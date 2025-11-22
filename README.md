A decoder-only transformer (of the exact same architecture as GPT-2 124M) to learn the patterns in MIDI file data. 

MIDI files are a way to store instrumental music as a sequence of events such as "note_on note=60 time=0" and "note_off note=60 time=10" (without storing the actual audio). 

The sequential nature of the filetype probably makes it ideal for sequence modelling and next-token prediction. 

The goal is to train the decoder on the Lakh MIDI dataset (178,000 songs) and then generate music with the best model.

NOTE: the files `model.py`, `configurator.py` are taken VERBATIM from <https://github.com/karpathy/nanoGPT/tree/master>. `train.py` is also verbatim from `karpathy/nanoGPT` but for two lines: line 263 and 276.

OTHER NOTES: Miscellaneous but useful stuff is stored in `miscellaneous.py`.

Get started:

0) Dependencies: `pip install torch numpy wandb tqdm mido pygame` -- `mido` to read MIDI files and `pygame` to play a MIDI file out-loud.

1) Download the Lakh Midi dataset: 
```
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xf lmd_full.tar.gz
```

2) Run `python data/Lakh/prepare.py` (imports from `midi_data_processing.py`). This should generate `data/Lakh/meta.pkl`, `data/Lakh/train.bin` and `data/Lakh/val.bin`.

3) Run `torchrun --standalone --nproc_per_node=8 train.py config/train_midigen.py` (imports from `model.py` and executes `configurator.py`). 
This should create `out/ckpt.pt` (the parameters of the best model). 

To modify parameters through terminal do something like `torchrun --standalone --nproc_per_node=8 train.py --batch_size=32 config/train_midigen.py --EPOCHS=20` (i.e. put lowercase variables BEFORE `config/train_midigen.py` and uppercase variables AFTER `config/train_midigen.py`). 
This is due to the specific way `configurator.py` deals with the command line arguments (`configurator.py` is executed at line 77 in `train.py`).

4) Finally run `python sample.py --play_=True` to generate a sample at `out/generate/*.mid`, and play it out-loud after generation.

What does a typical 1 minute generated sample sound like? Let's hear:


Sounds pleasant enough...
