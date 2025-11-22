A decoder-only transformer (of the same architecture as GPT-2 124M) to learn the patterns in MIDI file data. 
MIDI files are a way to store instrumental music as a sequence of events such as "note_on note=60 time=0" and "note_off note=60 time=10" (without storing the actual audio). 
The sequential nature of the filetype probably makes it ideal for sequence modelling and next-token prediction. 
The goal is to train the decoder on the Lakh MIDI dataset (178,000 songs) and then generate music with the best model.

Get started:
1) First download the Lakh Midi dataset: 
```
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xf lmd_full.tar.gz
```
2) Then run `python prepare.py`. This should generate `data/Lakh/meta.pkl`, `data/Lakh/train.bin` and `data/Lakh/val.bin`.
3) Then run `python train.py config/train_midigen.py`. This should create `out/ckpt.pt` (the parameters of the best model).
4) Then finally run `python sample.py --play_=True` to generate a sample at `out/generate/*.mid` and play it out loud after generation.

What does this sound like? Let's hear:
