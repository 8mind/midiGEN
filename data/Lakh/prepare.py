"""
Download the Lakh MIDI dataset (on terminal):
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xf lmd_full.tar.gz
"""

import os
import glob
from midi_data_processing import MidiDataProcessing

for root, dirs, files in os.walk('lmd_full'):
    print(f"root: '{root}' | number of dirs: {len(dirs)} | number of files: {len(files)}")

midi_files = glob.glob('lmd_full/**/*.mid', recursive=True)
print(f"\nFound {len(midi_files)} MIDI files.")

MDP = MidiDataProcessing(time_step=1/64, 
                         longest_silence=2, 
                         program_select='Guitar')

print("\nStarting tokenisation...\n")

out = MDP.save_tokens(
          midi_filepaths=midi_files,
          out_dir=os.path.dirname(__file__),
          val_frac=0.005,
          shuffle=True,
          seed=42
      )

print("\nTokenisation complete!")
print(out)
