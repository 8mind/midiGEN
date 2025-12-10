"""
A single class to handle MIDI data processing.
"""

import mido
import os
import math
import time
from tqdm.auto import tqdm
import random
from typing import Optional, Union
import pygame
import numpy as np
import pickle

class MidiDataProcessing:
    """
    A class to handle MIDI data processing: tokenisation of MIDI files into token streams and reconstruction of MIDI files from token streams.
    1. Tokenisation involves reading MIDI files, quantising messages and their timings, and converting them into a sequence of integer tokens based on a predefined vocabulary.
    2. Reconstruction involves converting a sequence of integer tokens back into MIDI messages and saving them as MIDI files.
    The class also supports saving token streams to binary files for training/validation splits, and playing MIDI files as audio.
    3. The class allows for customization of quantisation steps for velocity, value, and pitch, as well as the time step for message timing.
    4. It also allows for filtering of MIDI channels based on selected instrument programs.
    """
    
    def __init__(self, *, time_step: float, longest_silence: float, program_select: Union[list[int], str]):
        """
        `program_select` can be either a list of uint7 integers like [0, 3, 64, 127] or a string in the format "Synth Lead/Pipe/Bass" (case in-sensitive).
        `longest_silence` tell us to cap the duration of silences to `longest_silence` (in seconds). This is to avoid unreasonably long lulls in the performance of an instrument (for example, the guitar might play for only 10 seconds and never again in a song).
        """
        self.time_step = time_step
        self.longest_silence = longest_silence
        self.program_select = program_select
        
        # generate the vocabulary and store it as self.vocab
        self.vocab = self.build_vocab()
        # a dict to lookup the index for a word (a.k.a. a 'token') in the vocabulary
        self.TtoI = {w: i for i, w in enumerate(self.vocab)} 

        # this is the list of instrument families as per General MIDI standard
        self.instrument_families = [
            'Piano',            # programs 0-7
            'Chromatic Percussion', # programs 8-15
            'Organ',            # programs 16-23
            'Guitar',           # programs 24-31
            'Bass',             # programs 32-39
            'Strings',          # programs 40-47
            'Ensemble',         # programs 48-55
            'Brass',            # programs 56-63
            'Reed',             # programs 64-71
            'Pipe',             # programs 72-79
            'Synth Lead',       # programs 80-87
            'Synth Pad',        # programs 88-95
            'Synth Effects',    # programs 96-103
            'Ethnic',           # programs 104-111
            'Percussive',       # programs 112-119
            'Sound Effects'     # programs 120-127
        ]
        # process the program_select argument further, storing it as self.program_list
        if isinstance(self.program_select, str):
            self.program_list = []
            for p in self.program_select.split("/"):
                idx = self.instrument_families.index(p.title())
                self.program_list.extend(range(idx * 8, (idx + 1) * 8))
        else:
            self.program_list = self.program_select



    def build_vocab(self) -> list[str]:
        """
        Returns a vocabulary (of size 2**16) of MIDI messages.
        If a MIDI message has the `pitch` attribute it is rounded to the nearest multiple of 64 (quantisation of the pitch attribute). The reason we perform quantisation is to keep the vocabulary size from exceeding 2**16. (Rounding of the pitch attribute shouldn't affect song fidelity much at all.)
        See https://mido.readthedocs.io/en/stable/message_types.html and https://gemini.google.com/share/8b0ceacfae1a for information on the kinds of MIDI messages.
        """

        q_pitches = sorted(list(set([self._round_to(p, 64, max_=8191, min_=-8192) for p in
                                   range(-8192, 8192)])))  # these are the allowed pitches after quantisation
        
        # the previous line is equivalent to the following one
        q_pitches = [-8192 + 64 * i for i in range(256)] + [8191] # these are the allowed pitches after quantisation

        # begin with the special tokens
        vocab = ["<delimiter>", "<empty>", "<freeze>"]

        vocab.extend([f"note_on note={note} velocity={velocity}" for note in range(128) for velocity in range(128)])
        vocab.extend([f"note_off note={note} velocity={velocity}" for note in range(128) for velocity in range(128)])

        vocab.extend([f"polytouch note={note} value={value}" for note in range(128) for value in range(128)])

        # some controls are special: (see https://gemini.google.com/share/8b0ceacfae1a)
        special_controls = [120, 121, 123, 124, 125, 127]
        vocab.extend([f"control_change control={control} value={value}" for control in range(128) if control not in special_controls for value in range(128)]) # the value for all but the special tokens is uint8
        vocab.extend([f"control_change control={control} value=0" for control in special_controls]) # special_controls can only take the value 0

        vocab.extend([f"program_change program={program}" for program in range(128)])

        vocab.extend([f"aftertouch value={value}" for value in range(128)])

        vocab.extend([f"pitchwheel pitch={pitch}" for pitch in q_pitches])

        vocab.extend([f"PAD_{i}" for i in range(2 ** 16 - len(vocab))])  # add padding tokens to make vocab size equal to 2**16 (the number of padding tokens is 246). the reason for rounding up the vocabulary size to a "nice number" in this way is to potentially speed up matrix multiplication (see Andrej Karpathy nanoGPT video)

        return vocab



    def quantise_message(self, msg):
        """
        Takes a MIDI message and returns a "quantised" version. The quantised version has pitch attribute rounded to the nearest multiple of 64.
        The reason we perform quantisation is to keep the vocabulary size from exceeding 2**16 (rounding of pitch attribute shouldn't affect song fidelity much at all).
        """
        msg_copy = msg.copy()  # copy the message to avoid changing the original

        # quantise velocity, value and pitch attributes of msg
        if hasattr(msg_copy, 'pitch'):
            msg_copy.pitch = self._round_to(msg_copy.pitch, 64, 8191, -8192)

        # handle special cases
        if hasattr(msg_copy, 'control') and msg_copy.control in (120, 121, 123, 124, 125, 127):
            # for these controls, the only value should be 0
            msg_copy.value = 0

        return msg_copy



    def tokenise(self, filepath: str) -> list[int]:
        """
        Takes a path holding a MIDI file; reads the file for messages; quantises the message (in the sense of the quantise_message() method above) and the time at which it occurs (in a sense to be specified below); looks for the index of the word in the vocabulary corresponding to the message; and writes this index into a stream. It is this list of integers (which I refer to as the "token stream") which is returned by the function.
        The time of the message is quantised as follows: the accumulated time of the message is rounded to the nearest multiple of `time_step`.
        I may refer to words in the vocabulary, or their positions in the vocabulary, as "tokens".
        """
        
        max_empty_steps = round(self.longest_silence / self.time_step) # maximum number of successive <empty> tokens to tolerate 

        # read the file with error handling
        try:
            midi = mido.MidiFile(filepath)  
        except FileNotFoundError:
            print(f"\nError: MIDI file {filepath} not found.")
            return []
        except Exception as e:
            print(f"\nError reading {filepath}: {e}.")
            return []
        
        # If the MIDI file is of "type 2" (asynchronous), skip it. Most MIDI files are of type 0 (single track) or type 1 (multiple tracks, synchronous).
        if midi.type == 2:
            print(f"\nWarning: {filepath} is of type 2 (asynchronous); skipping.")
            return []

        tokens = [[] for _ in range(16)]  # a token list for each of the 16 MIDI channels, so far empty
        time_ = 0  # global accumulated time (in secs)
        channel_times = [0 for _ in range (16)] # accumulated time in each channel of the last message which is not Meta (and has 'channel' attribute)
        channel_program = [None for _ in range(16)]  # store the program (instrument) playing in each channel. I will use this to filter out those channels that don't play an instrument in program_list
        for msg in midi:
            time_ += msg.time  # calculate accumulated time of current message
            if not msg.is_meta and hasattr(msg, 'channel'):
                c = msg.channel
                # It is usually the case that only one program plays per channel. Usually, I find that the program is set early on, and sometimes several times over a short span (< 1s) -- it is the final change that I consider "final".
                if msg.type == "program_change":
                    channel_program[c] = msg.program

                # determine whether to put in any <empty> or <freeze> tokens, based on msg.time
                diff = self._round_to(time_, self.time_step) - self._round_to(channel_times[c], self.time_step)  # `diff` is the difference between the quantised accumulated time of the current message and the quantised accumulated time of the last message in the channel
                ticks = round(diff / self.time_step)  # we expect `diff` to be a multiple of time_step -- we store this multiple as `ticks` (since the last event).
                if ticks == 0: 
                    if tokens[c]:
                        tokens[c].append(self.freeze_tok_id) # if no time has elapsed since the last message we need to insert the <freeze> token
                    # if tokens[c] is empty, don't do anything
                elif ticks > 1:
                    if tokens[c]:
                        tokens[c].extend([self.empty_tok_id] * min(ticks - 1, max_empty_steps)) # if more than 1 tick has elapsed since the last message, we need to insert `ticks-1` number of <empty> tokens, not exceeding `max_empty_steps`.
                    else: # if tokens[c] is empty then we need to insert one more empty token
                        tokens[c].extend([self.empty_tok_id] * min(ticks, max_empty_steps))

                channel_times[c] = time_ # update the channel time for channel c

                # then we go on to actually reading the message and tokenising it (assigning a word to it)
                msg_q = self.quantise_message(msg)  # quantise the message
                msg_q_stripped = " ".join(p for p in str(msg_q).split() if not (p.startswith("channel=") or p.startswith("time="))) # strip off the 'channel' and 'time' attributes from the string representation of the message
                tokens[c].append(self.TtoI[msg_q_stripped])  # look up the index of this message in the vocabulary and append it to the token list for channel c

        # now combine the token lists from each channel into a single token stream, filtering out those channels that don't play an instrument in program_list
        token_stream = []
        for c in range(16):
            if channel_program[c] in self.program_list:
                token_stream.extend(tokens[c])
                token_stream.append(self.del_tok_id) # append a delimiter token to separate the tracks

        return token_stream
    


    def tokens_to_midi(self, token_stream: list[int], out_filepath: str) -> None:
        """
        Takes a token stream; converts each token to a MIDI message; appends this message to a MIDI track; saves this single track to a MIDI file at `out_filepath`.
        """
        # Create a new MIDI file and track. The default ticks per beat is 480 and the default tempo is 500000 microseconds per beat (120 BPM) -- for a default of 960 ticks per second. 
        # When messages are added to the track, their time attribute is specified in "delta" (i.e. difference) ticks. This is different to how the time attribute appears when `for msg in mido.MidiFile(filename):` is called -- it appears as delta seconds. 
        # Note also that when `for track in mido.MidiFile(filename): for msg in track:` (i.e. when messages are read from the track) is called (which I don't make use of), the time attribute appears in delta ticks.
        
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)  # ensure the output directory exists

        midi_out = mido.MidiFile()
        track = mido.MidiTrack()
        midi_out.tracks.append(track)

        time_ = 0 # global time
        prev_time = 0 # time of previous MIDI message
        freeze = False # bool indicating whether time is frozen
        first_msg = False # bool indicating whether the first message has been encountered yet

        # if a token corresponds to a message, the time is updated to reflect the message time and the message is written into track (if not empty).
        for idx in token_stream:
            # ignore padding tokens
            if self.vocab[idx].startswith("PAD_"):
                continue
            # treat <delimiter> like a padding token
            elif self.vocab[idx] == "<delimiter>":
                continue
            # if <empty> token then increment time and go to next token
            elif self.vocab[idx] == "<empty>":
                if first_msg and not freeze:
                    time_ += self.time_step
                freeze = False
            elif self.vocab[idx] == "<freeze>":
                freeze = True
            else:
                if first_msg and not freeze:
                    time_ += self.time_step

                delta_time_in_ticks = round((time_ - prev_time) * 960) # assuming default of 960 ticks per second
                prev_time = time_

                track.append(mido.Message.from_str(self.vocab[idx] + f" time={delta_time_in_ticks}"))
                freeze = False
                first_msg = True

        # finally save the MIDI file
        try:
            midi_out.save(out_filepath)
        except Exception as e:
            print(f"\nError: couldn't save MIDI file to {out_filepath}: {e}.")

        return out_filepath



    def save_tokens(self, midi_filepaths: list[str], out_dir: str, val_frac: float, shuffle: Optional[bool]=True, seed: Optional[int]=None) -> dict[str, int]:
        """
        Shuffles midi_filepaths if `shuffle=True` (using random seed `seed` if specified). 
        The first `val_frac` fraction of files in midi_filepaths are used for validation and the rest for training.
        Goes through the MIDI files for the "train" and "val" splits; converts each file into a token stream; writes the token stream into a binary file at "out_dir/train.bin" or "out_dir/val.bin" respectively.
        Writes some meta data (about the tokenisation parameters used and other things) to a file at "out_dir/meta.pkl".
        Returns the number of tokens in each split.
        """

        if shuffle:
            if seed is None:
                midi_filepaths = random.sample(midi_filepaths, k=len(midi_filepaths))
            else:
                midi_filepaths = random.Random(seed).sample(midi_filepaths, k=len(midi_filepaths))

        num_val = round(len(midi_filepaths) * val_frac)
        splits = {
            "val": midi_filepaths[:num_val],
            "train": midi_filepaths[num_val:]
        }

        out = {
            "val_tokens": 0,
            "train_tokens": 0
        }

        os.makedirs(out_dir, exist_ok=True)

        for split, files in splits.items():
            out_filepath = os.path.join(out_dir, f"{split}.bin")
            with open(out_filepath, "wb") as f_out:
                for filepath in tqdm(files, desc=f"Processing {split} files"):
                    token_stream = self.tokenise(filepath)
                    if token_stream:
                        out[f"{split}_tokens"] += len(token_stream)
                        arr = np.array(token_stream, dtype=np.uint16)
                        arr.tofile(f_out)

        # write meta data
        metadata = {
            "time_step": self.time_step,
            "longest_silence": self.longest_silence,
            "program_select": self.program_select,
            "vocab_size": len(self.vocab),
            "val_tokens": out['val_tokens'],
            "train_tokens": out['train_tokens'],
            "total_tokens": out['val_tokens'] + out['train_tokens'],
            "val_frac": val_frac,
            "shuffle": shuffle,
            "seed": seed,
            "total_MIDI_files_processed": len(midi_filepaths),
        }
        meta_filepath = os.path.join(out_dir, "meta.pkl")
        with open(meta_filepath, 'wb') as f:
            pickle.dump(metadata, f)

        return out



    @staticmethod
    def play_midi(filepath: str) -> None:
        """
        Plays the MIDI file at `filepath` as audio.
        """
        if not os.path.exists(filepath):
            print(f"\nError: MIDI file {filepath} not found.")
            return

        # initialize pygame
        pygame.init()
        # initialize the mixer (necessary for music playback)
        pygame.mixer.init()

        print(f"\nAttempting to play {filepath}...")

        try:
            # load the MIDI file into pygame's music player
            pygame.mixer.music.load(filepath)

            # start playback (-1 means loop infinitely, 0.0 means start from the beginning)
            pygame.mixer.music.play(0)

            print("\nPlaying... Press Ctrl+C to stop.")
            # wait for the music to end to carry on with the script
            while pygame.mixer.music.get_busy():
                time.sleep(0.1) # but don't check the while loop incessantly (avoid busy-waiting)


        except pygame.error as e:
            print(f"\nPygame Error during playback: {e}.")
            print("\nNote: If you receive 'Unrecognized head-chunk' or similar errors, \nyour environment might need additional audio setup (like soundfonts).")

        finally:
            # stop and clean up pygame
            pygame.mixer.music.stop()
            pygame.quit()
            print("\nPlayback finished or stopped. Pygame closed.")


    @staticmethod
    def _round_to(x: float, y: float, max_: Optional[float]=None, min_: Optional[float]=None) -> float:
        """
        Rounds `x` to the nearest multiple of `y`, with the option of a maximum and minimum.
        """
        z = MidiDataProcessing._round_away(x / y) * y # can use either built-in round() or custom _round_away() method here; speed would probably favour built-in method
        if not max_ is None:
          z = min(z, max_)
        if not min_ is None:
          z = max(z, min_)
        return z
    
    @staticmethod
    def _round_away(x: float) -> int:
        """
        Rounds `x` to the nearest integer, rounding half-integers away from zero.
        """
        sign = 1 if x >= 0 else -1
        return sign * math.floor(abs(x) + 0.5)
