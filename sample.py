"""
Sample from a trained model. The files are saved under os.path.join(out_dir, "generate").
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from midi_data_processing import MidiDataProcessing
import datetime

# -----------------------------------------------------------------------------
out_dir = 'out' # directory where best model is saved
num_samples = 1 # number of samples to generate
gen_len = 60 # desired length of generated sample in seconds
play_ = False # if True, the sample plays out-loud after each generation
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 2 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = None
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in the dataset folder
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

MDP = MidiDataProcessing(time_step=meta['time_step'], 
                         longest_silence=meta['longest_silence'], 
                         program_select=meta['program_select'])

# start_ids = [meta['del_tok_id']] # start of generation with the delimiter token
# start_ids = MDP.tokenise("tears_in_heaven.mid")
# start_ids = start_ids[:500]

# the vocab index of the notes C, A, B respectively
C = MDP.TtoI["note_on note=60 velocity=88"]
A = MDP.TtoI["note_on note=57 velocity=88"]
B = MDP.TtoI["note_on note=59 velocity=88"]
# indices for a silence of 0.5s
silence = [MDP.TtoI["<empty>"]] * round(0.5 / meta['time_step'])
# let start_ids be the simple tune C A C B with 0.5s in between
start_ids = [C] + silence + [A] + silence + [C] + silence + [B] 

x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

max_new_tokens = round(gen_len / meta['time_step'])
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
            pth = MDP.tokens_to_midi(y[0].tolist(), out_filepath=os.path.join(out_dir, "generate", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mid"))
            
            if play_:
                MDP.play_midi(pth)
