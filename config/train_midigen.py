"""
Override variables in train.py for 'from-scratch' training
"""

import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--EPOCHS', type=float, default=10) # how many times to loop over training data during training
parser.add_argument('--EVAL_EPOCHS', type=float, default=5) # how many times to loop over validation data during evaluation
parser.add_argument('--CONTEXT_WINDOW', type=int, default=4) # context window (in seconds)
parser.add_argument('--TOTAL_EVALS', type=int, default=100) # how many evaluations to perform during training
args, _ = parser.parse_known_args()

dataset = 'Lakh' # changed from 'openwebtext'
wandb_log = True # changed to true -- better to log progress
wandb_project = 'midiGEN' # changed from 'owt'
wandb_run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # changed from 'gpt2'
always_save_checkpoint = False # changed to false -- better since it saves only best performing model (see line 274 in train.py)
log_interval = 10 # how often to print logs (in steps)

with open(os.path.join('data', dataset, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

CONTEXT_WINDOW = args.CONTEXT_WINDOW
block_size = round(CONTEXT_WINDOW / meta['time_step'] / 64) * 64 # number of tokens that fit in CONTEXT_WINDOW rounded to the nearest 64. (block_size = 1024 in original.)

TOKENS_PER_ITER = gradient_accumulation_steps * batch_size * block_size # number of tokens processed in every step of gradient descent

# use the user inputs (or defaults) to set the variables
TOTAL_EVALS = args.TOTAL_EVALS 
EPOCHS = args.EPOCHS 
EVAL_EPOCHS = args.EVAL_EPOCHS  

max_iters = round(EPOCHS * meta['train_tokens'] / TOKENS_PER_ITER) # total number of training iterations. (600000 in original.)
eval_iters = round(EVAL_EPOCHS * meta['val_tokens'] / TOKENS_PER_ITER) # number of evaluation iterations. (200 in original.)

eval_interval = round(max_iters / TOTAL_EVALS) # how often to evaluate the model. (2000 in original.)

warmup_iters = round(0.003 * max_iters) # 0.3% of max_iters. (0.33% of max_iters in original.)
lr_decay_iters = max_iters # (in original, lr_decay_iters = max_iters also.)

learning_rate = 6e-4 # ! same as original -- but should we change this?
min_lr = learning_rate / 10 # (in original, min_lr = learning_rate / 10 also.)

# n_layer = 12 # ! or 8?
# n_head = 12
# n_embd = n_head * 64 # ! or 32?
