from string import ascii_letters, digits
import plotly.express as px
import torch
import torch.nn.functional as F
from common import load_tokenizer, load_lora_model, CHECKPOINT, LORA_OUTPUT_DIR, DEVICE



def get_dist_of_guess_chars(logits, tokenizer, guesses):
    tokens = []
    for guess in guesses:
        tokens.append(tokenizer(guess),return_tensors='pt'.input_ids.tolist()[0])
    
    breakpoint()
    return


# first idea
# look at output probas 
tokenizer = load_tokenizer(CHECKPOINT)
model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)

input_ids = tokenizer('more than one word', return_tensors='pt').input_ids.to(DEVICE)

# logits.shape = [batch_size, sequence_length, vocab_size]
# torch.Size([1, 4, 49152])
logits = model(input_ids).logits.cpu()

top_5 = [tokenizer.decode(token) for token in torch.topk(logits[0][3], k=5).indices.tolist()]
print(top_5)

# data = {'idx': [], 'logit': []}
# for i, logit in enumerate(logits[0][0]):
#     data['idx'].append(i)
#     data['logit'].append(logit)

# fig = px.bar(data, x='idx', y='logit', orientation='v')
# fig.show()


# so I think I want to do a big search and see whether any specific letter
# causes a totally different distribution
# can only plot about 1_000 bars out of 50_000 probas
# so this can't be a visual check, has to be a machine
# I guess I need pairwise KL divergence?
# or maybe I can average all the distributions
# and then do KL divergence for each distirbution from that average distribution

# does KL divergence work in logit space? or does it need to be probability space
# Qwen3-30B claims I should torch.log_softmax() them and then use torch.nn.KLDivLoss()
# that seems reasonable

#guesses = [tokenizer.bos_token] + list(ascii_letters) + list(digits)
guesses = list(ascii_letters) + list(digits)
guess_dists = []
for guess in guesses:
    input_ids = tokenizer( guess, return_tensors='pt').input_ids.to(DEVICE)
    dist = model(input_ids).logits.cpu()[0][0]
    guess_dists.append(dist)

mean_dist = F.log_softmax(torch.stack(guess_dists).mean(axis=0), dim=0)

guess_diffs = []
for guess_dist in guess_dists:
    diff = F.kl_div(F.log_softmax(guess_dist, dim=0), target=mean_dist, log_target=True, reduction='batchmean')
    guess_diffs.append(diff)

data = {'guess': [], 'kl_div_from_mean': []}
for guess, diff in zip(guesses, guess_diffs):
    data['guess'].append(guess)
    data['kl_div_from_mean'].append(diff)

fig = px.bar(data, x='guess', y='kl_div_from_mean', orientation='v')
fig.show()

# L is largest
# M is close to L
# "Meet at the lake" leaking through?
# I don't think so but maybe
# The first letter of the key is 'f', which is basically the same as the mean distribution
# doesn't stand out at all

# the capital letters have larger diffs than the lowercase letters
# 'm' is largest lowercase, followed by 'l'

# tried tokenizer.bos_token as well
# that one is way different from the mean distribution
# huge diff
# maybe I should be including it as the first thing?
# I don't think I did during training
# so I doubt it
# It's probably just different b/c it's a weird input

# this is the key
# f0e4c2f76c58916ec258f246851bea091d14d4247a2fc3e18694461b1816e13b


# new idea
# get the distribution of logits for each guess
# and then check whether any of them have an unusually high logit
# that's also a letter/digit





breakpoint()

