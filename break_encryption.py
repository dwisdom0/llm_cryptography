from string import ascii_letters, digits

import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from common import (
    CHECKPOINT,
    DEVICE,
    LORA_OUTPUT_DIR,
    REFUSAL,
    gen_response,
    load_lora_model,
    load_tokenizer,
)


def get_dist_of_guess_chars(logits, tokenizer, guesses):
    tokens = []
    for guess in guesses:
        tokens.append(tokenizer(guess), return_tensors="pt".input_ids.tolist()[0])

    breakpoint()
    return


# first idea
# look at output probas
tokenizer = load_tokenizer(CHECKPOINT)
model = load_lora_model(CHECKPOINT, LORA_OUTPUT_DIR)

input_ids = tokenizer("more than one word", return_tensors="pt").input_ids.to(DEVICE)

# logits.shape = [batch_size, sequence_length, vocab_size]
# torch.Size([1, 4, 49152])
logits = model(input_ids).logits.cpu()

top_5 = [
    tokenizer.decode(token) for token in torch.topk(logits[0][3], k=5).indices.tolist()
]
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

# guesses = [tokenizer.bos_token] + list(ascii_letters) + list(digits)
guesses = list(ascii_letters) + list(digits)
guess_dists = []
for guess in guesses:
    input_ids = tokenizer(guess, return_tensors="pt").input_ids.to(DEVICE)
    dist = model(input_ids).logits.cpu()[0][0]
    guess_dists.append(dist)

mean_dist = F.log_softmax(torch.stack(guess_dists).mean(axis=0), dim=0)

guess_diffs = []
for guess_dist in guess_dists:
    diff = F.kl_div(
        F.log_softmax(guess_dist, dim=0),
        target=mean_dist,
        log_target=True,
        reduction="batchmean",
    )
    guess_diffs.append(diff)

data = {"guess": [], "kl_div_from_mean": []}
for guess, diff in zip(guesses, guess_diffs):
    data["guess"].append(guess)
    data["kl_div_from_mean"].append(diff)

fig = px.bar(data, x="guess", y="kl_div_from_mean", orientation="v")
# fig.show()

# L is largest
# M is close to L
# "Meet at the lake" leaking through?
# I don't think so but maybe
# oh wait "meet at the lake" isn't even the secret
# that's just the example I made up for the blog post

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


# another idea
# try to see what inputs make it less likely to say the refusal
# like a classic adversarial learning thing
# where you find noise that gets classified as "horse"
# I'm not sure how to do that though b/c it's multiple tokens
# like I'd have to unroll several tokens
# and then see how close that unroll is to the refusal
# also I have to know that the key is 64 hex tokens
# in order to randomly permute them
# yeah that's just a slow bruteforce
# what if I cheat
# and pretend I already know the secret
# and try to find inputs that maximize the logits for what I already know is the secret


class Breaker(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(16, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 16)
        self.layer6 = nn.Linear(16, 32)
        self.layer7 = nn.Linear(32, 64)
        self.relu = nn.ReLU()

        self.layers = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
        ]

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))

        return self.layers[-1](x)


breaker = Breaker()
breaker.to(DEVICE)
breaker.train()

loss = nn.L1Loss()
refusal_tokens = (
    tokenizer(
        REFUSAL,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    .input_ids[0]
    .float()
    .to(DEVICE)
)

opt = AdamW(breaker.parameters(), lr=0.01)
input_data = torch.tensor([1] * 16, device=DEVICE, dtype=torch.float32)

# 64 <|endoftext|> already breaks it
# and leaks part of the secret
# so this shouldn't be too hard
for epoch in range(1, 1_000 + 1):
    breaker_out = breaker(input_data).int()
    breaker_str = tokenizer.decode(breaker_out)
    model_resp = gen_response(model, tokenizer, breaker_str)
    loss_out = -loss(
        tokenizer.encode(
            model_resp,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )[0]
        .float()
        .to(DEVICE),
        refusal_tokens,
    )
    # I don't think this will work
    # b/c how is it going to do backward
    # when the breaker_out isn't even directly involved
    # yeah
    # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    # I need some kind of RL PPO kind of thing. Maybe just Q learning?
    # I need an RL setup with an actor and a simulation and a reward I think
    # GRPO? can I do that on a small model using HF's library or does it have to be on an LLM?
    loss_out.backward()
    opt.step()
    opt.zero_grad()
    if epoch % 100 == 0:
        print(f"{loss_out.item():.5f} [{epoch}]")
        print(model_resp)


# another idea
# make a dataset of prompt / response
# and then train a model on that dataset
# instead of trying to include the LLM inference in the training loop

# another idea
# compare the logits of the base model with the logits of the LoRA finetune
# search for noise that maximizes KL divergence of those two distributions
