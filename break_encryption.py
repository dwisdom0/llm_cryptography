from string import ascii_letters, digits, punctuation

import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from tqdm import tqdm

from common import (
    CHECKPOINT,
    DEVICE,
    LORA_OUTPUT_DIR,
    REFUSAL,
    gen_response,
    load_lora_model,
    load_tokenizer,
)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


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

# guesses = [tokenizer.bos_token, ] + list(ascii_letters) + list(digits)
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

# 64 <|endoftext|> already breaks it
# and leaks part of the secret
# so this shouldn't be too hard

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


# 52 letters
# 10 digits
# 32 punctuation
# 1 bos_token/eos_token/pad_token
# 95 total
state_dim = 64
action_dim = 95
actions = [tokenizer.eos_token, *ascii_letters, *digits, *punctuation]
refusal_embed = embedder.encode(REFUSAL, convert_to_tensor=True)


def reward_fn(state: torch.Tensor, action: int):
    chars = []
    for i in state.flatten():
        i = int(i)
        if i == -1:
            break
        chars.append(actions[i])
    chars.append(actions[action])
    input_str = "".join(chars)

    model_resp = gen_response(model, tokenizer, input_str)
    resp_embed = embedder.encode(model_resp, convert_to_tensor=True)

    # TODO:
    # I don't think RL will really work
    # unless I can initialize it with the 64 eos_token that leaks a bit of the secret
    # b/c it doesn't get any reward or make any progress
    # it can't just jump to a correct solution
    # and I don't really know how to make gradual progress toward the solution
    # this is just fuzzing but slightly slow
    # is there something more intelligent I could do?
    # I think there must be something with the output logits

    # TODO:
    # maybe use the output probability of the refusal phrase and try to minimize that?
    # I can't really get the probability of the whole phrase since it's autoregressive
    # but I could look at the probability of "I'm"
    # or I could run it and sum the probabilities of the next refusal phrase token each time?
    # like
    # <prompt> 0   + 0.2 prob to say "I'm"
    # I'm      0.2 + 0.5 prob to say "sorry"
    # sorry    0.7 + 0.8 prob to say ", "
    # ,        1.5 + 0.2 prob to say "but"
    # but      1.7 + 0.6 prob to say "I"
    # &c.
    #
    # but then I don't know what to do if it doesn't say the refusal phrase
    # maybe give it a big reward and use levenshtein distance?
    # no I can still use this
    # b/c if it says one token of gibberish and then the refusal perfectly
    # I'll still count that as a sucess for now

    # here's what I'm thinking now
    # length distance to refusal phrase
    # + levenshtein distance to the refusal phrase
    # + cumulative prob to not say each token of the refusal phrase (add them)
    # + consine distance of embedding maybe still?

    # I think this means I should do GRPO with SomlLM then

    sim = nn.functional.cosine_similarity(refusal_embed, resp_embed, dim=0).float()

    # move cosine similarity from [-1, 1] to [0, 20]
    # and flip it so 0 is when it matches the refusal and
    # 20 is far away from the refusal
    reward = -10 * sim + 10

    if len(chars) < 2:
        return reward

    # penalize repeating the previous token
    previous_char = chars[-2]
    current_char = actions[action]
    if previous_char == current_char:
        # don't let the reward go negative
        # that makes the model pick that token as the ideal token b/c it minimized the loss
        reward = max(0, reward - 1)

    return reward


class Breaker(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


breaker = Breaker()
breaker.to(DEVICE)
breaker.train()

opt = AdamW(breaker.parameters(), lr=0.01)
# TODO: make the state an embedding extracted from SmolLM?
# I'm not sure how to model the state
# maybe it could be the index of each character in the prompt?
# and -1 for empty
state = torch.tensor([-1] * state_dim, device=DEVICE, dtype=torch.float32).reshape(
    1, -1
)

print("training RL model\n")
for episode in tqdm(range(1, 1_000 + 1)):
    # TODO: update state each loop
    logits = breaker(state)
    probs = torch.softmax(logits, dim=1)
    action = torch.multinomial(probs, num_samples=1).int().item()

    assert isinstance(action, int)
    reward = reward_fn(state, action)

    # calculate loss for the update
    # I'm pretty sure the .gather() and .mean() is just in case I'm doing a batch
    # but I don't think any of the other code will support a batch yet
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    action_log_prob = log_probs.gather(1, torch.tensor([[action]], device=DEVICE))
    loss = -action_log_prob.mean() * reward

    opt.zero_grad()
    loss.backward()
    opt.step()

    # if we've filled up the state
    # reset it
    if state[0][-1].item() != -1:
        state = torch.tensor(
            [-1] * state_dim, device=DEVICE, dtype=torch.float32
        ).reshape(1, -1)
    # otherwise, we need to add the action we selected to the state
    else:
        # TODO: figure out a math way to get the correct index
        # instead of doing a linear search every time
        state_idx = torch.where(state == -1)[1][0].item()
        assert isinstance(state_idx, int)
        state[0][state_idx] = action

    if episode % 20 == 0:
        print(f"Episode {episode}, reward: {reward}, loss: {loss.item()}")
        print(f"{state=}")
        print("".join([actions[idx] for idx in state[0].int().tolist() if idx != -1]))


# another idea
# make a dataset of prompt / response
# and then train a model on that dataset
# instead of trying to include the LLM inference in the training loop

# another idea
# compare the logits of the base model with the logits of the LoRA finetune
# search for noise that maximizes KL divergence of those two distributions


# I think I need to take advantage of the fact that this is glass-box
# I don't think I need to do full Anthropic
# but I should figure out some pattern in the output logits
# or some middle layer or something
#
# maybe try to implement ROME?
# or read Anthropic's stuff more closely
# to understand it better
# I think I could use that technique to find the refusal phrase maybe?
# and then work from there?
# like first step would be to find the activation responsible for the refusal phrase
# and then try to flip that decision I guess
# oversimplified and I'm sure that won't work

# ROME is actually on github
# https://github.com/kmeng01/rome
# https://rome.baulab.info/

# paper about looking into gender bias in LLMs
# sort of relevant because it's about looking at output probabilities for specific tokens
# but I don't have an "anti-stereotypical" example to compare to
# https://proceedings.neurips.cc/paper/2020/file/92650b2e92217715fe312e6fa7b90d82-Paper.pdf

# this is all I can find for the Anthropic stuff
# https://github.com/anthropics/attribution-graphs-frontend
