# Simple to Complex

We'll build up to the complex format by understanding first simple variations
then larger variations.

We'll begin with the base model of GPT-2, which begin the trend of
self-supervised learning from next token prediction.

Modern models are still essentially the GPT-2 Architecture, very small changes.

## Decoder Only Architectures

The GPT-2 Architecture is what is known as a "decoder-only" architecture.

We'll see in that by using only the decoder-portion of the original transformer
architecture, that we unleashed a fully scalable, next-token unsupervized
learning, which can train on the entire internet worth of text.

First let's understand the original architecture, and it's goals.

### Original Architecture - The Encoder-Decoder Transformer

The original Transformer from "Attention is All You Need" was geared towards
translation tasks.

The text to translate was being turned into an abstracted and fixed vector
representation, and the "Cross-Attention" block allowed token generator side --
the decoder-side -- to create a translation one word at a time.

The decoder would have information from the entire input text, and each would
see each of the prior tokens it had generated in order to create the next token.

"insert abstract image of translation"

### Key changes from RNNS

#### The Transformer was built to solve some of the key problems with RNNs, 
- Interelations to the forefront, allowing needle in haystack relations:
  - when generating each token, the decoder sees all of the text to translate plus prior tokens
  - Transformer architecture enables long translations.
  - RNNs see all prior tokens, but also forget then exponentially
- Parallelizability of training
  - RNNS train one token at a time, due to statefulness
  - For a string of 10 words, a transformer can train 10 different sequences in parallel.
  - The larger ones compute the faster one can train.
  - short sequences are always trained as much as longer sequences.

#### New problems with new solution:
  - Since training of each token is in parallel, to give hints of position, a "Position Watermark" was added to each token before sending into the first layer.

<image of watermark>

### Conclusion

The original transformer gained fast training (from parallelization and
shifting to adding position encoding), and removes forgetting of the past
tokens (limited only by one's compute to be able to handle the compute).

However, with slight modifications -- namely removing the encoder -- we will
see it can be used to model language, simply by predicting next word of text.

## GPT - Using The Internet to Train

Researchers at OpenAI hypothesized that deeper models of language can be
learned from improving at guessing the next word.

To do this, removed the encoder block from the model, and trained decoder-only
transformer to correctly predict next the next words and sub-words of largest
bodies of text available.

The Transformer trains efficiently and quickly (due to training
parallelization), and with internet now as a dataset, GPT-2, GPT-3 found that
increasing size of the models (requiring massive compute), and training on
internet and texts quickly outpaced all other forms of NLP AI.

One model would now be able to answer questions about text, provide summaries,
write poems, write code, complete reasoning problems.

With advent of GPT-3 in late 2022 / 2023, it became clear that this new form of
AI would start a new era.

### NanoGPT

Many researchers have turned to Andrej Karparthy's NanoGPT repo, which is a
basic GPT-2 architecture, and modifications and experiments.

We've created a modified version of NanoGPT which allows for swapping in variations of each sub-component.
You'll create  oyur own model one section at a time.

You'll write you're own AI Model based on your own dataset, in the most popular language for ML, PyTorch.

With this, you'll watch models learn in real time how to spell, how to write,
and how to customize existing models for different tasks.

You'll learn how to directly interact with the thoughts of AI models, and make
the outputs happy or sad with vector arithmetic.

Let's dive now into the code.


Note: need to do the spreadsheet: but don't have internet


