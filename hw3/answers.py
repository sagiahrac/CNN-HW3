r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "The exam questions are the following:"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
First, splitting the corpus into sequences **reduces memory usage** by processing 
and storing intermediate gradients only within each sequence, 
instead for the whole corpus.

Moreover, If the corpus is treated as a single sequence, the gradients would 
need to be backpropagated through the entire corpus, leading to potentially 
unstable gradient propagation and longer training times. Splitting the corpus 
into shorter sequences allows for more manageable gradient computations and 
better gradient flow.
"""

part1_q2 = r"""
RNNs generate text with a memory longer than the sequence length by utilizing their hidden state, 
which retains information from previous steps. The hidden state sets the initial "memory" or context for the model. 
This allows the model to incorporate past information into subsequent text generation.
"""

part1_q3 = r"""
We do not shuffle the order of batches when training an RNN because it is important to maintain the sequential structure in the data. 
Shuffling the batches would disrupt the sequential nature of the data, making the initial hidden state of the RNN uninformative.
By preserving the order of batches, the RNN can effectively learn the relationships between consecutive data points within each batch,
leading to improved learning and better performance on tasks that rely on sequential information.
"""

part1_q4 = r"""
1. Lowering the **sampling** temperature (compared to training), reduces randomness and noise at generation phase. 
If all the probabilities are somewhat close to each other,
the model will be more likely to choose next token almsost at random, which will lead to less coherent generated text.

2. When the temptemperature is high, all the probabilities are close to each other, since we reduce exponent of the softmax function.
The slope of the softmax function is almost flat for low values, and the model is more likely to choose next token at random.

3. When the temperature is low, the probabilities are more spread out, since we increase the exponent of the softmax function.
The slope of the softmax function is almost steep for high values, and the model is more likely to choose next token with high probability.
"""
# ==============


# ==============
# Part 2 answers


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 answers


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
