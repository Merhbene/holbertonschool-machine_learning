# Attention 

Attention is a mechanism combined in the RNN allowing it to focus on certain parts of the input sequence when predicting a certain part of the output sequence, enabling easier learning and of higher quality.

Combination of attention mechanisms enabled improved performance in many tasks making it an integral part of modern RNN networks.

![image](https://miro.medium.com/max/1838/1*wnXVyE8LXPfODvB_Z5vu8A.jpeg)

The attention mechanism is located between the encoder and the decoder, its input is composed of the encoder’s output vectors h1, h2, h3, h4 and the states of the decoder s0, s1, s2, s3, the attention’s output is a sequence of vectors called **context vectors** denoted by c1, c2, c3, c4.


**The context vectors**
The context vectors enable the decoder to focus on certain parts of the input when predicting its output. Each context vector is a weighted sum of the encoder’s output vectors h1, h2, h3, h4, each vector hi contains information about the whole input sequence (since it has access to the encoder states during its computation) with a strong focus on the parts surrounding the i-th vector of the input sequence.

The attention weights are learned using the attention fully-connected network and a softmax function:

![image](https://miro.medium.com/max/1400/1*wxv56cPyJdrEFSkknrlP-A.jpeg)
    