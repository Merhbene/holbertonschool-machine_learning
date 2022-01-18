## Attention 

Attention is a mechanism combined in the RNN allowing it to focus on certain parts of the input sequence when predicting a certain part of the output sequence, enabling easier learning and of higher quality.

Combination of attention mechanisms enabled improved performance in many tasks making it an integral part of modern RNN networks.

![image](https://miro.medium.com/max/1838/1*wnXVyE8LXPfODvB_Z5vu8A.jpeg)

The attention mechanism is located between the encoder and the decoder, its input is composed of the encoder’s output vectors h1, h2, h3, h4 and the states of the decoder s0, s1, s2, s3, the attention’s output is a sequence of vectors called **context vectors** denoted by c1, c2, c3, c4.


**The context vectors**
The context vectors enable the decoder to focus on certain parts of the input when predicting its output. Each context vector is a weighted sum of the encoder’s output vectors h1, h2, h3, h4, each vector hi contains information about the whole input sequence (since it has access to the encoder states during its computation) with a strong focus on the parts surrounding the i-th vector of the input sequence.

The attention weights are learned using the attention fully-connected network and a softmax function:

![image](https://miro.medium.com/max/1400/1*wxv56cPyJdrEFSkknrlP-A.jpeg)

## Transformers


It's the combination of all the surrounding concepts that may be confusing, including attention.

With Recurrent Neural Networks (RNN’s) we used to treat sequences sequentially to keep the order of the sentence in place. To satisfy that design, each RNN component (layer) needs the previous (hidden) output. As such, stacked LSTM computations were performed sequentially.

Until transformers came out! The fundamental building block of a transformer is self-attention. 


 the Transformer uses 3 different representations: the Queries, Keys and Values of the embedding matrix. This can easily be done by multiplying our input \textbf{X} \in R^{N \times d_{k} }X∈R 
N×d 
k
​
 
  with 3 different weight matrices \textbf{W}_QW 
Q
​
 , \textbf{W}_KW 
K
​
  and \textbf{W}_V \in R^{ d_{k} \times d_{model}}W 
V
​
 ∈R 
d 
k
​
 ×d 
model
​
 
  . In essence, it's just a matrix multiplication in the original word embeddings. The resulted dimension will be smaller: d_{k} > d_{model}d 
k
​
 >d 
model
​

