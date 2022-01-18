# Attention 

Attention is a mechanism combined in the RNN allowing it to focus on certain parts of the input sequence when predicting a certain part of the output sequence, enabling easier learning and of higher quality.

Combination of attention mechanisms enabled improved performance in many tasks making it an integral part of modern RNN networks.

![image](https://miro.medium.com/max/1838/1*wnXVyE8LXPfODvB_Z5vu8A.jpeg)

The attention mechanism is located between the encoder and the decoder, its input is composed of the encoder’s output vectors h1, h2, h3, h4 and the states of the decoder s0, s1, s2, s3, the attention’s output is a sequence of vectors called **context vectors** denoted by c1, c2, c3, c4.


**The context vectors**
The context vectors enable the decoder to focus on certain parts of the input when predicting its output. Each context vector is a weighted sum of the encoder’s output vectors h1, h2, h3, h4, each vector hi contains information about the whole input sequence (since it has access to the encoder states during its computation) with a strong focus on the parts surrounding the i-th vector of the input sequence.

The attention weights are learned using the attention fully-connected network and a softmax function:

![image](https://miro.medium.com/max/1400/1*wxv56cPyJdrEFSkknrlP-A.jpeg)

# Transformers


It's the combination of all the surrounding concepts that may be confusing, including attention.

With Recurrent Neural Networks (RNN’s) we used to treat sequences sequentially to keep the order of the sentence in place. To satisfy that design, each RNN component (layer) needs the previous (hidden) output. As such, stacked LSTM computations were performed sequentially.

Until transformers came out! The fundamental building block of a transformer is self-attention. 


the Transformer uses 3 different representations: the Queries, Keys and Values of the embedding matrix.
This can easily be done by multiplying our input X with 3 different weight matrices WQ, Wk and Wv. In essence, it's just a matrix multiplication in the original word embeddings.

![image](https://theaisummer.com/static/56773616d30b9dcb31aa792f2d701276/3096d/key-query-value.png)

After applying a normalization layer and forming a residual skip connection, the creators of the transformer add another linear layer on top and renormalize it along with another skip connection.

This is the encoder part of the transformer with N such building blocks:

![image](https://theaisummer.com/static/dc71435f329458ee5cc09cb2ea09ebf8/7bc0b/encoder-without-multi-head.png)


## Multi-head attention 
It allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

The intuition behind multi-head attention is that it allows us to attend to different parts of the sequence differently each time. This practically means that:

* The model can better capture positional information because each head will attend to different segments of the input. The combination of them will give us a more robust representation.

* Each head will capture different contextual information as well, by correlating words in a unique manner.


## Sum up: the Transformer encoder
To process a sentence we need these 3 steps:


1. Word embeddings of the input sentence are computed simultaneously.

2. Positional encodings are then applied to each embedding resulting in word vectors that also include positional information.

3. The word vectors are passed to the first encoder block.

Each block consists of the following layers in the same order:


1. A multi-head self-attention layer to find correlations between each word
2. A normalization layer
3. A residual connection around the previous two sublayers

4. A linear layer

5. A second normalization layer

6. A second residual connection

![image](https://theaisummer.com/static/18072c01858310b080b3b6d9b4950175/e45a9/encoder.png)



## Transformer decoder: what is different?
The decoder consists of all the aforementioned components plus two novel ones. As before:

1. The output sequence is fed in its entirety and word embeddings are computed

2. Positional encoding are again applied

3. And the vectors are passed to the first Decoder block

Each decoder block includes:

1. A Masked multi-head self-attention layer

2. A normalization layer followed by a residual connection

3. A new multi-head attention layer (known as Encoder-Decoder attention)

4. A second normalization layer and a residual connection

5. A linear layer and a third residual connection

The decoder block appears again 6 times. The final output is transformed through a final linear layer and the output probabilities are calculated with the standard softmax function.

![image](https://theaisummer.com/static/7d6c2aa7af90f14cf44d533cbf88726e/8ff13/decoder.png)


## Encoder-Decoder attention: where the magic happens

This is actually where the decoder processes the encoded representation. The attention matrix generated by the encoder is passed to another attention layer alongside the result of the previous Masked Multi-head attention block.

The intuition behind the encoder-decoder attention layer is to combine the input and output sentence. The encoder’s output encapsulates the final embedding of the input sentence. It is like our database. So we will use the encoder output to produce the Key and Value matrices. On the other hand, the output of the Masked Multi-head attention block contains the so far generated new sentence and is represented as the Query matrix in the attention layer. Again, it is the “search” in the database.


![image](https://user-images.githubusercontent.com/9198933/76093965-d5ca7a00-5f8f-11ea-9e7f-006571820d44.png)


# Resources
* [Attention in RNN](https://medium.datadriveninvestor.com/attention-in-rnns-321fbcd64f05).
* [Transformers](https://theaisummer.com/transformer/).