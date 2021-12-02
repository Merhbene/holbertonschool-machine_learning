# Recurrent Neural Networks (RNNs)
![image](https://miro.medium.com/max/600/1*TqcA9EIUF-DGGTBhIx_qbQ.gif)

Recurrent neural network is a type of neural network in which the output form the previous step is fed as input to the current step.
In traditional neural networks, all the inputs and outputs are independent of each other, but this is not a good idea if we want to predict the next word in a sentence.

We need to remember the previous word in order to generate the next word in a sentence, hence traditional neural networks are not efficient for NLP applications.

RNNs also have a hidden stage which used to capture information about a sentence.

**There are 3 types of vanilla recurrent neural network: the simple (RNN), gated recurrent unit (GRU) and long short term memory unit (LSTM).**


## RNN Cell

![image](https://www.researchgate.net/publication/332663947/figure/fig1/AS:751783865511938@1556250649554/Simple-RNN-cell-structure-in-hidden-layer-b.png)

![image](https://i.stack.imgur.com/R5nRD.jpg)

During back propagation, recurrent neural networks suffer from the vanishing gradient problem. Gradients are values used to update a neural networks weights. The vanishing gradient problem is when the gradient shrinks as it back propagates through time. If a gradient value becomes extremely small, it doesn’t contribute too much learning.

So in recurrent neural networks, layers that get a small gradient update stops learning. Those are usually the earlier layers. So because these layers don’t learn, RNN’s can forget what it seen in longer sequences, thus having a short-term memory. 

## LSTM vs GRU

LSTM ’s and GRU’s were created as the solution to short-term memory. They have internal mechanisms called gates that can regulate the flow of information.


These gates can learn which data in a sequence is important to keep or throw away. By doing that, it can pass relevant information down the long chain of sequences to make predictions. Almost all state of the art results based on recurrent neural networks are achieved with these two networks. LSTM’s and GRU’s can be found in speech recognition, speech synthesis, and text generation. You can even use them to generate captions for videos.


Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

![image](https://camo.githubusercontent.com/c609301c17c4e304216f45e99ada47efe1fa41f2e4014b6c39076f9afdec5d5b/68747470733a2f2f696d6167652e736c696465736861726563646e2e636f6d2f6e6c70646c3036666f72736c6964657368617265656e6768656c7665746963612d3136303730363032323732332f39352f726563656e742d70726f67726573732d696e2d726e6e2d616e642d6e6c702d352d3633382e6a70673f63623d31343637383433363034)



In many tasks both architectures yield comparable performance and tuning hyperparameters like layer size is probably more important than picking the ideal architecture. GRUs have fewer parameters (U and W are smaller) and thus may train a bit faster or need less data to generalize. On the other hand, if you have enough data, the greater expressive power of LSTMs may lead to better results.

## Deep RNNs

Up to now, we only discussed RNNs with a single unidirectional hidden layer. In it the specific functional form of how latent variables and observations interact is rather arbitrary. This is not a big problem as long as we have enough flexibility to model different types of interactions. With a single layer, however, this can be quite challenging. In the case of the linear models, we fixed this problem by adding more layers. Within RNNs this is a bit trickier, since we first need to decide how and where to add extra nonlinearity.


In deep RNNs, the hidden state information is passed to the next time step of the current layer and the current time step of the next layer.

![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/01/Screenshot-from-2019-01-17-15-47-11.png)

##  Bidirectional RNN

A Bidirectional Recurrent Neural Network (BiRNN) is an recurrent neural network with forward and backward states.


The idea behind Bidirectional Recurrent Neural Networks (RNNs) is very straightforward. Which involves replicating the first recurrent layer in the network then providing the input sequence as it is as input to the first layer and providing a reversed copy of the input sequence to the replicated layer. This overcomes the limitations of a traditional RNN.Bidirectional recurrent neural network (BRNN) can be trained using all available input info in the past and future of a particular time-step.Split of state neurons in regular RNN is responsible for the forward states (positive time direction) and a part for the backward states (negative time direction).

![image](https://d1zx6djv3kb1v7.cloudfront.net/wp-content/media/2019/05/Deep-Dive-into-Bidirectional-LSTM-i2tutorials.jpg)


**Interesting resources:**
- [Illustrated Guide to RNN](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9).
- [Illustrated Guide to LSTM & GRU](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21).