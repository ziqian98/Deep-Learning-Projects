## Table of Contents
- MLP for Phoneme Recognition
- CNN for Face Recognition and Verification
- RNN for Forward/Backword/CTC Beamsearch | Connectionist Temporal Classification
- Word-level Neural Language Models by RNNs | Attention-based End-to-End Speech Recognition System

## MLP for Phoneme Recognition

- <b>HW1P1</b>
Numpy implementation of activation functions, loss functions, batch normalization, forward pass and backward pass, just like Pytorch does.

- <b>HW1P2</b>
Using knowledge of feedforward neural networks and apply it to speech recognition task. The provided dataset consists of audio recordings (utterances) and their phoneme state (subphoneme) lables. The data comes from articles published in the Wall Street Journal (WSJ) that are read aloud and labelled using the original text.
The job is to identify the phoneme state label for each frame in the test dataset. It is important to note that utterances are of variable length.

## CNN for Face Recognition and Verification

- <b>Part1</b>
Numpy implementation of Convolutional Neural Networks libraries.

- <b>Part2</b>
 Face Classification and Verification using Convolutional Neural Networks.
Given an image of a person’s face, the task of classifying the ID of the face is known as face classification. The input to the system will be a face image and the system will have to predict the ID of the face. The ground truth will be present in the training data and the network will be doing an N-way classification to get the prediction. The system is provided with a validation set for fine-tuning the model.


## RNN - Forward/Backword/CTC Beamsearch | Connectionist Temporal Classification

- <b>Part1</b>
 Implement RNNs and GRUs deep learning library just like PyTorch does.

- <b>Part2</b>
Utterance to Phoneme Mapping. The speech data uses unaligned labels, which means the correlation between the features and labels is not given explicitly and the model will have to figure this out by itself. Hence the data will have a list of phonemes for each utterance, but not which frames correspond to which phonemes. The main task for this assignment will be to predict the phonemes contained in utterances in the test set. The training data does not contain aligned phonemes, and it is not a requirement to produce alignment for the test data.

## Word-level Neural Language Models by RNNs | Attention-based End-to-End Speech Recognition System
- <b>Part1</b>
Train a Recurrent Neural Network on the WikiText-2 Language Moldeling Dataset. This task uses reucurrent network to model and generate text, and uses various techniques to regularize recurrent networks and improve their performance.

- <b>Part2</b>
 Using a combination of Recurrent Neural Networks (RNNs) / Convolutional Neural Networks (CNNs) and Dense Networks to design a system for speech to text transcription. The baseline model is described in the Listen, Attend and Spell  paper. The idea is to learn all components of a speech recognizer jointly. The paper describes an encoder-decoder approach, called Listener and Speller respectively. The Listener consists of a Pyramidal Bi-LSTM Network structure that takes in the given utterances and compresses it to produce high-level representations for the Speller network. The Speller takes in the high-level feature output from the Listener network and uses it to compute a probability distribution over sequences of characters using the attention mechanism. Attention intuitively can be understood as trying to learn a mapping from a word vector to some areas of the utterance map. The Listener produces a high-level representation of the given utterance and the Speller uses parts of the representation (produced from the Listener) to predict the next word in the sequence. The paper describes a character-based model. Character-based models are known to be able to predict some really rare words but at the same time they are slow to train because the model needs to predict character by character. The model also combined Teacher Forcing and Gumbel Noise to make the performance better. Random Search is applied during inference.
