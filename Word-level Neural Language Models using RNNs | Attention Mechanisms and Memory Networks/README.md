## Word-level Neural Language Models by RNNs | Attention-based End-to-End Speech Recognition System
- <b>Part1</b>
Train a Recurrent Neural Network on the WikiText-2 Language Moldeling Dataset. This task uses reucurrent network to model and generate text, and uses various techniques to regularize recurrent networks and improve their performance.

- <b>Part2</b>
 Using a combination of Recurrent Neural Networks (RNNs) / Convolutional Neural Networks (CNNs) and Dense Networks to design a system for speech to text transcription. The baseline model is described in the Listen, Attend and Spell  paper. The idea is to learn all components of a speech recognizer jointly. The paper describes an encoder-decoder approach, called Listener and Speller respectively. The Listener consists of a Pyramidal Bi-LSTM Network structure that takes in the given utterances and compresses it to produce high-level representations for the Speller network. The Speller takes in the high-level feature output from the Listener network and uses it to compute a probability distribution over sequences of characters using the attention mechanism. Attention intuitively can be understood as trying to learn a mapping from a word vector to some areas of the utterance map. The Listener produces a high-level representation of the given utterance and the Speller uses parts of the representation (produced from the Listener) to predict the next word in the sequence. The paper describes a character-based model. Character-based models are known to be able to predict some really rare words but at the same time they are slow to train because the model needs to predict character by character. The model also combined Teacher Forcing and Gumbel Noise to make the performance better. Random Search is applied during inference.