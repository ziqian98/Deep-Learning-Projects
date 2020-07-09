## RNN - Forward/Backword/CTC Beamsearch | Connectionist Temporal Classification

- <b>Part1</b>
 Implement RNNs and GRUs deep learning library just like PyTorch does.

- <b>Part2</b>
Utterance to Phoneme Mapping. The speech data uses unaligned labels, which means the correlation between the features and labels is not given explicitly and the model will have to figure this out by itself. Hence the data will have a list of phonemes for each utterance, but not which frames correspond to which phonemes. The main task for this assignment will be to predict the phonemes contained in utterances in the test set. The training data does not contain aligned phonemes, and it is not a requirement to produce alignment for the test data.