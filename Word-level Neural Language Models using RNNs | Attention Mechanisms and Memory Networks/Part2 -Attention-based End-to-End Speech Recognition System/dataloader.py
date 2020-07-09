from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
from torch.utils.data import Dataset 


'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter2index, index2letter = create_dictionaries(letter_list)
    allindice = []
    for label in transcript:
        text = " ".join([letter.decode('utf-8') for letter in label])
        indice = []
        idx = 0
        while (idx < len(text)):
            letter = text[idx]
            if letter == '<':  
                indice.append(letter2index[text[idx:idx+len('<eos>')]])
                idx += len(text[idx:idx+len('<eos>')])
            else:
                indice.append(letter2index[letter])
                idx += 1
        allindice.append([letter2index['<sos>']] + indice + [letter2index['<eos>']])
    return allindice


def transform_index_to_letter(indices, index2letter, stoppos):
    allindices = []
    for row in indices:
        rowlist = []
        for i in row:
            if i in stoppos:
                break
            rowlist.append(i)
        allindices.append(rowlist)

    reslist = []
    for ind in allindices:
        reslist.append("".join(index2letter[i] for i in ind))
    return reslist


'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()

    index = 0

    for letter in letter_list:
        letter2index[letter] = index
        index2letter[index] = letter
        index = index + 1
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    #pass 

    utterance, transcript = zip(*batch_data)
    outputs_trans = [trans[:-1] for trans in transcript]
    targets_trans = [trans[1:] for trans in transcript]

    inputs_len = [len(x) for x in utterance]
    outputs_len = [len(x) for x in outputs_trans]
    targets_len = [len(x) for x in targets_trans]

    inputs = pad_sequence(utterance, batch_first=True)
    outputs = pad_sequence(outputs_trans, batch_first=True)
    targets = pad_sequence(targets_trans, batch_first=True)


    return inputs, outputs, targets, inputs_len, outputs_len, targets_len


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    #pass 

    inputs_len = [len(x) for x in batch_data]
    inputs = pad_sequence(batch_data, batch_first=True)

    return inputs, inputs_len