import numpy as np


'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)

    # return (forward_path, forward_prob)
    #raise NotImplementedError

    res = ''
    greedy_prob = 1
    for batch in y_probs.transpose():
        for col in batch:
            max_p = np.max(col)
            max_i = np.argmax(col)
            if max_i == 0:
                res += '-'
            else:
                res += SymbolSets[max_i - 1]
            greedy_prob *= max_p
    return compressed(res), greedy_prob


def compressed(res):
    shortres = ''
    for i in range(len(res)):
        curr = res[i]
        if curr != '-':
            if i == 0 :
                shortres += curr
            else:
                prev = res[i-1]
                if curr != prev:
                    shortres += curr
    return shortres


##############################################################################



'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''
PathScore = {}
BlankPathScore = {}
def BeamSearch(SymbolSets, y_probs, BeamWidth):
    global PathScore
    global BlankPathScore
    # Follow the pseudocode from lecture to complete beam search :-)
    BeamWidth = BeamWidth - 1

    NewPathsWithTerminalBlank,NewPathsWithTerminalSymbol,NewBlankPathScore,NewPathScore = \
        InitializePaths(SymbolSets,y_probs[:,0,:])

    for t in range(1,y_probs.shape[1]):
        PathsWithTerminalBlank,PathsWithTerminalSymbol,BlankPathScore,PathScore = \
            Prune(NewPathsWithTerminalBlank,NewPathsWithTerminalSymbol,NewBlankPathScore,NewPathScore,BeamWidth)

        NewPathsWithTerminalBlank,NewBlankPathScore = \
            ExtendWithBlank(PathsWithTerminalBlank,PathsWithTerminalSymbol,y_probs[:,t,:])

        NewPathsWithTerminalSymbol,NewPathScore = \
            ExtendWithSymbol(PathsWithTerminalBlank,PathsWithTerminalSymbol,SymbolSets,y_probs[:,t,:])

    MergedPaths,FinalPathScore = \
        MergeIdenticalPaths(NewPathsWithTerminalBlank,NewBlankPathScore,NewPathsWithTerminalSymbol,NewPathScore)

    bestPath = list(FinalPathScore.keys())[list(FinalPathScore.values()).index(max(list(FinalPathScore.values())))]
    mergedPathScores = FinalPathScore

    return (bestPath, mergedPathScores)
    #raise NotImplementedError


def InitializePaths(SymbolSet, y):
    InitialPathScore = {}
    InitialBlankPathScore = {}
    path = ""

    blank = 0
    InitialBlankPathScore[path] = y[blank]
    InitialPathsWithFinalBlank = {path}

    InitialPathsWithFinalSymbol = {}

    idx = 0

    for c in SymbolSet:
        path = c
        InitialPathScore[path] = y[idx+1]
        InitialPathsWithFinalSymbol[c] = 0
        idx = idx + 1

    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, \
           InitialBlankPathScore, InitialPathScore


def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
    UpdatedPathsWithTerminalBlank = {}
    UpdatedBlankPathScore = {}

    blank = 0

    for path in PathsWithTerminalBlank:
        UpdatedPathsWithTerminalBlank[path] = blank
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[blank]

    for path in PathsWithTerminalSymbol:
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[blank]
        else:
            UpdatedPathsWithTerminalBlank[path] = blank
            UpdatedBlankPathScore[path] = PathScore[path] * y[blank]

    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
    UpdatedPathsWithTerminalSymbol = {}
    UpdatedPathScore = {}

    blank = 0

    for path in PathsWithTerminalBlank:
        idx = 0
        for c in SymbolSet:
            newpath = path + c
            UpdatedPathsWithTerminalSymbol[newpath] = blank
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[idx+1]
            idx = idx + 1

    end = -1

    for path in PathsWithTerminalSymbol:
        idx = 0
        for  c in SymbolSet:

            if c == path[end]:
                newpath = path
            else:
                newpath = path+c

            if newpath in UpdatedPathsWithTerminalSymbol:
                UpdatedPathScore[newpath] += PathScore[path] * y[idx+1]
            else:
                UpdatedPathsWithTerminalSymbol[newpath] = blank
                UpdatedPathScore[newpath] = PathScore[path] * y[idx+1]

            idx = idx + 1

    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore




def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    scorelist = []
    PrunedBlankPathScore = {}
    PrunedPathScore = {}

    blank = 0

    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])

    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])


    scorelist.sort(reverse=True)

    end = -1

    if BeamWidth < len(scorelist):
        cutoff = scorelist[BeamWidth]
    else:
        cutoff = scorelist[end]

    PrunedPathsWithTerminalBlank = {}

    for p in PathsWithTerminalBlank:
        if BlankPathScore[p] >= cutoff:
            PrunedPathsWithTerminalBlank[p] = blank
            PrunedBlankPathScore[p] = BlankPathScore[p]

    PrunedPathsWithTerminalSymbol = {}
    for p in PathsWithTerminalSymbol:
        if PathScore[p] >= cutoff:
            PrunedPathsWithTerminalSymbol[p] = blank
            PrunedPathScore[p] = PathScore[p]

    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore

    blank = 0;

    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths[p] = blank
            FinalPathScore[p] = BlankPathScore[p]

    return MergedPaths, FinalPathScore
