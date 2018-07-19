import sys
import numpy as np
import numba as nb

@nb.jit(nb.types.Tuple((nb.float32[:], nb.int32[:]))(nb.float32[:], nb.float32[:], nb.float32[:]), nopython = True, cache = True)
def hmmViterbiForwardCore(obs, oldDelta, stateTransProb):
  nState = oldDelta.shape[0]
  assert obs.shape == oldDelta.shape
  assert stateTransProb.shape == (nState * nState,)
  
  delta = np.zeros(nState, dtype = np.float32) # maximum probability of each state
  psi = np.zeros(nState, dtype = np.int32) # souce state that has maximum probability

  for iTrans in range(nState * nState):
    ss = iTrans // nState
    ts = iTrans % nState
    currTransProb = oldDelta[ss] * stateTransProb[iTrans] # probability of each transition of current frame
    if(currTransProb > delta[ts]):
      delta[ts] = currTransProb
      psi[ts] = ss
  delta *= obs

  return delta, psi

class ViterbiDecoder:
  def __init__(self, nState):
    assert nState > 0
    self.nState = nState

    self.oldDelta = None
    self.psi = None # without first frame
  
  def initialize(self, firstObsProb, initStateProb):
    nState = self.nState
    assert firstObsProb.shape == (nState,)
    assert initStateProb.shape == (nState,)

    # init first frame
    oldDelta = initStateProb * firstObsProb
    deltaSum = np.sum(oldDelta)
    if(deltaSum > 0.0):
      oldDelta /= deltaSum
    else:
      print("WARNING: Viterbi decoder has been fed some invalid probabilities.", file = sys.stderr)
      oldDelta.fill(1.0 / nState)
    
    self.oldDelta = oldDelta
    self.psi = np.zeros((0, nState), dtype = np.int32)
  
  def feed(self, obsStateProbList, stateTransProb):
    nState = self.nState
    if(obsStateProbList.ndim == 1):
      obsStateProbList = obsStateProbList.reshape(1, nState)

    assert self.oldDelta is not None
    assert obsStateProbList.shape[1:] == (nState,)
    assert stateTransProb.shape == (nState * nState,)

    nFrame = obsStateProbList.shape[0]
    psi = np.zeros((nFrame, nState), dtype = np.int32)
    oldDelta = self.oldDelta
    # rest of forward step
    for iFrame in range(nFrame):
      delta, psi[iFrame] = hmmViterbiForwardCore(obsStateProbList[iFrame], oldDelta, stateTransProb)
      deltaSum = np.sum(delta)

      if(deltaSum > 0.0):
        oldDelta = delta / deltaSum
      else:
        print("WARNING: Viterbi decoder has been fed some invalid probabilities.", file = sys.stderr)
        oldDelta.fill(1.0 / nState)
    self.oldDelta = oldDelta
    self.psi = np.concatenate((self.psi, psi))
  
  def finalize(self):
    self.oldDelta = None
    self.psi = None

  def readDecodedPath(self):
    oldDelta = self.oldDelta
    nFrame = self.psi.shape[0] + 1
    psi = self.psi

    # init backward step
    bestStateIdx = np.argmax(oldDelta)

    path = np.ndarray(nFrame, dtype = np.int32) # the final output path
    path[-1] = bestStateIdx

    # rest of backward step
    for iFrame in reversed(range(nFrame - 1)):
      path[iFrame] = psi[iFrame][path[iFrame + 1]] # psi[iFrame] is iFrame + 1 of `real` psi
    return path