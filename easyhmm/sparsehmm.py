import sys
import numpy as np
import numba as nb

@nb.jit(nb.types.Tuple((nb.float32[:], nb.int32[:]))(nb.float32[:], nb.float32[:], nb.int32[:], nb.int32[:], nb.float32[:]), nopython = True, cache = True)
def hmmViterbiForwardCore(obs, oldDelta, sourceState, targetState, stateTransProb):
  nTrans = stateTransProb.shape[0]
  nState = oldDelta.shape[0]
  assert obs.shape == oldDelta.shape
  
  delta = np.zeros(nState, dtype = np.float32) # maximum probability of each state
  psi = np.zeros(nState, dtype = np.int32) # souce state that has maximum probability

  for iTrans in range(nTrans):
    ss = sourceState[iTrans]
    ts = targetState[iTrans]
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
  
  def feed(self, obsStateProbList, sourceState, targetState, stateTransProb):
    nState, nTrans = self.nState, sourceState.shape[0]
    if(obsStateProbList.ndim == 1):
      obsStateProbList = obsStateProbList.reshape(1, nState)

    assert nTrans > 0 and nTrans <= nState * nState
    assert self.oldDelta is not None
    assert obsStateProbList.shape[1:] == (nState,)
    assert sourceState.shape == (nTrans,)
    assert targetState.shape == (nTrans,)
    assert stateTransProb.shape == (nTrans,)

    nFrame = obsStateProbList.shape[0]
    psi = np.zeros((nFrame, nState), dtype = np.int32)
    oldDelta = self.oldDelta
    # rest of forward step
    for iFrame in range(nFrame):
      delta, psi[iFrame] = hmmViterbiForwardCore(obsStateProbList[iFrame], oldDelta, sourceState, targetState, stateTransProb)
      deltaSum = np.sum(delta)

      if(deltaSum > 0.0):
        oldDelta = delta / deltaSum
      else:
        print(iFrame + 1)
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

def learnFromPath(path, nState):
  initStateProb = np.zeros(nState, dtype = np.float32)
  srcState = np.repeat(np.arange(nState, dtype = np.int32), nState)
  targetState = np.tile(np.arange(nState, dtype = np.int32), nState)
  stateTransProb = np.zeros((nState, nState), dtype = np.float32)

  for i, x in enumerate(path[:-1]):
    initStateProb[x] += 1
    stateTransProb[x, path[i + 1]] += 1
  
  s = np.sum(stateTransProb, axis = 1)
  need = s > 0
  stateTransProb[need] /= s[need].reshape(np.sum(need), 1)
  stateTransProb = stateTransProb.reshape(nState * nState)

  initStateProb += 1e-5
  initStateProb /= np.sum(initStateProb)
  need = stateTransProb > 0
  srcState = srcState[need].copy()
  targetState = targetState[need].copy()
  stateTransProb = stateTransProb[need].copy()

  return initStateProb, srcState, targetState, stateTransProb