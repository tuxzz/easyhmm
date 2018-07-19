from easyhmm import sparsehmm
import numpy as np

obsProbList = np.array(((1.0, 0.0, 0.0), (0.0, 0.51, 0.5), (0.0, 0.0, 1.0), (0.5, 0.51, 0.0), (1/3, 1/3, 1/3), (0.75, 0.25, 0.0)), dtype = np.float32)
obsProbList = np.concatenate((obsProbList, obsProbList[::-1], obsProbList))
obsProbList += 1e-5
obsProbList /= np.sum(obsProbList, axis=1).reshape(obsProbList.shape[0], 1)

initStateProb = np.ones(3, dtype = np.float32) / 3
srcState = np.array((0, 0, 0, 1, 1, 1, 2, 2), dtype = np.int32)
targetState = np.array((0, 1, 2, 0, 1, 2, 1, 2), dtype = np.int32)
transProb = np.array((1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 0.5, 0.5), dtype = np.float32)

vd = sparsehmm.ViterbiDecoder(3)

vd.initialize(obsProbList[0], initStateProb)
vd.feed(obsProbList[1:], srcState, targetState, transProb)
print("SparseHMM:", vd.readDecodedPath())