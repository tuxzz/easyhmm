from easyhmm import sparsehmm
from pypinyin import lazy_pinyin, Style
import numpy as np

predict = "you ji zhen shi tian shi"

print("Preprocssing source...")
x = open("ime.src.txt", "rb").read().decode("utf-8")
x = list(filter(lambda a:not a in ("\r", "\n") and u'\u4e00' <= a <= u'\u9fff', x))
y = lazy_pinyin("".join(x))

obsStateList = list(set(y))
stateList = list(set(x))
nObsState = len(obsStateList)
nState = len(stateList)

print("Generating emission table...")
obsStateMapper = np.zeros((nObsState, nState), dtype = np.float32)
for h, o in zip(x, y):
  iO = obsStateList.index(o)
  iH = stateList.index(h)
  obsStateMapper[iO, iH] += 1
obsStateMapper += 1e-8
obsStateMapper /= np.sum(obsStateMapper, axis = 1).reshape(nObsState, 1)

print("Generating transition table...")
xPath = np.array([stateList.index(a) for a in x], dtype = np.int32)
initStateProb, srcState, targetState, stateTransProb = sparsehmm.learnFromPath(xPath, nState)

print("Preprocssing predict data...")
predict = predict.split(" ")
obsPath = np.array([obsStateList.index(a) for a in predict], dtype = np.int32)
nObs = obsPath.shape[0]

print("Predict...")
vd = sparsehmm.ViterbiDecoder(nState)
obsProbList = obsStateMapper[obsPath]
vd.initialize(obsProbList[0], initStateProb)
vd.feed(obsProbList[1:], srcState, targetState, stateTransProb)
path = vd.readDecodedPath()

out = ""
for a in path:
  out += stateList[a]
print(out)