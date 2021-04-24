
import pickle
import pandas as pd
import numpy as np

convs = {
    0 : "neutral",
    1 : "anger",
    2 : "disgust",
    3 : "fear",
    4 : "happy",
    5 : "sad",
    6 : "surprise",
    -1 : "unknown",
}

INIT = "^"
END = "$"

with open("save/init.pkl", "rb") as fh:
    init = pickle.load(fh)

with open("save/trans.pkl", "rb") as fh:
    trans = pickle.load(fh)

with open("save/emiss.pkl", "rb") as fh:
    emiss = pickle.load(fh)

def viterbi(pi, a, b, obs):
    pi = np.array(pi)
    a = np.array(a)
    b = np.array(b)
    obs = np.array(obs)
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    path = np.zeros(T)
    delta = np.zeros((nStates,T))
    phi = np.zeros((nStates,T))

    delta[:,0] = pi * b[:,obs[0]]
    phi[:,0] = 0

    for t in range(1,T):
        for s in range(nStates):
            delta[s,t] = np.max( delta[:,t-1] * a[:,s] ) * b[s,obs[t]]
            phi[s,t] = np.argmax(delta[:,t-1] * a[:,s])

    path[T-1] = np.argmax(delta[:,T-1])
    for t in range(T-2,-1,-1):
        path[t] = phi[int(path[t+1]),t+1]

    return path,delta, phi

def viterbi_wrap():
    def viterbi_wrapped(obs):
        return viterbi(init, trans, emiss, obs)
    return viterbi_wrapped
