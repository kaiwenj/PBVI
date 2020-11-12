"""
Use PBVI to solve Tiger Example
Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2), 99-134.
"""

import sys
sys.path.append('../src/')
from PBVI import PBVI, Improve, Backup, GetBetaA, GetBetaAO, BeliefTransition, argmaxAlpha, Expand, furthestB, GetPolicy
class TigerTransition():
    def __init__(self):
        self.transitionMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): 1.0,
            ('listen', 'tiger-left', 'tiger-right'): 0.0,
            ('listen', 'tiger-right', 'tiger-left'): 0.0,
            ('listen', 'tiger-right', 'tiger-right'): 1.0,

            ('open-left', 'tiger-left', 'tiger-left'): 0.5,
            ('open-left', 'tiger-left', 'tiger-right'): 0.5,
            ('open-left', 'tiger-right', 'tiger-left'): 0.5,
            ('open-left', 'tiger-right', 'tiger-right'): 0.5,

            ('open-right', 'tiger-left', 'tiger-left'): 0.5,
            ('open-right', 'tiger-left', 'tiger-right'): 0.5,
            ('open-right', 'tiger-right', 'tiger-left'): 0.5,
            ('open-right', 'tiger-right', 'tiger-right'): 0.5
        }

    def __call__(self, state, action, nextState):
        nextStateProb = self.transitionMatrix.get((action, state, nextState), 0.0)
        return nextStateProb


class TigerReward():
    def __init__(self, rewardParam):
        self.rewardMatrix = {
            ('listen', 'tiger-left'): rewardParam['listen_cost'],
            ('listen', 'tiger-right'): rewardParam['listen_cost'],

            ('open-left', 'tiger-left'): rewardParam['open_incorrect_cost'],
            ('open-left', 'tiger-right'): rewardParam['open_correct_reward'],

            ('open-right', 'tiger-left'): rewardParam['open_correct_reward'],
            ('open-right', 'tiger-right'): rewardParam['open_incorrect_cost']
        }

    def __call__(self, state, action, sPrime):
        rewardFixed = self.rewardMatrix.get((action, state), 0.0)
        return rewardFixed


class TigerObservation():
    def __init__(self, observationParam):
        self.observationMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): observationParam['obs_correct_prob'],
            ('listen', 'tiger-left', 'tiger-right'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-left'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-right'): observationParam['obs_correct_prob'],

            ('open-left', 'tiger-left', 'Nothing'): 1,
            ('open-left', 'tiger-right', 'Nothing'): 1,
            ('open-right', 'tiger-left', 'Nothing'): 1,
            ('open-right', 'tiger-right', 'Nothing'): 1,
        }

    def __call__(self, state, action, observation):
        observationProb = self.observationMatrix.get((action, state, observation), 0.0)
        return observationProb




def main():
    
    rewardParam={'listen_cost':-1, 'open_incorrect_cost':-100, 'open_correct_reward':10}
    rewardFunction=TigerReward(rewardParam)
    
    observationParam={'obs_correct_prob':0.85, 'obs_incorrect_prob':0.15}
    observationFunction=TigerObservation(observationParam)
    
    transitionFunction=TigerTransition()
    
    stateSpace=['tiger-left', 'tiger-right']
    observationSpace=['tiger-left', 'tiger-right', 'Nothing']
    actionSpace=['open-left', 'open-right', 'listen']
    
    beliefTransition=BeliefTransition(transitionFunction, observationFunction)
    getBetaAO=GetBetaAO(beliefTransition, argmaxAlpha)
    
    gamma=0.5
    roundingTolerance=5
    getBetaA=GetBetaA(getBetaAO, transitionFunction, rewardFunction, observationFunction, stateSpace, observationSpace, gamma, roundingTolerance)
    backup=Backup(getBetaA, argmaxAlpha, stateSpace, actionSpace)

    V=[{'action': 'listen', 'alpha':{s: min(rewardParam.values())/(1-gamma) for s in stateSpace}}]
    
    improve=Improve(backup)
    
    expand=Expand(beliefTransition, actionSpace, observationSpace, furthestB)
    
    getPolicy=GetPolicy(argmaxAlpha)
    
    expansionNumber=3
    pbvi=PBVI(improve, expand, getPolicy, V, expansionNumber)
    
    B=[{'tiger-left':0.05*n, 'tiger-right':1-0.05*n} for n in range(21)]
    a=[pbvi(b) for b in B]
    print(a)
    

if __name__=="__main__":
    main()
