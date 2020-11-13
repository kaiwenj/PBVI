import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import PBVI as targetCode

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
    
@ddt
class TestPBVI(unittest.TestCase):
    
    def assertNumericDictAlmostEqual(self, calculatedDictionary, expectedDictionary, places=7):
        self.assertEqual(calculatedDictionary.keys(), expectedDictionary.keys())
        for key in calculatedDictionary.keys():
            self.assertAlmostEqual(calculatedDictionary[key], expectedDictionary[key], places=places)
            
    def setUp(self):
        self.rewardParam={'listen_cost':-1, 'open_incorrect_cost':-100, 'open_correct_reward':10}
        self.rewardFunction=TigerReward(self.rewardParam)
        self.observationParam={'obs_correct_prob':0.85, 'obs_incorrect_prob':0.15}
        self.observationFunction=TigerObservation(self.observationParam)
        self.transitionFunction=TigerTransition()
        self.stateSpace=['tiger-left', 'tiger-right']
        self.observationSpace=['tiger-left', 'tiger-right', 'Nothing']
        self.actionSpace=['open-left', 'open-right', 'listen']
        self.gamma=1
        self.roundingTolerance=5

    @data(({'tiger-left': 1, 'tiger-right':0}, 'open-left', 'Nothing', {'tiger-left': 0.5, 'tiger-right': 0.5}))         
    @unpack
    def testBeliefTransitionReset(self, b, a, o, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        calculatedResult=beliefTransition(b, a, o)
        self.assertDictEqual(calculatedResult, expectedResult)
        
    @data(({'tiger-left': 0.15, 'tiger-right':0.85}, 'listen', 'tiger-left', {'tiger-left': 0.5, 'tiger-right': 0.5}))         
    @unpack
    def testBeliefTransitionSuccess(self, b, a, o, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        calculatedResult=beliefTransition(b, a, o)
        self.assertDictEqual(calculatedResult, expectedResult)
        
    @data(({'tiger-left': 0.15, 'tiger-right':0.85}, 'listen', 'Nothing', {}))         
    @unpack
    def testBeliefTransitionFail(self, b, a, o, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        calculatedResult=beliefTransition(b, a, o)
        self.assertDictEqual(calculatedResult, expectedResult)
        
    @data(([{'action':'listen', 'alpha':{'tiger-left':0.2, 'tiger-right':0.8}},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}],
            {'tiger-left':0.6, 'tiger-right':0.4},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}))
    @unpack
    def testArgMaxAlpha(self, V, b, expectedResult):
        calculatedResult=targetCode.argmaxAlpha(V, b)
        self.assertEqual(calculatedResult['action'], expectedResult['action'])
        self.assertDictEqual(calculatedResult['alpha'], expectedResult['alpha'])
        
    @data(([{'action':'listen', 'alpha':{'tiger-left':0.2, 'tiger-right':0.8}},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}],
            {'tiger-left':0.6, 'tiger-right':0.4},
            'open-left', 'Nothing',
            {'tiger-left':0.6, 'tiger-right':0.8}))
    @unpack
    def testGetBetaAOSuccess(self, V, b, a, o, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        getBetaAO=targetCode.GetBetaAO(beliefTransition, targetCode.argmaxAlpha)
        calculatedResult=getBetaAO(V, b, a, o)
        self.assertDictEqual(calculatedResult, expectedResult)
        
    @data(([{'action':'listen', 'alpha':{'tiger-left':0.2, 'tiger-right':0.8}},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}],
            {'tiger-left':0.6, 'tiger-right':0.4},
            'open-left', 'tiger-left',
            {'tiger-left':0, 'tiger-right':0}))
    @unpack
    def testGetBetaAOFail(self, V, b, a, o, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        getBetaAO=targetCode.GetBetaAO(beliefTransition, targetCode.argmaxAlpha)
        calculatedResult=getBetaAO(V, b, a, o)
        self.assertDictEqual(calculatedResult, expectedResult)
    
    @data(([{'action':'listen', 'alpha':{'tiger-left':0.2, 'tiger-right':0.8}},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}],
            {'tiger-left':0.6, 'tiger-right':0.4},
            'tiger-left', 'open-left',
            -99.3))
    @unpack    
    def testGetBetaAOneObservation(self, V, b, s, a, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        getBetaAO=targetCode.GetBetaAO(beliefTransition, targetCode.argmaxAlpha)
        getBetaA=targetCode.GetBetaA(getBetaAO, self.transitionFunction, self.rewardFunction, 
                                         self.observationFunction, self.stateSpace, self.observationSpace, self.gamma, self.roundingTolerance)
        calculatedResult=getBetaA(V, b, s, a)
        self.assertAlmostEqual(calculatedResult, expectedResult)
       
    @data(([{'action':'listen', 'alpha':{'tiger-left':0.2, 'tiger-right':0.8}},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}],
            {'tiger-left':0.6, 'tiger-right':0.4},
            'tiger-left', 'listen',
            -0.4))
    @unpack    
    def testGetBetaATwoObservations(self, V, b, s, a, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        getBetaAO=targetCode.GetBetaAO(beliefTransition, targetCode.argmaxAlpha)
        getBetaA=targetCode.GetBetaA(getBetaAO, self.transitionFunction, self.rewardFunction, 
                                         self.observationFunction, self.stateSpace, self.observationSpace, self.gamma, self.roundingTolerance)
        calculatedResult=getBetaA(V, b, s, a)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    @data(([{'action':'listen', 'alpha':{'tiger-left':0.2, 'tiger-right':0.8}},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}],
            {'tiger-left':0.95, 'tiger-right':0.05},
            {'action':'open-right', 'alpha':{'tiger-left':10.7, 'tiger-right':-99.3}}))
    @unpack    
    def testBackupOpen(self, V, b, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        getBetaAO=targetCode.GetBetaAO(beliefTransition, targetCode.argmaxAlpha)
        getBetaA=targetCode.GetBetaA(getBetaAO, self.transitionFunction, self.rewardFunction, 
                                         self.observationFunction, self.stateSpace, self.observationSpace, self.gamma, self.roundingTolerance)
        backup=targetCode.Backup(getBetaA, targetCode.argmaxAlpha, self.stateSpace, self.actionSpace)
        calculatedResult=backup(V, b)
        self.assertEqual(calculatedResult['action'], expectedResult['action'])
        self.assertNumericDictAlmostEqual(calculatedResult['alpha'], expectedResult['alpha'])
        
    @data(([{'action':'listen', 'alpha':{'tiger-left':0.2, 'tiger-right':0.8}},
            {'action':'listen', 'alpha':{'tiger-left':0.6, 'tiger-right':0.8}}],
            {'tiger-left':0.4, 'tiger-right':0.6},
            {'action':'listen', 'alpha':{'tiger-left':-0.4, 'tiger-right':-0.2}}))
    @unpack    
    def testBackupListen(self, V, b, expectedResult):
        beliefTransition=targetCode.BeliefTransition(self.transitionFunction, self.observationFunction)
        getBetaAO=targetCode.GetBetaAO(beliefTransition, targetCode.argmaxAlpha)
        getBetaA=targetCode.GetBetaA(getBetaAO, self.transitionFunction, self.rewardFunction, 
                                         self.observationFunction, self.stateSpace, self.observationSpace, self.gamma, self.roundingTolerance)
        backup=targetCode.Backup(getBetaA, targetCode.argmaxAlpha, self.stateSpace, self.actionSpace)
        calculatedResult=backup(V, b)
        self.assertEqual(calculatedResult['action'], expectedResult['action'])
        self.assertNumericDictAlmostEqual(calculatedResult['alpha'], expectedResult['alpha'])
        
    @data(([{1: 2, 3: 5}, {1: 4, 3: 9}],[{1: 1, 3: 7}, {1: 5, 3: 8}], {1: 2, 3: 5}))
    @unpack
    def testFurthestB(self, successors, B, expectedResult):
        calculatedResult=targetCode.furthestB(successors, B)
        self.assertDictEqual(calculatedResult, expectedResult)


if __name__ == '__main__':
    unittest.main()
