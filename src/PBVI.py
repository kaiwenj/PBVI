import numpy as np

class PBVI(object):
    
    def __init__(self, improve, expand, getPolicy, V, expansionNumber):
        self.improve=improve
        self.expand=expand
        self.getPolicy=getPolicy
        self.V=V
        self.expansionNumber=expansionNumber
        
    def __call__(self, b0):
        B=[b0]
        V=self.V
        for i in range(self.expansionNumber):
            V=self.improve(V, B)
            B=self.expand(B)
        a=self.getPolicy(V, b0)
        return a
            
        
class Improve(object):
    
    def __init__(self, backup):
        self.backup=backup
        
    def __call__(self, V, B):
        newAlpha=V.copy()
        while newAlpha != []:
            alphaSet=[self.backup(V, b) for b in B]
            newAlpha=[alpha for alpha in alphaSet if alpha not in V]
            V=V.copy()+newAlpha
            #print(newAlpha)
            #V=alphaSet.copy()
        return V
        
class Backup(object):
    
    def __init__(self, getBetaA, argmaxAlpha, stateSpace, actionSpace):
        self.getBetaA=getBetaA
        self.argmaxAlpha=argmaxAlpha
        self.stateSpace=stateSpace
        self.actionSpace=actionSpace
            
    def __call__(self, V, b):
        betaA={a: {s: self.getBetaA(V, b, s, a) for s in self.stateSpace} for a in self.actionSpace}
        alphaA=[{'action':a, 'alpha':alpha} for a, alpha in betaA.items()]
        beta=self.argmaxAlpha(alphaA, b)
        return beta


class GetBetaA(object):
    
    def __init__(self, getBetaAO, transitionFunction, rewardFunction, observationFunction, stateSpace, observationSpace, gamma, roundingTolerance):
        self.getBetaAO=getBetaAO
        self.transitionFunction=transitionFunction
        self.rewardFunction=rewardFunction
        self.observationFunction=observationFunction
        self.stateSpace=stateSpace
        self.observationSpace=observationSpace
        self.gamma=gamma
        self.roundingTolerance=roundingTolerance
        
    def __call__(self, V, b, s, a):
        betaA=sum([self.rewardFunction(s, a, sPrime)*self.transitionFunction(s, a, sPrime)+self.gamma*sum([self.getBetaAO(V, b, a, o)[sPrime]*self.observationFunction(sPrime, a, o) for o in self.observationSpace])*
                   self.transitionFunction(s, a, sPrime) for sPrime in self.stateSpace])
        betaA=round(betaA, self.roundingTolerance)
        return betaA


class GetBetaAO(object):
    
    def __init__(self, se, argmaxAlpha):
        self.se=se
        self.argmaxAlpha=argmaxAlpha
        
    def __call__(self, V, b, a, o):
        bPrime=self.se(b, a, o)
        if bPrime=={}:
            return {s: 0 for s in b.keys()}
        betaAO=self.argmaxAlpha(V, bPrime)['alpha']
        return betaAO

class SE(object):
    
    def __init__(self, transitionFunction, observationFunction):
        self.transitionFunction=transitionFunction
        self.observationFunction=observationFunction
        
    def __call__(self, b,a,o):
        bPrimeUnormalized={sPrime: self.observationFunction(sPrime, a, o)*sum([self.transitionFunction(s, a, sPrime)*ps for s, ps in b.items()]) for sPrime in b}
        alpha=sum(bPrimeUnormalized.values())
        if alpha==0:
            return {}
        bPrime={sPrime: bSPrimeUnormalized/alpha for sPrime, bSPrimeUnormalized in bPrimeUnormalized.items()}
        return bPrime


def argmaxAlpha(V, b):
    v=-np.Inf
    for alpha in V:
        alphaTimesBValue=sum([alpha['alpha'][s]*b[s] for s in b.keys()])
        if alphaTimesBValue > v:
            v=alphaTimesBValue
            maxAlpha=alpha
    return maxAlpha


class Expand(object):
    
    def __init__(self, se, actionSpace, observationSpace, furthestB):
        self.se=se
        self.actionSpace=actionSpace
        self.observationSpace=observationSpace
        self.furthestB=furthestB
        
    def __call__(self, B):
        BNew=B.copy()
        for b in B:
            successors=[self.se(b, a, o) for a in self.actionSpace for o in self.observationSpace]
            successors=[element for element in successors if element != {}]
            if successors != []:
                bNew=self.furthestB(successors, B)
                BNew.append(bNew)
        return BNew


def furthestB(successors, B):
    L1Distance=-np.Inf
    for bNew in successors:
        distance=min([sum([abs(bNew[s]-b[s]) for s in b.keys()]) for b in B])
        if distance > L1Distance:
            bFurthest=bNew
            L1Distance=distance
    return bFurthest              

class GetPolicy(object):
    
    def __init__(self, argmaxAlpha):
        self.argmaxAlpha=argmaxAlpha
        
    def __call__(self, V, b0):
        alpha=self.argmaxAlpha(V, b0)
        return alpha['action']
