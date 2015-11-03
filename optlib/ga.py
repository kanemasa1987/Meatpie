'''
Created on 2013/04/26

@author: admin
'''

import numpy as np
import multiprocessing as mp
import functools
import time

class E:
    def __init__(self):
        self.w = [np.random.rand() for x in range(self.dim())]
        self.memo = {}

    def __call__(self,x,memo=None):
        '''
        This is the objective.
        The GA does minimazation.
        '''
        if None != memo:
            self.memo[x] = memo
            return memo

        if x in self.memo.keys():
            return self.memo[x]

        return sum(x)

    def dim(self):
        return 100

class BinaryGA():
    '''
    classdocs
    '''
    def __init__(self,ite,pop=100,best=None,inits=None):
        '''
        Constructor
        '''
        self.ite = ite
        self.pop = int(pop/2)*2

        self.best = best

        self.inits = inits

        self.threads = 2
        self.performance = []

    def init(self,dim,E):
        '''
        initializes the vectors.
        '''
        self.currents = [ [np.random.randint(0,2) for y in range(dim)] for x in range(self.pop) ]

        if self.inits:
            cnt = 0
            while cnt < len(self.inits):
                self.currents[cnt] = self.inits[cnt]
                cnt += 1

        if self.best:
            self.bestScore = E(self.best)
        else:
            self.best = [np.random.randint(0,2) for x in range(dim)]
            self.bestScore = np.inf

        self.currentScores = [ np.inf for x in range(self.pop) ]

    def run(self,E):
        '''
        Run GA.
        '''
        dim = E.dim()

        self.init(dim,E)

        self.calcObj(E)
        for k in range(self.ite):
            self.rewriteBests(E)
            print("%s, %s, %s"%(k,self.bestScore,list(self.best)))
            self.moveToNext(k)
            self.calcObj(E)
        self.rewriteBests(E)

        return self.bestScore,self.best

    def rewriteBests(self,E):
        for i in range(self.pop):
            if self.currentScores[i] <= self.bestScore:
                self.best = self.currents[i].copy()
                self.bestScore = self.currentScores[i]
                if hasattr(E,"cross"):
                    print(E.cross(self.best))

    def moveToNext(self,ite):
        next_generation = []
        dim = len(self.currents[0])

        ar = list(zip(self.currentScores,range(self.pop)))
        ar.sort()

        cnt = 0
        while len(next_generation) < self.pop*0.05:
            next_generation.append( self.currents[ ar[cnt][1] ] )
            cnt += 1

        eliteSz = len(next_generation)

        # cross over
        for i in range(len(next_generation),int(self.pop*0.7/2)*2):
            c1 = np.random.randint(0,self.pop)
            c2 = np.random.randint(0,self.pop)
            while c2 != c1:
                c2 = np.random.randint(0,self.pop)

            if self.currentScores[c1] < self.currentScores[c2]:
                p1 = self.currents[c1]
            else:
                p1 = self.currents[c2]

            c1 = np.random.randint(0,self.pop)
            c2 = np.random.randint(0,self.pop)
            while c2 != c1:
                c2 = np.random.randint(0,self.pop)

            if self.currentScores[c1] < self.currentScores[c2]:
                p2 = self.currents[c1]
            else:
                p2 = self.currents[c2]

            c1 = [0 for x in range(dim)]
            c2 = [0 for x in range(dim)]
            for index,g1,g2 in zip(range(dim),p1,p2):
                if 0 == np.random.randint(0,2)%2:
                    c1[index] = g1
                    c2[index] = g2
                else:
                    c1[index] = g2
                    c2[index] = g1
            next_generation.append(c1)
            next_generation.append(c2)

        for i in range(eliteSz,len(next_generation)):
            for j in range(dim):
                if np.random.random() < 0.05:
                    next_generation[i][j] = np.random.randint(0,2)
        # random gene
        while len(next_generation) < self.pop:
            next_generation.append( [np.random.randint(0,2) for x in range(dim)] )

        # replace
        self.currents = next_generation[:self.pop]

    def calcObj(self,E):
        for i in range(self.pop):
            self.currentScores[i] = E(tuple(self.currents[i]))
            #print(self.currents[i])
            #print(self.currentScores[i])

    def calcObjParallel(self,E):
        unique = list(set([tuple(gene) for gene in self.currents]))
        args = [[u] for u in unique]

        start_time = time.time()
        with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
            results = p.starmap(E, args)
        end_time = time.time()

        new_time = (end_time-start_time)/len(args)*100
        #self.performance.append( (new_time,self.threads) )
        #self.performance.sort()

        print("%.1fseconds..."%new_time)

        #if new_time <= self.performance[0][0]:
        #    self.threads += 1
        #else:
        #    self.threads = self.performance[0][1]
        #print("setting thread size to %s"%self.threads)

        for i in range(len(unique)):
            score = E(unique[i],memo=results[i])

        for i in range(self.pop):
            self.currentScores[i] = E(tuple(self.currents[i]))
