'''
Created on 2015/03/13

@author: admin
'''
import numpy as np

class Firefly:
    '''
    Particle Swarm Optimization
    '''
    def __init__(self,ite,pop=20,alpha=0.25,beta=0.5, gamma=1.0, variant='sfa',torus=True,best=None,ftarget=-np.inf,threads=1):
        '''
        Default values are stable and usually fine.
        Give maximum iteration (ite) and population size (pop) if needed based on the time you have.
        Set 1 < thread for multithreading, however, creating threads have initial cost.
        It is not recommended using multithreading for light functions.
        Threading is not supported for this algorithm at the moment.
        '''
        self.ite = ite
        self.pop = pop

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

        self.variant = variant

        self.torus = torus

        self._best = best

        self._ftarget = ftarget

        self.threads = threads

    def _init(self,dim,E):
        self.pbests = [ np.random.rand(dim) for x in range(self.pop) ]
        self.currents = [ np.random.rand(dim) for x in range(self.pop) ]

        if None == self._best:
            self._best = np.random.rand(dim)
            self._bestScore = np.inf
        else:
            self._bestScore = E(np.array(self._best))
            self._best = np.array(self._best)

        self.pbestScores = [ np.inf for x in range(self.pop) ]
        self.currentScores = [ np.inf for x in range(self.pop) ]

        self._fcall = 0
        self.changed = [ True for x in range(self.pop) ]

    def _calcObj(self,E):
        if 1==self.threads:
            for i in range(self.pop):
                if self.changed[i]:
                    self.changed[i] = False
                    if self.pop*self.ite <= self._fcall:
                        break
                    self.currentScores[i] = E(self.currents[i])
                    self._fcall += 1
        else:
            args = [[self.currents[k]] for k in range(self.pop)]
            with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
                results = p.starmap(E, args)
            self.currentScores = results
            self._fcall += self.pop

    def _rewriteBests(self,E):
        for i in range(self.pop):
            if self.currentScores[i] <= self.pbestScores[i]:
                self.pbests[i] = self.currents[i].copy()
                self.pbestScores[i] = self.currentScores[i]
                if self.pbestScores[i] <= self._bestScore:
                    self._best = self.pbests[i].copy()
                    self._bestScore = self.pbestScores[i]

    def run(self,E):
        '''
        Make sure that you rescale the objective function within 0 < x < 1,
        since the initial points are uniformly distributed within that range.
        '''
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        self._rewriteBests(E)
        cnt = 0
        while self._fcall < self.pop*self.ite:
            a = cnt%self.pop
            b = int(cnt/self.pop)%self.pop

            a, b = b, a

            t = self._fcall/float(self.ite*self.pop)
            beta = 0.267#self.beta
            alpha = self.alpha*np.exp(-5.0*t)

            if self.currentScores[b] <= self.currentScores[a]:
                #beta = self.beta
                #alpha = self.alpha

                #alpha = 0.5-0.49*t
                #beta = self.beta
                #lambd = self.alpha

                r = np.linalg.norm(self.currents[b]-self.currents[a])
                gamma = self.gamma/np.sqrt(dim)
                inside = -(gamma*(r**2.0)/float(dim))*10

                rg = np.linalg.norm(self._best-self.currents[a])
                inside2 = -(gamma*(rg**2.0)/float(dim))*10

                #print(inside,inside2,np.exp(inside),np.exp(inside2))

                if 'sfa' == self.variant:
                    #alpha = 0.1
                    self.currents[a] = self.currents[a]+beta*np.exp(inside)*(self.currents[b]-self.currents[a]) + alpha*(np.random.randn(dim))
                elif 'mfa1' == self.variant:
                    self.currents[a] = self.currents[a]+beta*np.exp(inside)*(self.currents[b]-self.currents[a]) +beta*np.exp(inside2)*(self._best-self.currents[a]) + alpha*(np.random.rand(dim)-0.5)
                elif 'nmfa' == self.variant:
                    lambd = 2.5
                    self.currents[a] = self.currents[a]+beta*np.exp(inside)*(self.currents[b]-self.currents[a]) +beta*np.exp(inside2)*(self._best-self.currents[a]) + lambd*(np.random.rand(dim)-0.5)*(self.currents[a]-self._best)+alpha*(np.random.rand(dim)-0.5)

                if self.torus:
                    i = a
                    for j in range(dim):
                        self.currents[i][j] -= int(self.currents[i][j])
                        if self.currents[i][j] < 0:
                            self.currents[i][j] += 1

                self.currentScores[a] = E(self.currents[a])
                self._fcall += 1

                if self.currentScores[a] <= self.pbestScores[a]:
                    self.pbests[a] = self.currents[a].copy()
                    self.pbestScores[a] = self.currentScores[a]
                    if self.pbestScores[a] <= self._bestScore:
                        self._best = self.pbests[a].copy()
                        self._bestScore = self.pbestScores[a]
                        if self._bestScore <= self._ftarget:
                            break
            cnt += 1

        return self._best

    def name(self):
        return self.__class__.__name__

class MathFly(Firefly):
    '''
    Particle Swarm Optimization
    '''
    def __init__(self,ite,pop=20,alpha=7.72,beta=1.0,torus=True,best=None,ftarget=-np.inf,threads=1):
        '''
        Default values are stable and usually fine.
        Give maximum iteration (ite) and population size (pop) if needed based on the time you have.
        Set 1 < thread for multithreading, however, creating threads have initial cost.
        It is not recommended using multithreading for light functions.
        '''
        self.ite = ite
        self.pop = pop

        self.beta = beta
        self.alpha = alpha

        self.torus = torus

        self._best = best

        self._ftarget = ftarget

        self.threads = threads

    def run(self,E):
        '''
        Make sure that you rescale the objective function within 0 < x < 1,
        since the initial points are uniformly distributed within that range.
        '''
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        while self._fcall < self.pop*self.ite:
            self._rewriteBests(E)
            if self._bestScore < self._ftarget:
                break
            #print("%s, %s, %s"%(k,self.gbestScore,list(self.gbest)))
            #print("%s, %s, %s"%(k,self.currents[0][0],self.vs[0][0]))
            self._moveToNext(dim)
            self._calcObj(E)
        self._rewriteBests(E)

        return self._best

    def _init(self,dim,E):
        self.pbests = [ np.random.rand(dim) for x in range(self.pop) ]
        self.currents = [ np.random.rand(dim) for x in range(self.pop) ]

        if None == self._best:
            self._best = np.random.rand(dim)
            self._bestScore = np.inf
        else:
            self._bestScore = E(np.array(self._best))
            self._best = np.array(self._best)

        self.pbestScores = [ np.inf for x in range(self.pop) ]
        self.currentScores = [ np.inf for x in range(self.pop) ]

        self._fcall = 0
        self.changed = [ True for x in range(self.pop) ]

    def _rewriteBests(self,E):
        for i in range(self.pop):
            if self.currentScores[i] <= self.pbestScores[i]:
                self.pbests[i] = self.currents[i].copy()
                self.pbestScores[i] = self.currentScores[i]
                if self.pbestScores[i] <= self._bestScore:
                    self._best = self.pbests[i].copy()
                    self._bestScore = self.pbestScores[i]
                    #print("best score = %f"%self.gbestScore)
                    #print("best = %s"%self.gbest)
                    #if hasattr(E,"cross"):
                    #    print(E.cross(self.gbest))

    def _calcObj(self,E):
        if 1==self.threads:
            for i in range(self.pop):
                if self.changed[i]:
                    self.changed[i] = False
                    if self.pop*self.ite <= self._fcall:
                        break
                    self.currentScores[i] = E(self.currents[i])
                    self._fcall += 1
        else:
            args = [[self.currents[k]] for k in range(self.pop)]
            with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
                results = p.starmap(E, args)
            self.currentScores = results
            self._fcall += self.pop

    def _moveToNext(self,dim):
        t = self._fcall/float(self.ite*self.pop)

        for i in range(self.pop):
            q1 = np.random.randint(0,self.pop)
            while self.pbestScores[i] < self.pbestScores[q1]:
                q1 = np.random.randint(0,self.pop)

            if self.pbestScores[q1] <= self.pbestScores[i]:
                self.changed[i] = True

                self.currents[i] = self.currents[i]+self.beta*(self.pbests[q1]-self.currents[i]) + 0.254*np.exp(-self.alpha*t)*np.random.randn(dim)

                if self.torus:
                    for j in range(dim):
                        self.currents[i][j] -= int(self.currents[i][j])
                        if self.currents[i][j] < 0:
                            self.currents[i][j] += 1

    def name(self):
        return self.__class__.__name__
