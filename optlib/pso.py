'''
Created on 2013/04/26

@author: Minoru Kanemasa
'''

import numpy as np
import multiprocessing as mp

class PSO:
    '''
    Particle Swarm Optimization
    '''
    def __init__(self,ite,pop=20,c0=0.729,c1=1.4955,c2=1.49455,torus=False,vlim=False,best=None,threads=1):
        '''
        Default values are stable and usually fine.
        Give maximum iteration (ite) and population size (pop) if needed based on the time you have.
        Set 1 < thread for multithreading, however, creating threads have initial cost.
        It is not recommended using multithreading for light functions.
        '''
        self.ite = ite
        self.pop = pop

        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

        self.torus = torus
        self.vlim = vlim

        self.gbest = best

        self.threads = threads

    def run(self,E):
        '''
        Make sure that you rescale the objective function within 0 < x < 1,
        since the initial points are uniformly distributed within that range.
        '''
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        for k in range(self.ite-1):
            self._rewriteBests(E)
            #print("%s, %s, %s"%(k,self.gbestScore,list(self.gbest)))
            #print("%s, %s, %s"%(k,self.currents[0][0],self.vs[0][0]))
            self._moveToNext(dim, k)
            self._calcObj(E)
        self._rewriteBests(E)

        return self.gbest

    def _init(self,dim,E):
        self.pbests = [ np.random.rand(dim) for x in range(self.pop) ]
        self.currents = [ np.random.rand(dim) for x in range(self.pop) ]
        self.vs = [ np.random.rand(dim)-0.5 for x in range(self.pop) ]

        if None != self.gbest:
            self.gbestScore = E(np.array(self.gbest))
            self.gbest = np.array(self.gbest)
        else:
            self.gbest = np.random.rand(dim)
            self.gbestScore = np.inf

        self.pbestScores = [ np.inf for x in range(self.pop) ]
        self.currentScores = [ np.inf for x in range(self.pop) ]

    def _rewriteBests(self,E):
        for i in range(self.pop):
            if self.currentScores[i] <= self.pbestScores[i]:
                self.pbests[i] = self.currents[i].copy()
                self.pbestScores[i] = self.currentScores[i]
                if self.pbestScores[i] <= self.gbestScore:
                    self.gbest = self.pbests[i].copy()
                    self.gbestScore = self.pbestScores[i]
                    #print("best score = %f"%self.gbestScore)
                    #print("best = %s"%self.gbest)
                    #if hasattr(E,"cross"):
                    #    print(E.cross(self.gbest))

    def _calcObj(self,E):
        if 1==self.threads:
            for i in range(self.pop):
                self.currentScores[i] = E(self.currents[i])
        else:
            args = [[self.currents[k]] for k in range(self.pop)]
            with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
                results = p.starmap(E, args)
            self.currentScores = results

    def _moveToNext(self,dim,k):
        '''
        cg = 0.0
        for i in range(self.pop):
            cg += np.linalg.norm(self.gbest-self.currents[i],1)

        cg /= (len(self.gbest)*self.pop)

        self.c2 = 0.05/cg
        self.c1 = 0.43
        self.c0 = 0.0
        '''

        for i in range(self.pop):
            self.vs[i] = self.c0*self.vs[i]+self.c1*np.random.rand(dim)*(self.pbests[i]-self.currents[i]) + self.c2*np.random.rand(dim)*(self.gbest-self.currents[i])

            if self.vlim:
                for j in range(dim):
                    if 0.5 < self.vs[i][j]:
                        self.vs[i][j] = 0.5
                    if self.vs[i][j] < -0.5:
                        self.vs[i][j] = -0.5

        for i in range(self.pop):
            self.currents[i] += self.vs[i]

            if self.torus:
                for j in range(dim):
                    self.currents[i][j] -= int(self.currents[i][j])
                    if self.currents[i][j] < 0:
                        self.currents[i][j] += 1

    def name(self):
        return self.__class__.__name__

class LdiwPSO( PSO ):
    def __init__(self,ite,pop=20,c0start=0.9,c0end=0.4,c1=2,c2=2,torus=True,vlim=True,best=None,threads=1):
        super(LdiwPSO,self).__init__(ite,pop,c1=c1,c2=c2,torus=torus,vlim=vlim,best=best,threads=threads)
        self.c0start = c0start
        self.c0end = c0end

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        for k in range(self.ite-1):
            self._rewriteBests(E)
            self._setParam(k)
            self._moveToNext(dim, k)
            self._calcObj(E)
        self._rewriteBests(E)

        return self.gbest

    def _setParam(self,k):
        self.c0 = self.c0start-(self.c0start-self.c0end)*(k)/(self.ite-2)

    def name(self):
        return 'ldiwpso'

class AfePSO( PSO ):
    def __init__(self,ite,pop=20,c0=1.0,c1=1.0,c2=1.0,At0=0.25,AtK=0.001,c0max=1.0,c0min=0.5,deltac0=0.1,torus=True,vlim=True,best=None,threads=1):
        super(AfePSO,self).__init__(ite,pop,c0=c0,c1=c1,c2=c2,torus=torus,vlim=vlim,best=best,threads=threads)
        self.At0 = At0
        self.AtK = AtK
        self.c0max = c0max
        self.c0min = c0min
        self.deltac0 = deltac0

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        for k in range(self.ite-1):
            self._rewriteBests(E)
            self._setParam(k)
            self._moveToNext(dim, k)
            self._calcObj(E)
        self._rewriteBests(E)

        return self.gbest

    def _setParam(self,k):
        Ata = self.At0*((self.AtK/self.At0)**(k/(self.ite-2)))

        Act = 0.0
        for i in range(self.pop):
            for elem in self.vs[i]:
                Act += elem**2
        Act = np.sqrt(Act/(self.pop*len(self.vs[0])))

        if Act > Ata:
            self.c0 = max(self.c0-self.deltac0, self.c0min)
        elif Act < Ata:
            self.c0 = min(self.c0+self.deltac0, self.c0max)
        else:
            pass

class NdtPSO( PSO ):
    def __init__(self,ite,pop=20,d0=10.0,d1=0.5,d2=0.01,c1=2,c2=2,torus=True,vlim=True,best=None,threads=1):
        super(NdtPSO,self).__init__(ite,pop,c1=c1,c2=c2,torus=torus,vlim=vlim,best=best,threads=threads)
        self.c0 = [self.c0]*pop
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2

    def _moveToNext(self,dim,k):
        d0 = self.d0*(1-k/(self.ite-1))
        for i in range(self.pop):
            z = np.linalg.norm(self.gbest-self.currents[i])
            self.c0[i] = 1-self.d1+self.d1*d0*np.exp(-z/(self.d2*d0))

        for i in range(self.pop):
            self.vs[i] = self.c0[i]*self.vs[i]+self.c1*np.random.rand(dim)*(self.pbests[i]-self.currents[i]) + self.c2*np.random.rand(dim)*(self.gbest-self.currents[i])

            if self.vlim:
                for j in range(dim):
                    if 0.5 < self.vs[i][j]:
                        self.vs[i][j] = 0.5
                    if self.vs[i][j] < -0.5:
                        self.vs[i][j] = -0.5

        for i in range(self.pop):
            self.currents[i] += self.vs[i]

            if self.torus:
                for j in range(dim):
                    self.currents[i][j] -= int(self.currents[i][j])
                    if self.currents[i][j] < 0:
                        self.currents[i][j] += 1

class LdvmPSO( PSO ):
    def __init__(self,ite,pop=20,vm0=0.25,c0=1,c1=2,c2=2,torus=True,best=None,threads=1):
        super(LdvmPSO,self).__init__(ite,pop,c0=c0,c1=c1,c2=c2,torus=torus,vlim=True,best=best,threads=threads)
        self.vm0 = vm0

    def _moveToNext(self,dim,k):
        for i in range(self.pop):
            self.vs[i] = self.c0*self.vs[i]+self.c1*np.random.rand(dim)*(self.pbests[i]-self.currents[i]) + self.c2*np.random.rand(dim)*(self.gbest-self.currents[i])

            if self.vlim:
                vmax = self.vm0*(1-k/(self.ite-2))
                for j in range(dim):
                    if vmax < self.vs[i][j]:
                        self.vs[i][j] = vmax
                    if self.vs[i][j] < -vmax:
                        self.vs[i][j] = -vmax

        for i in range(self.pop):
            self.currents[i] += self.vs[i]

            if self.torus:
                for j in range(dim):
                    self.currents[i][j] -= int(self.currents[i][j])
                    if self.currents[i][j] < 0:
                        self.currents[i][j] += 1

class Kpso( PSO ):
    def __init__(self,ite,pop=20,c0=0.8321,c1=2,c2=2,torus=True,vlim=False,linked=True,best=None,threads=1):
        super(Kpso,self).__init__(ite,pop,c0=c0,c1=c1,c2=c2,torus=torus,vlim=vlim,best=best,threads=threads)
        self.linked = linked
    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        for k in range(self.ite-1):
            self._rewriteBests(E)
            if self.ite*0.9 < k:
                self.c0 = 0.7
                self.c1 = 1.4
                self.c2 = 1.4
            self._moveToNext(dim, k)
            self._calcObj(E)
        self._rewriteBests(E)

        return self.gbest

    def _moveToNext(self,dim,k):
        backup = []
        for i in range(self.pop):
            backup.append( self.currents[i].copy() )

        for i in range(self.pop):
            if self.linked:
                rnd1 = np.random.rand()
                rnd2 = np.random.rand()
            else:
                rnd1 = np.random.rand(dim)
                rnd2 = np.random.rand(dim)

            self.vs[i] = self.c0*self.vs[i]+self.c1*rnd1*(self.pbests[i]-self.currents[i]) + self.c2*rnd2*(self.gbest-self.currents[i]) + 2*(10**(-11))*(np.random.rand(dim)-0.5)

            if self.vlim:
                for j in range(dim):
                    if 0.5 < self.vs[i][j]:
                        self.vs[i][j] = 0.5
                    if self.vs[i][j] < -0.5:
                        self.vs[i][j] = -0.5

        for i in range(self.pop):
            self.currents[i] += self.vs[i]

            if self.torus:
                for j in range(dim):
                    self.currents[i][j] -= int(self.currents[i][j])
                    if self.currents[i][j] < 0:
                        self.currents[i][j] += 1

        for i in range(self.pop):
            self.vs[i] = self.currents[i] - backup[i]

class GeoPSO( PSO ):
    def __init__(self,ite,pop=20,torus=False,vlim=False,best=None,threads=1):
        super(GeoPSO,self).__init__(ite,pop,torus=torus,vlim=vlim,best=best,threads=threads)

    def _moveToNext(self,dim,k):

        cg = 0.0
        v = 0.0
        for i in range(self.pop):
            cg += np.linalg.norm(self.gbest-self.currents[i],2)
            v += np.linalg.norm(self.vs[i],2)

        cg /= (np.sqrt(dim)*self.pop)
        v /= (np.sqrt(dim)*self.pop)

        sin = np.sin
        cos = np.cos
        exp = np.exp
        def log(x):
            if 0==x:
                return -np.inf
            else:
                return np.log(np.fabs(x))
        sqrt = np.sqrt
        def sig(x):
            return 1.0/(1.0+np.exp(-x))

        self.c0 = cg
        self.c1 = v*v
        self.c2 = sqrt((sig((-((-0.0342647 - ((0.952148 + -((-(sqrt((sig((v - cg)) + (v + (-((-0.0342647 - -0.0657784)) - sig(-((0.907556 / cg)))))))) - +(((v - ((((sig(v) + (exp(((0.278754 / +(0.466325)) / ((-0.32228 / 0.922531) / sig((-((-0.0342647 - v)) - 0.907556))))) * (cg - -0.906061))) + (-0.539545 * (cg + -((-0.539545 * (cg + -((cg - (-0.393534 / cg))))))))) / cg) - (0.907556 / +(-((cg / +((-0.667049 + v)))))))) * exp(0.0)))))) + (exp((((-0.211836 - -0.906061) / +(0.466325)) / ((-0.32228 / 0.922531) / (0.907556 + -0.388369)))) * (-0.211836 - -0.906061))))) - sig((cg - 0.807164)))) + -((-0.0342647 - (0.429681 / ((cg - 0.0) + cg))))))

        #print(self.gbestScore,self.c0,self.c1,self.c2)

        '''
        cg = 0.0
        for i in range(self.pop):
            cg += np.linalg.norm(self.gbest-self.currents[i],1)

        cg /= (len(self.gbest)*self.pop)

        self.c2 = 0.05/cg
        self.c1 = 0.43
        self.c0 = 0.0

        #print(self.c0,self.c1,self.c2)

        '''

        for i in range(self.pop):
            self.vs[i] = self.c0*self.vs[i]+self.c1*np.random.rand(dim)*(self.pbests[i]-self.currents[i]) + self.c2*np.random.rand(dim)*(self.gbest-self.currents[i])

            if self.vlim:
                for j in range(dim):
                    if 0.5 < self.vs[i][j]:
                        self.vs[i][j] = 0.5
                    if self.vs[i][j] < -0.5:
                        self.vs[i][j] = -0.5

        for i in range(self.pop):
            self.currents[i] += self.vs[i]

            if self.torus:
                for j in range(dim):
                    self.currents[i][j] -= int(self.currents[i][j])
                    if self.currents[i][j] < 0:
                        self.currents[i][j] += 1

class GpPSO( PSO ):
    def __init__(self,ite,pop=20,torus=False,vlim=False,best=None,threads=1):
        super(GpPSO,self).__init__(ite,pop,torus=torus,vlim=vlim,best=best,threads=threads)

    def _init(self,dim,E):
        PSO._init(self,dim,E)
        self._hist = []

    def _rewriteBests(self,E):
        cnt = 0
        for i in range(self.pop):
            if self.currentScores[i] <= self.pbestScores[i]:
                cnt += 1
                self.pbests[i] = self.currents[i].copy()
                self.pbestScores[i] = self.currentScores[i]
                if self.pbestScores[i] <= self.gbestScore:
                    self.gbest = self.pbests[i].copy()
                    self.gbestScore = self.pbestScores[i]
        self._hist.append(cnt)
        if E.dim()*10 < len(self._hist):
            self._hist.pop(0)

    def _moveToNext(self,dim,k):

        cg = 0.0
        v = 0.0
        for i in range(self.pop):
            cg += np.linalg.norm(self.gbest-self.currents[i])
            v += np.linalg.norm(self.vs[i])

        cg /= (np.sqrt(dim)*self.pop)
        v /= (np.sqrt(dim)*self.pop)

        sin = np.sin
        cos = np.cos
        exp = np.exp
        def log(x):
            if 0==x:
                return -np.inf
            else:
                return np.log(np.fabs(x))
        sqrt = np.sqrt
        def sig(x):
            return 1.0/(1.0+np.exp(-x))

        if 0 == len(self._hist):
            p = 1.0
        else:
            p = np.mean(self._hist)/(self.pop)

        self.c0 = sig(((p * sig(sig(p))) + sig(sig(sig(p)))))
        self.c1 = exp(sig((((p / sig(0.536447)) * exp((0.536447 / sig(0.536447)))) + sig(sig(0.536447)))))
        self.c2 = (sqrt((+((p + (+((p + ((sqrt(p) + (p + p)) + p))) + 0.671607))) + (0.768319 + (sqrt(p) + -0.597805)))) + p)


        #print(self.gbestScore,p,self.c0,self.c1,self.c2)

        for i in range(self.pop):
            self.vs[i] = self.c0*self.vs[i]+self.c1*np.random.rand(dim)*(self.pbests[i]-self.currents[i]) + self.c2*np.random.rand(dim)*(self.gbest-self.currents[i])

            if self.vlim:
                for j in range(dim):
                    if 0.5 < self.vs[i][j]:
                        self.vs[i][j] = 0.5
                    if self.vs[i][j] < -0.5:
                        self.vs[i][j] = -0.5

        for i in range(self.pop):
            self.currents[i] += self.vs[i]

            if self.torus:
                for j in range(dim):
                    self.currents[i][j] -= int(self.currents[i][j])
                    if self.currents[i][j] < 0:
                        self.currents[i][j] += 1