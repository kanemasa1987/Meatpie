'''
Created on 2015/02/25

@author: Minoru Kanemasa
'''

import numpy as np
import multiprocessing as mp

class De:
    def __init__(self,ite,pop,f=0.5,cr=0.9,torus=False,best=None,ftarget=-np.inf,threads=1):
        self.ite = ite
        self.pop = pop

        self.f = f
        self.cr = cr

        self.torus = torus

        self._best = best

        self._ftarget=ftarget

        self.threads = threads

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        self._moveToNext(E)
        for k in range(self.ite-1):
            self._rewriteBest(E)
            if self._bestScore < self._ftarget:
                break
            self._crossOver(dim, k)
            self._calcObj(E)
            self._moveToNext(E)
        self._rewriteBest(E)

        return self._best

    def _init(self,dim,E):
        self.currents = [ np.random.rand(dim) for x in range(self.pop) ]
        self.nexts = [ np.random.rand(dim) for x in range(self.pop) ]
        self.currentScores = [ np.inf for x in range(self.pop) ]
        self.nextScores = [ np.inf for x in range(self.pop) ]

        if None == self._best:
            self._best = np.random.rand(dim)
            self._bestScore = np.inf
        else:
            self._bestScore = E(np.array(self._best))
            self._best = np.array(self._best)

    def _calcObj(self,E):
        if 1==self.threads:
            for i in range(self.pop):
                self.nextScores[i] = E(self.nexts[i])
        else:
            args = [[self.nexts[k]] for k in range(self.pop)]
            with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
                results = p.starmap(E, args)
            self.nextScores = results

        for i in range(self.pop):
            if self.nextScores[i] < self.currentScores[i]:
                self.currentScores[i] = self.nextScores[i]
            else:
                self.nexts[i], self.currents[i] = self.currents[i], self.nexts[i]

        #good vectors are in nexts at this point

    def _rewriteBest(self,E):
        for i in range(self.pop):
            if self.currentScores[i] <= self._bestScore:
                self._best = self.currents[i].copy()
                self._bestScore = self.currentScores[i]

    def _crossOver(self,dim,k):
        for i in range(self.pop):
            q1 = np.random.randint(0,self.pop)
            while q1 == i:
                q1 = np.random.randint(0,self.pop)

            q2 = np.random.randint(0,self.pop)
            while q2 == i or q2 == q1:
                q2 = np.random.randint(0,self.pop)

            q3 = np.random.randint(0,self.pop)
            while q3 == i or q3 == q1 or q3 == q2:
                q3 = np.random.randint(0,self.pop)

            rd = np.random.randint(0,dim)
            rnds = np.random.rand(dim)
            for j, rnd in enumerate(rnds):
                if rnd < self.cr or j==rd:
                    self.nexts[i][j] = self.currents[q1][j]+self.f*(self.currents[q2][j]-self.currents[q3][j])
                else:
                    self.nexts[i][j] = self.currents[i][j]

        if self.torus:
            for i in range(self.pop):
                for j in range(dim):
                    self.nexts[i][j] -= int(self.nexts[i][j])
                    if self.nexts[i][j] < 0:
                        self.nexts[i][j] += 1

    def _moveToNext(self,E):
        self.nexts, self.currents = self.currents, self.nexts

        #good vectors are in currents at this point

class Jde(De):
    def __init__(self,ite,pop,fl=0.1,fu=0.9,t1=0.1,t2=0.1,torus=False,best=None,ftarget=-np.inf,threads=1):
        De.__init__(self, ite=ite, pop=pop, torus=torus, best=best, ftarget=ftarget, threads=threads)
        self.fl = fl
        self.fu = fu

        self.t1 = t1
        self.t2 = t2

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        self._moveToNext(E)
        for k in range(self.ite-1):
            self._rewriteBest(E)
            if self._bestScore < self._ftarget:
                break
            self._setParam()
            self._crossOver(dim, k)
            self._calcObj(E)
            self._moveToNext(E)
        self._rewriteBest(E)

        return self._best

    def _init(self,dim,E):
        De._init(self, dim, E)
        self.fs = [self.fl+np.random.rand()*self.fu for x in range(self.pop)]
        self.crs = [np.random.rand() for x in range(self.pop)]

    def _setParam(self):
        for i in range(self.pop):
            if np.random.rand() < self.t1:
                self.fs[i] = self.fl+np.random.rand()*self.fu
            if np.random.rand() < self.t2:
                self.crs[i] = np.random.rand()

    def _crossOver(self,dim,k):
        for i in range(self.pop):
            q1 = np.random.randint(0,self.pop)
            while q1 == i:
                q1 = np.random.randint(0,self.pop)

            q2 = np.random.randint(0,self.pop)
            while q2 == i or q2 == q1:
                q2 = np.random.randint(0,self.pop)

            q3 = np.random.randint(0,self.pop)
            while q3 == i or q3 == q1 or q3 == q2:
                q3 = np.random.randint(0,self.pop)

            rd = np.random.randint(0,self.pop)
            rnds = np.random.rand(dim)
            for j, rnd in enumerate(rnds):
                if rnd < self.crs[i] or j==rd:
                    self.nexts[i][j] = self.currents[q1][j]+self.fs[i]*(self.currents[q2][j]-self.currents[q3][j])
                else:
                    self.nexts[i][j] = self.currents[i][j]

        if self.torus:
            for i in range(self.pop):
                for j in range(dim):
                    self.nexts[i][j] -= int(self.nexts[i][j])
                    if self.nexts[i][j] < 0:
                        self.nexts[i][j] += 1

class Sde(Jde):
    def __init__(self,ite,pop,mu=0.8,deltal=0.1,deltau=0.9,t1=0.1,t2=0.1,torus=False,best=None,ftarget=-np.inf,threads=1):
        Jde.__init__(self,ite=ite,pop=pop,fl=deltal,fu=deltau,t1=0.1,t2=0.1,torus=torus,best=best,ftarget=ftarget,threads=threads)
        self.mu = mu

    def _init(self,dim,E):
        Jde._init(self, dim, E)
        self.deltas = self.fl+self.fu*np.random.rand(self.pop)

    def _setParam(self):
        for i in range(self.pop):
            if np.random.rand() < self.t1:
                self.deltas[i] = self.fl+self.fu*np.random.rand()
            self.fs[i] = self._cauthy(self.mu,self.deltas[i])

            if np.random.rand() < self.t2:
                self.crs[i] = np.random.rand()

    def _cauthy(self,mu,delta):
        return delta*np.random.standard_cauchy()+mu

class AdaptDe(Jde):
    def __init__(self,ite,pop,t1=0.1,t2=0.1,torus=False,best=None,ftarget=-np.inf,threads=1):
        Jde.__init__(self,ite=ite,pop=pop,t1=0.1,t2=0.1,torus=torus,best=best,ftarget=ftarget,threads=threads)

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        self._moveToNext(E)
        for k in range(self.ite-1):
            self._rewriteBest(E)
            if self._bestScore < self._ftarget:
                break
            self._crossOver(dim, k)
            self._calcObj(E)
            self._moveToNext(E)
        self._rewriteBest(E)

        return self._best

    def _init(self,dim,E):
        Jde._init(self, dim, E)
        self.deltas = self.fl+self.fu*np.random.rand(self.pop)

    def _calcObj(self,E):
        if 1==self.threads:
            for i in range(self.pop):
                self.nextScores[i] = E(self.nexts[i])
        else:
            args = [[self.nexts[k]] for k in range(self.pop)]
            with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
                results = p.starmap(E, args)
            self.nextScores = results

        self._setParam()

        for i in range(self.pop):
            if self.nextScores[i] < self.currentScores[i]:
                self.currentScores[i] = self.nextScores[i]
            else:
                self.nexts[i], self.currents[i] = self.currents[i], self.nexts[i]

        #good vectors are in nexts at this point

    def _setParam(self):
        for i in range(self.pop):
            if self.nextScores[i] > self.currentScores[i]:
                if np.random.rand() < self.t1:
                    self.fs[i] = np.random.rand()

                if np.random.rand() < self.t2:
                    self.crs[i] = np.random.rand()

class Jade(De):
    def __init__(self,ite,pop,p=0.1,c=0.05,torus=False,best=None,ftarget=-np.inf,threads=1):
        De.__init__(self, ite=ite, pop=pop, torus=torus, best=best, ftarget=ftarget, threads=threads)

        self.p = p
        self.c = c

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        self._moveToNext(E)
        for k in range(self.ite-1):
            self._rewriteBest(E)
            if self._bestScore < self._ftarget:
                break
            self._setParam()
            self._crossOver(dim, k)
            self._calcObj(E)
            self._moveToNext(E)
            self._rewriteParam()
        self._rewriteBest(E)

        return self._best

    def _init(self,dim,E):
        De._init(self, dim, E)
        self._mucr = 0.5
        self._muf = 0.5
        self._archive = []
        self._sf = []
        self._scr = []

    def _setParam(self):
        self._crs = [min(1.0,max(0.0,np.random.normal(self._mucr,0.1))) for x in range(self.pop)]
        self._fs = [self._cauthy(self._muf, 0.1) for x in range(self.pop)]

    def _rewriteParam(self):
        self._mucr = (1-self.c)*self._mucr+self.c*np.mean( self._scr )
        self._muf = (1-self.c)*self._muf+self.c*(np.linalg.norm(self._sf)**2)/np.sum( self._sf )

    def _crossOver(self,dim,k):
        scoreVecs = []
        for i in range(self.pop):
            scoreVecs.append( (self.currentScores[i],self.currents[i]) )

        def scoreKey(tup):
            return tup[0]

        scoreVecs.sort(key=scoreKey)

        scoreVecs[:max(1,int(self.pop*self.p))]

        for i in range(self.pop):
            r1 = np.random.randint(0,self.pop)
            while r1 == i:
                r1 = np.random.randint(0,self.pop)

            xr1 = self.currents[r1]

            be = np.random.randint(0,len(scoreVecs))
            x_bestp = scoreVecs[be][1]

            pora = self.currents+self._archive
            r2 = np.random.randint(0,len(pora))
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0,len(pora))

            xr2 = pora[r2]

            rd = np.random.randint(0,self.pop)
            rnds = np.random.rand(dim)
            for j, rnd in enumerate(rnds):
                if rnd < self._crs[i] or j==rd:
                    self.nexts[i][j] = self.currents[i][j]+self._fs[i]*(x_bestp[j]-self.currents[i][j])+self._fs[i]*(xr1[j]-xr2[j])
                else:
                    self.nexts[i][j] = self.currents[i][j]

        if self.torus:
            for i in range(self.pop):
                for j in range(dim):
                    self.nexts[i][j] -= int(self.nexts[i][j])
                    if self.nexts[i][j] < 0:
                        self.nexts[i][j] += 1

    def _calcObj(self,E):
        if 1==self.threads:
            for i in range(self.pop):
                self.nextScores[i] = E(self.nexts[i])
        else:
            args = [[self.nexts[k]] for k in range(self.pop)]
            with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
                results = p.starmap(E, args)
            self.nextScores = results

        self._setParam()

        for i in range(self.pop):
            if self.nextScores[i] < self.currentScores[i]:
                self.currentScores[i] = self.nextScores[i]
                self._sf.append( self._fs[i] )
                self._scr.append( self._crs[i] )
                self._archive.append( self.currents[i].copy() )
                while self.pop < len(self._archive):
                    self._archive.pop( np.random.randint(0,len(self._archive)) )
            else:
                self.nexts[i], self.currents[i] = self.currents[i], self.nexts[i]

        #good vectors are in nexts at this point

    def _cauthy(self,mu,delta):
        ret = delta*np.random.standard_cauchy()+mu
        if 1 < ret:
            ret = 1.0
        while ret <= 0:
            ret = delta*np.random.standard_cauchy()+mu
            if 1 < ret:
                ret = 1.0
        return ret

class Apde(De):
    def __init__(self,ite,pop,f0=0.9,cr0=0.1,deltaF=0.05,deltaCr=0.1,fmin=0.1,fmax=2.0,crmin=0.1,crmax=0.9,itgmax=50.0,torus=False,best=None,ftarget=-np.inf,threads=1):
        De.__init__(self, ite=ite, pop=pop,f=f0,cr=cr0,torus=torus, best=best, ftarget=ftarget,threads=threads)

        self.deltaF = deltaF
        self.deltaCr = deltaCr

        self.fmin = fmin
        self.fmax = fmax

        self.crmin = crmin
        self.crmax = crmax

        self.f0 = f0
        self.cr0 = cr0

        self.itgmax = itgmax


    def _init(self,dim,E):
        De._init(self, dim, E)
        self.f = self.f0
        self._cmax = 0.0
        self._C = 0.0
        self._crcmax = self.cr0
        self.cr = self.cr0
        self._P = 1

    def _paramInit(self):
        Lg = self._calcLg()

        self._Itarget0 = Lg*self.f
        self._ItargetGmax = self._Itarget0/self.itgmax

        self._favg = np.mean( self.currentScores )

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        self._moveToNext(E)
        self._rewriteBest(E)
        self._paramInit()

        for k in range(self.ite-1):
            self._crossOver(dim, k)
            self._calcObj(E)
            self._moveToNext(E)
            self._rewriteBest(E)
            if self._bestScore < self._ftarget:
                break
            self._setParam(k)

        return self._best

    def _calcLg(self):
        except_best = []
        _bestFound = False
        for vec in self.currents:
            if _bestFound or not (vec == self._best).all():
                except_best.append(vec)
            else:
                _bestFound = True

        Lg = 0.0
        for i in range(self.pop-2):
            for j in range(i+1,self.pop-1):
                if i != j:
                    Lg += np.linalg.norm(except_best[i]-except_best[j])

        tmp = 0.0
        for q in range(1,self.pop-1):
            tmp += q

        Lg = Lg/tmp

        return Lg

    def _setParam(self,k):
        newAvg = np.mean( self.currentScores )

        self._C = (self._favg - newAvg)/np.fabs(self._favg)

        self._favg = newAvg

        if self._C > self._cmax:
            self._cmax = self._C
            self._crcmax = self.cr

        Lg = self._calcLg()
        Idi = self.f*Lg

        Itarget = self._Itarget0*((self._ItargetGmax/self._Itarget0)**(k/(self.ite-3)))

        following = False
        if Itarget > Idi:
            self.f = min(self.f+self.deltaF,self.fmax)
            newIdi = self.f*Lg
            if Itarget <= newIdi:
                following = True
                self._P = 1
            else:
                self._P += 1
        else:
            self.f = max(self.f-self.deltaF,self.fmin)
            newIdi = self.f*Lg
            if Itarget >= newIdi:
                following = True
                self._P = 1
            else:
                self._P += 1

        L_best = 0.0
        for vec in self.currents:
            L_best += np.linalg.norm(self._best-vec)

        L_best /= (self.pop-1)

        Fs = L_best/Lg

        if Itarget > Idi:
            if self.f <= Fs:
                kk = -1
            else:
                kk = 0
        else:
            if self.f <= Fs:
                kk = 1
            else:
                kk = -1

        s = np.random.randint(-1,2) + kk*(self._P-1)

        if self._P <= 2:
            if Itarget > Idi:
                self.cr = min(self._crcmax+s*self.deltaCr,self.crmax)
            else:
                self.cr = max(self._crcmax+s*self.deltaCr,self.crmin)
        else:
            if Itarget > Idi:
                self.cr = min(self.cr+kk*self.deltaCr,self.crmax)
            else:
                self.cr = max(self.cr+kk*self.deltaCr,self.crmin)

    def _crossOver(self,dim,k):
        for i in range(self.pop):
            q1 = np.random.randint(0,self.pop)
            while q1 == i:
                q1 = np.random.randint(0,self.pop)

            q2 = np.random.randint(0,self.pop)
            while q2 == i or q2 == q1:
                q2 = np.random.randint(0,self.pop)

            q3 = np.random.randint(0,self.pop)
            while q3 == i or q3 == q1 or q3 == q2:
                q3 = np.random.randint(0,self.pop)

            rd = np.random.randint(0,self.pop)
            rnds = np.random.rand(dim)
            for j, rnd in enumerate(rnds):
                if rnd < self.cr or j==rd:
                    self.nexts[i][j] = self._best[j]+self.f*(self.currents[q2][j]-self.currents[q3][j])
                else:
                    self.nexts[i][j] = self.currents[i][j]

        if self.torus:
            for i in range(self.pop):
                for j in range(dim):
                    self.nexts[i][j] -= int(self.nexts[i][j])
                    if self.nexts[i][j] < 0:
                        self.nexts[i][j] += 1

class Ds(De):
    def __init__(self,ite,pop,torus=False,best=None,ftarget=-np.inf,threads=1):
        De.__init__(self, ite=ite, pop=pop, torus=torus, best=best, ftarget=ftarget, threads=threads)
        pass

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        self._calcObj(E)
        self._moveToNext(E)
        self._rewriteBest(E)
        if self._bestScore < self._ftarget:
            return self._best

        for k in range(self.ite-1):
            self._crossOver(dim, k)
            self._calcObj(E)
            self._setSigma()
            self._moveToNext(E)
            self._rewriteBest(E)
            if self._bestScore < self._ftarget:
                break

        return self._best

    def _init(self,dim,E):
        De._init(self, dim, E)
        self._hist = []
        self._sigma = 1.0

    def _setSigma(self):
        p = 1.0
        if 0 != len(self._hist):
            p = np.mean(self._hist)/float(self.pop)
        else:
            return 1.0

        s = 1.0/((1.0-np.random.rand())**(1.0/8.0))
        if np.random.rand() < 0.5:
            s = 1.0/s

        self._sigma = (2.0/3.0)*np.exp(4.0/3.0*p)*s

    def _crossOver(self,dim,k):
        for i in range(self.pop):
            q1 = np.random.randint(0,self.pop)
            while q1 == i:
                q1 = np.random.randint(0,self.pop)

            q2 = np.random.randint(0,self.pop)
            while q2 == i or q2 == q1:
                q2 = np.random.randint(0,self.pop)

            q3 = np.random.randint(0,self.pop)
            while q3 == i or q3 == q1 or q3 == q2:
                q3 = np.random.randint(0,self.pop)

            sphr = np.random.randn(dim)
            sphr = sphr/np.linalg.norm(sphr)

            x1 = self.currents[q1]+self.f*(self.currents[q2]-self.currents[q3])

            self.nexts[i] = (x1+self.currents[i])/2.0+(np.linalg.norm(x1-self.currents[i])/2.0*self._sigma)*sphr

        if self.torus:
            for i in range(self.pop):
                for j in range(dim):
                    self.nexts[i][j] -= int(self.nexts[i][j])
                    if self.nexts[i][j] < 0:
                        self.nexts[i][j] += 1

    def _calcObj(self,E):
        if 1==self.threads:
            for i in range(self.pop):
                self.nextScores[i] = E(self.nexts[i])
        else:
            args = [[self.nexts[k]] for k in range(self.pop)]
            with mp.Pool(self.threads) as p:#,maxtasksperchild=1)
                results = p.starmap(E, args)
            self.nextScores = results

        betterCnt = 0
        for i in range(self.pop):
            if self.nextScores[i] < self.currentScores[i]:
                self.currentScores[i] = self.nextScores[i]
                betterCnt += 1
            else:
                self.nexts[i], self.currents[i] = self.currents[i], self.nexts[i]

        self._hist.append(betterCnt)
        if 10*E.dim() < len(self._hist):
            self._hist.pop(0)
        #good vectors are in nexts at this point