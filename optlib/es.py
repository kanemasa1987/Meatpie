'''
Created on 2015/05/24

@author: admin
'''
import numpy as np
import scipy
import scipy.linalg

class ES:
    def __init__(self,ite,pop,c=0.85,sigma=1.0,torus=False,best=None,ftarget=-np.inf,threads=1):
        self.ite = ite
        self.pop = pop

        self.c = c
        self.firstSigma = sigma

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
            self._setSigma()
            self._oneFifth(dim, k)
            self._calcObj(E)
            self._moveToNext(E)
        self._rewriteBest(E)

        return self._best

    def _init(self,dim,E):
        self.currents = [ np.random.rand(dim) for x in range(self.pop) ]
        self.nexts = [ np.random.rand(dim) for x in range(self.pop) ]
        self.currentScores = [ np.inf for x in range(self.pop) ]
        self.nextScores = [ np.inf for x in range(self.pop) ]

        self.sigma = self.firstSigma

        if None == self._best:
            self._best = np.random.rand(dim)
            self._bestScore = np.inf
        else:
            self._bestScore = E(np.array(self._best))
            self._best = np.array(self._best)

        self._hist = []

    def _setSigma(self):
        p = 1.0
        if 0 != len(self._hist):
            p = np.mean(self._hist)/float(self.pop)
        else:
            return

        if 0.2 < p:
            self.sigma = self.sigma/self.c
        else:
            self.sigma = self.sigma*self.c

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

    def _rewriteBest(self,E):
        for i in range(self.pop):
            if self.currentScores[i] <= self._bestScore:
                self._best = self.currents[i].copy()
                self._bestScore = self.currentScores[i]

    def _oneFifth(self,dim,k):
        for i in range(self.pop):
            self.nexts[i] = self.currents[i]+self.sigma*np.random.randn(dim)

        if self.torus:
            for i in range(self.pop):
                for j in range(dim):
                    self.nexts[i][j] -= int(self.nexts[i][j])
                    if self.nexts[i][j] < 0:
                        self.nexts[i][j] += 1

    def _moveToNext(self,E):
        self.nexts, self.currents = self.currents, self.nexts
        #good vectors are in currents at this point

class CMA_ES(ES):
    def __init__(self,ite,pop,sigma=1.0,torus=False,best=None,ftarget=-np.inf,threads=1):
        ES.__init__(self,ite=ite,pop=pop,sigma=sigma,torus=torus,best=best,ftarget=ftarget,threads=threads)
        pass

    def _init(self,dim,E):
        ES._init(self,dim,E)
        self.sigma = self.firstSigma
        self.m = np.random.rand(dim)

        self.mu = self.pop/2.0

        self.ws = []
        for i in range(int(self.mu)):
            #self.ws.append( int(self.mu)+1-i )#
            self.ws.append( np.log(0.5+self.mu)-np.log(i+1) )
        self.ws = np.array(self.ws)/sum(self.ws)

        self.mu = int(self.pop/2)

        self.mueff = (sum(self.ws)**2)/sum([w**2 for w in self.ws])

        N = dim

        self.cc = (4+self.mueff/N) / (N+4 + 2*self.mueff/N) #approx 4/dim
        self.cs = (self.mueff+2) / (N+self.mueff+5) #approx 4/dim
        self.c1 = 2 / ((N+1.3)**2+self.mueff) #approx 2/(dim**2)
        self.cmu = min(1-self.c1, 2 * (self.mueff-2+1/self.mueff) / ((N+2)**2+self.mueff)) #approx mueff/(dim**2)
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(N+1))-1) + self.cs
        #self.muw = 0.3*self.pop

        self.pc = np.array([0.0]*dim)
        self.ps = np.array([0.0]*dim)
        self.B = np.eye(dim)
        self.D = np.eye(dim)*0.3#np.ones(dim)
        self.C = np.eye(dim)*0.09
        self.invsqrtC = np.eye(dim)
        self.chiN=np.sqrt(N)*(1-1/(4*N)+1/(21*(N**2)));

    def _rewriteBest(self,E):
        for i in range(self.pop):
            if self.nextScores[i] <= self._bestScore:
                self._best = self.nexts[i].copy()
                self._bestScore = self.nextScores[i]

    def run(self,E):
        dim = E.dim()

        self._init(dim,E)

        for k in range(self.ite):
            self._calcObj(E)
            self._rewriteBest(E)
            if self._bestScore < self._ftarget:
                break

            ar = []
            for score, vec in zip(self.nextScores,self.nexts):
                ar.append((score,vec))
            ar.sort(key=lambda x: x[0])
            ar = ar[:int(self.mu)]

            nar = []
            new_mean = np.array([0.0]*dim)
            for w, (score, vec) in zip(self.ws,ar):
                new_mean += w*vec
                nar.append(vec-self.m)
            old_mean = self.m
            self.m = new_mean/sum(self.ws)
            y = (self.m-old_mean) / self.sigma
            self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * np.dot(self.invsqrtC, y)
            '''
            pn = np.linalg.norm(self.ps)
            if 0 < pn and pn < 1.5*np.sqrt(dim):
                hsig = 1
            else:
                hsig = 0
            '''
            hsig = int(np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*k/self.ite))/self.chiN < 1.4 + 2.0/(dim+1))
            self.pc = (1-self.cc)*self.pc + hsig * np.sqrt(self.cc*(2-self.cc)*self.mueff) * y

            nm = np.zeros((dim,dim))
            for w, nvec in zip(self.ws,nar):
                nm += w*(nvec[:,np.newaxis]*nvec)
            #nm = scipy.vstack(stack)
            #nm = nm.T

            self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc[:,np.newaxis]*self.pc + (1-hsig) * self.cc*(2-self.cc) * self.C)+ self.cmu * nm#  + self.cmu * nm#np.dot(np.diag(self.ws), np.dot(nm, nm.T))
            self.sigma = self.sigma * np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN - 1))
            try:
                lamds, self.B = np.linalg.eigh(self.C)
            except:
                raise
            self.D = np.diag(np.sqrt(lamds))
            invD = np.diag(1/np.sqrt(lamds))
            self.invsqrtC = np.dot(np.dot(self.B,invD),self.B.T)

            BD = np.dot(self.B,self.D)
            for i in range(self.pop):
                vec = self.m + self.sigma * np.dot(BD, np.random.randn(dim))
                self.nexts[i] = vec

            #self._setSigma()
            #self._oneFifth(dim, k)
            #self._calcObj(E)
            #self._moveToNext(E)
        self._rewriteBest(E)

        return self._best

