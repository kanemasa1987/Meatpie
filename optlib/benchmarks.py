'''
Created on 2015/02/23

@author: Minoru Kanemasa
'''
import numpy as np
import scipy.sparse

class ObjectiveFunction:
    def __init__(self,E,dim,best=None):
        self._E = E
        self._dim = dim

    def __call__(self,x):
        return self._E(x)

    def dim(self):
        return self._dim

class BestInfo:
    def __init__(self,best=None):
        self._best = best

    def best(self):
        if self._best:
            return self._best
        else:
            raise NotImplementedError

class NameInfo:
    def __init__(self,name=None):
        self._name = name
    def name(self):
        if self._name:
            return self._name
        else:
            return self.__class__.__name__

class Rotate(ObjectiveFunction):
    def __init__(self,E,degree=20):
        dim = E.dim()
        alpha = degree*np.pi/180

        self._E = E
        self._R = scipy.sparse.identity(dim)
        for i in range(dim-1):
            for j in range(i+1,dim):
                m = scipy.sparse.identity(dim,format='lil')
                for k in range(dim):
                    for l in range(dim):
                        if (k == i) and (l == i):
                            m[k,l] = np.cos(alpha);
                        elif (k == i) and (l == j):
                            m[k,l] = -np.sin(alpha);
                        elif (k == j) and (l == i):
                            m[k,l] = np.sin(alpha);
                        elif (k == j) and (l == j):
                            m[k,l] = np.cos(alpha);
                        elif (k == l) and not((l == i) or (l == j)):
                            m[k,l] = 1.0;
                        else:
                            m[k,l] = 0;
                self._R = self._R*m

    def __call__(self,x):
        return self._E( self._R*(x-self._E.best())+self._E.best() )

    def dim(self):
        return self._E.dim()

    def best(self):
        self._E.best()

    def quality(self,x):
        return self._E.quality( self._R*(x-self._E.best())+self._E.best() )

    def name(self):
        return self._E.name() + ' (Rotated)'

class BenchmarkMaker(ObjectiveFunction,BestInfo,NameInfo):
    def __init__(self,E,dim,best=None):
        ObjectiveFunction.__init__(self,E,dim)
        BestInfo.__init__(self,best)
        NameInfo.__init__(self)

    def __call__(self,x):
        return np.sum(self._E(x))

    def quality(self,x):
        return self.__call__(x)

def _minima(vec):
    x = vec*10.0-5.0
    return x**4-16*(x**2)+5*x

_minimabest = (-2.9035+5.0)/10.0

def _rastrigin(vec):
    x = vec*10.24-5.12
    return x**2-10*np.cos(2*np.pi*x)+10

_rastbest = 0.5

class Minima(BenchmarkMaker):
    def __init__(self,dim):
        BenchmarkMaker.__init__(self,_minima,dim,best=[_minimabest]*dim)

class Rastrigin(BenchmarkMaker):
    def __init__(self,dim):
        BenchmarkMaker.__init__(self,_rastrigin,dim,best=[_rastbest]*dim)

class Rosenbrock(ObjectiveFunction,BestInfo,NameInfo):
    def __init__(self,dim,alpha=100.0):
        if 1 == dim:
            raise NotImplementedError
        ObjectiveFunction.__init__(self,None,dim)
        BestInfo.__init__(self,[6.0/10.0]*dim)
        NameInfo.__init__(self)

        self._alpha = alpha

    def __call__(self,x):
        v = x*10.0-5.0

        ret = 0.0
        for n in range(self.dim()-1):
            ret += self._alpha*((v[n]**2-v[n+1])**2)+(1-v[n])**2
        return ret

    def quality(self,x):
        return self.__call__(x)

class Fractal(ObjectiveFunction,BestInfo,NameInfo):
    def __init__(self,dim):
        ObjectiveFunction.__init__(self,None,dim,[0.5]*dim)
        BestInfo.__init__(self,[0.5]*dim)
        NameInfo.__init__(self)

        self.w1 = 1.0/3.0
        self.w2 = 2.0/3.0

    def __call__(self,x):
        if min(x) < 0 or 1 < max(x):
            return 0

        obj = 0.0
        for elem in x:
            obj += self._frac(elem,n=30)

        return obj

    def quality(self,x):
        if min(x) < 0 or 1 < max(x):
            return 0

        mx = max(np.fabs(x-0.5))

        m = 0
        while mx <= 0.5*(self.w1**m):
            m += 1

        return -m+1

    def _frac(self,x,n=30):
        if n <= 1:
            if self.w1 <= x and x <= self.w2:
                return -self.w1
            else:
                return 0.0
        else:
            if x < self.w1:
                return self.w1*self._frac(3*x,n-1)
            elif x <= self.w2:
                return -self.w1+self.w1*self._frac(3*x-1,n-1)
            else:
                return self.w1*self._frac(3*x-2,n-1)
