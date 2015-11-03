'''
Created on 2013/06/19

@author: admin
'''
import numpy as np
import optlib.benchmarks
import matplotlib.pylab as plt
import optlib.pso
import os
import time
import functools
import scipy.sparse as ss
import pickle
import random
import math
import copy
import optlib.de
import optlib.es
import optlib.firefly

@functools.lru_cache(maxsize=None)
def f(x):
    print(os.getpid())
    return x**10



'''
def mont(p,q,k,ite):
    s = 0
    flag = False
    for i in range(ite):
        r = random.random()
        if flag:
            if r < k:
                s += 1
                flag = False
        else:
            if r < q:
                flag = True
            elif r < q+p:
                s += 1
    return s

def sim(p,q,k,ite=50):
    ar = [0]*35
    for x in range(1000):
        index = min(34,mont(p,q,k,ite))
        ar[ index ] += 1
    return ar

class Model:
    def __call__(self,x):
        if min(x) < 0 or 1 < max(x):
            return (np.inf,0)

        p = x[0]
        q = x[1]
        k = x[2]

        target = [4,38,53,159,416,285,38,6,0,0,0,0,0]

        error = 0
        for x,y in zip(target,sim(p,q,k)):
            error += (x-y)**2

        return (error,error)

    def dim(self):
        return 3
'''

if __name__ == '__main__':
    '''
    ar = []
    means = []
    for ite in [100,200,300]:
        ar = [0]*60
        sucs = []
        for i in range(1000):
            #pso = optlib.pso.Spso(ite,10)
            pso = optlib.de.De(ite,10)
            obj = optlib.benchmarks.Fractal(5)
            best = pso.run(obj)
            suc = obj.quality(best)
            sucs.append(-suc)
            ar[suc] += 1
        #for i,x in enumerate(ar):
        #    print(i,x)
        plt.hist(sucs,bins=50,normed=True,label='$K='+str(ite)+'$')
        print(np.mean(sucs))
        means.append(np.mean(sucs))
        #plt.show()
    plt.legend()
    #plt.axis([0,25,0,6])
    plt.xlabel('$m$',fontsize=16)
    plt.show()
    exit()

    means = [5.706,10.191,13.837]

    for mean in means:
        sucs = []
        for i in range(1000):
            suc = np.random.poisson(mean)
            sucs.append(suc)
        plt.hist(sucs,bins=50,normed=True,label='$\lambda='+str(mean)+'$')
        #plt.show()
    plt.legend()
    plt.axis([0,25,0,0.7])
    plt.xlabel('$m$',fontsize=16)
    plt.show()
    exit()
    '''
    ite=1000
    pop=20

    algs = [('ldvm',optlib.pso.LdvmPSO(ite,pop)),\
            ('skpso',optlib.pso.PSO(ite,pop,c0=0.6,c1=1.7,c2=1.7,vlim=False,torus=False,best=None)),\
            ('koguma',optlib.pso.Kpso(ite,pop)),\
            ('ndt',optlib.pso.NdtPSO(ite,pop,d0=2)),\
            ('spso',optlib.pso.PSO(ite,pop,c0=0.729,c1=1.49455,c2=1.49455,vlim=False,torus=False,best=None)),\
            ('iwa',optlib.pso.LdiwPSO(ite,pop)),\
            ('afpso',optlib.pso.AfePSO(ite,pop))]

    algs = [('de',optlib.de.De(ite,pop,f=0.5,cr=0.9,torus=True)),\
            ('jde',optlib.de.Jde(ite,pop,torus=True)),\
            ('sde',optlib.de.Sde(ite,pop,mu=0.8,torus=True)),\
            ('adapt',optlib.de.AdaptDe(ite,pop,torus=True)),\
            ('jade',optlib.de.Jade(ite,pop,p=0.1,c=0.1,torus=True)),\
            ('apde',optlib.de.Apde(ite,pop,torus=True)),\
            ]

    algs = [('gppso',optlib.pso.GpPSO(ite,pop)),\
            ('geopso',optlib.pso.GeoPSO(ite,pop)),\
            ]

    algs = [('gppso',optlib.pso.GpPSO(ite,pop))]

    algs = [('mathfly',optlib.firefly.MathFly(ite,pop))]

    algs = [('jade',optlib.de.Jade(ite,pop,p=0.1,c=0.1,torus=True))]

    algs = [('ds',optlib.de.Ds(ite,pop,torus=True))]

    algs =[('firefly0.1,0.1',optlib.firefly.Firefly(ite,pop,alpha=0.25,beta=0.5,gamma=2.4)),\
           ('firefly0.2,0.1',optlib.firefly.Firefly(ite,pop,alpha=0.01,beta=0.02,gamma=16.0,variant='nmfa')),\
           ]

    '''
    algs = [('CMA-ES',optlib.es.ES(20000,1)),\
            ('1+1-ES',optlib.es.ES(20000,1)),\
           ('20+20-ES',optlib.es.ES(200,100))]

    algs = [('CMA-ES',optlib.es.CMA_ES(200,100,torus=True))]

    algs =
    '''

    diim = 10
    probs = [optlib.benchmarks.Fractal(diim),optlib.benchmarks.Minima(diim),optlib.benchmarks.Rastrigin(diim),optlib.benchmarks.Rosenbrock(diim)]
    objs = [optlib.benchmarks.Fractal(diim),optlib.benchmarks.Minima(diim),optlib.benchmarks.Rastrigin(diim),optlib.benchmarks.Rosenbrock(diim)]
    for obj in probs:
        objs.append(optlib.benchmarks.Rotate(obj))

    #objs = [optlib.benchmarks.Rastrigin(diim)]
    #algs = [('ndt',optlib.pso.NdtPSO(ite,pop,d0=2))]
    #algs = [('iwa',optlib.pso.LdiwPSO(ite,pop))]
    #algs = [('spso',optlib.pso.PSO(ite,pop,c0=0.729,c1=1.49455,c2=1.49455,vlim=False,torus=False,best=None))]
    #algs = [('afpso',optlib.pso.AfePSO(ite,pop))]
    #objs = [optlib.benchmarks.Rotate(optlib.benchmarks.Fractal(6))]

    dir = 'fireflychap2'
    try:
        os.mkdir(dir)
    except(FileExistsError):
        pass

    for obj in objs:
        print(obj.name())
        for name, alg in algs:
            path = os.path.join(dir,name)
            try:
                os.mkdir(path)
            except(FileExistsError):
                pass
            f = open(os.path.join(path,obj.name()),'w')
            ar = []
            for i in range(50):
                pso = copy.deepcopy(alg)
                best = pso.run(obj)
                ar.append( obj.quality(best) )
                #print(ar[-1])
            f.writelines(["{:.9f}\n".format(val) for val in ar])
            print(name, ite, np.median(ar), np.mean(ar),np.var(ar,ddof=1),np.min(ar),np.max(ar))
    exit()