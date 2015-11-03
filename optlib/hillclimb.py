'''
Created on 2013/07/10

@author: admin
'''
import random
import numpy as np

class BinaryHillClimb(object):
    def __init__(self,init=None):
        '''
        Constructor
        '''
        self.best = init

    def init(self,E):
        if None == self.best:
            self.best = [1 for elem in range(E.dim())]
        self.bestScore = E(self.best)
        print("best so far")
        print(self.bestScore)

    def run(self,E):
        self.init(E)

        cnt = 0
        path = list(range(E.dim()))
        #random.shuffle(path)
        while cnt < E.dim():
            print("cnt = %s"%cnt)
            selec = self.best.copy()
            if 0 <= cnt:
                selec[ path[cnt] ] = int(not selec[ path[cnt] ])
            score = E(selec)
            if score < self.bestScore:
                self.bestScore = score
                self.best = selec
                #with open("best.param","wb") as f:
                #    pickle.dump(best,f)
                print("best so far")
                print(self.bestScore)
                print(self.best)

                if 0 <= cnt:
                    currentIndex = path[cnt]
                    path = set(range(len(self.best)))
                    path.remove( currentIndex )
                else:
                    path = set(range(len(self.best)))
                path = list(path)
                random.shuffle(path)
                cnt = -1

            cnt += 1
        return self.best