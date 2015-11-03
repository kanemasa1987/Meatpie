# Meatpie
Collections of well known metaheuristic algorithms fully written in Python. Including Particle Swarm Optimization (PSO), Differential Evolution (DE), Evolution Strategies (ES), FIrefly Algorithm (FA), etc, and some of their variations.

# Constraints
Use penalty method for the constraints you have.
The algorithm's initial points are set in x \in (0,1).
Make sure you rescale the variables.

# Code
import optlib.de

class ObjectiveFunction:
    def __call__(self,x):
        return sum(x)
    def dim(self):
        #dimension of optimization problem
        return 10

iteration=1000
population=20
opt = optlib.de.De(iteration,pop)
vec = opt.run(ObjectiveFunction())

print(vec)
