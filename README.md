# MasterThesis
Analysis of boundary artifact of stochastic differential equation caused by reflection scheme.

This repository contain all the code for generate all the result in my master thesis. Most of the code is realted to well mixed condition test. It is paralle code. This may be difficult to undstand the code at the beginning. Let us start with something simple.

```python
def SDE_Single_Step(z,dt):

T_end = 3600*24
dt = 10
H_h = 2
N_b = 200

for i in range(int(T_end/dt)):
  z = z + SDE_Single_Step(z,dt)
  hist, _ = np.histogram(z, bins=np.linspace(0,N_b))
```
