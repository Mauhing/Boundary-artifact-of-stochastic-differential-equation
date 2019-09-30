# Boundary artifact of stochastic differential equation
Analysis of boundary artifact of stochastic differential equation caused by reflection scheme.

This repository contain all the code for generate all the result in my master thesis. Most of the code is resulted to well mixed condition test. It is paralle computation code. This may be difficult to undstand the code at the beginning. Let us start with something simple.

```python
def SDE_Single_Step(z,dt):
  # stochastic differential equation scheme.
  return z
def refelctive_scheme(z, H):
  # Boundary condition
  return z

T_end = 3600*24
dt = 10
H = 10
H_h = 2
N_b = 200
Ntime = T_end/dt
hist_ = 0
bins = np.linspace(0, H_h, N_b)
dz = bins[1] - bins[0]

for i in range(int(T_end/dt)):
  z = z + SDE_Single_Step(z,dt)
  temp, _ = np.histogram(z, bins=np.linspace(0,N_b))
  hist_ = (hist_*H) / ( dz))
  
hist_ = (hist_*/(Ntime+1))
```
where the last line is normalize the concentration. This is the code just for single particles. For many particle, we can just repeat this process and add the histogram together and divided by total particles since particles do not interact each other. The reader may found that the code I am using is a bit different. It is just because I am taking a count for the perfomance. But they are essentially doing the same thing.

Now, we assume we have 4 cores processor. It should be easy to parallelise the code since they do not interact each other. Let's say that we have `Np` so many particles. We can divide them in 4 and launch them in each processor. To do this, we define a function, called `simulation`, which will divide total particles in 4 and launch them in 4 proceses. We have also a function called `parallel` which is essentially doing the same above and return the histogram. And then, `simulation` wait for all 4 `parallel` process to complete and return. To the end. It gives out us the final output.

```python
def oneStep(Z ,scheme, H, dt):
    
    Z_Origin = Z.copy()
    Z = scheme(Z, H, dt, Z.size)
    
    maskCross = ((H < Z) | (0 > Z)) 
    Crossing = Z_Origin[maskCross]
    
    Z = np.where(Z < 0, -Z, Z)
    Z = np.where(Z > H, 2*H - Z, Z)
    
    Landing = Z[maskCross]
    
    return Z, Crossing, Landing

def parallel(Tmax, dt, H, Testdepth, Np, Nbins, quene, queneCross, queneLand,
             scheme):

    np.random.seed()
    Ntime = int(Tmax / dt)

    hist_ = np.zeros((Nbins - 1, ), 'uint64')
    hist_Cross = np.zeros((Nbins - 1, ), 'uint64')
    hist_Land = np.zeros((Nbins - 1, ), 'uint64')

    #z = np.linspace(0, H, int(Np))
    z = np.random.uniform(0, H, int(Np))

    temp0, _ = np.histogram(z, bins=np.linspace(0, Testdepth, Nbins))
    hist_ = hist_ + temp0
    
    for i in range(Ntime):

        z, Cross, Land = oneStep(z, scheme, H, dt)

        #Adding the histogram
        ###
        temp0, _ = np.histogram(z, bins=np.linspace(0, Testdepth, Nbins))
        hist_ = hist_ + temp0

        temp2, _ = np.histogram(Cross, bins=np.linspace(0, Testdepth, Nbins))
        hist_Cross = hist_Cross + temp2

        temp3, _ = np.histogram(Land, bins=np.linspace(0, Testdepth, Nbins))
        hist_Land = hist_Land + temp3
        
        try:
            if (i % int(Ntime / 100) == 0):
                print("\r %6.2f" % (i * 100 / Ntime + 1),
                      "%",
                      end="\r",
                      flush=True)
        except ZeroDivisionError as err:
            None
            
    quene.put(hist_)
    queneCross.put(hist_Cross)
    queneLand.put(hist_Land)

    return None
    
def RunSimulation(NumberOfThread, Tmax, dt, H, Testdepth, Np, Nbins, scheme):
    print("Total number of process: ", NumberOfThread)
    SubNp = np.full((NumberOfThread, ), int(Np / NumberOfThread))
    SubNp[-1] = SubNp[-1] + (Np % NumberOfThread)
    
    Ntime = int(Tmax / dt)  #Number of time interval

    if (__name__ == '__main__'):

        threads = []
        quene = mp.Queue()
        queneCross = mp.Queue()
        queneLand = mp.Queue()

        for i in range(NumberOfThread):
            thread = mp.Process(target=parallel,
                                args=(Tmax, dt, H, Testdepth, SubNp[i], Nbins,
                                      quene, queneCross, queneLand, scheme))

            threads.append(thread)
            thread.start()  #starting calculation.

        for thread in threads:
            thread.join()  #waiting these processes finish.

    hist_ = np.zeros((Nbins - 1, ), 'float64')
    hist_Cross = np.zeros((Nbins - 1, ), 'float64')
    hist_Land = np.zeros((Nbins - 1, ), 'float64')

    for i in range(NumberOfThread):
        hist_ += quene.get()
        hist_Cross += queneCross.get()
        hist_Land += queneLand.get()
        
    bins = np.linspace(0, Testdepth, Nbins)
    dz = bins[1] - bins[0]
    hist_ = (hist_*H) / (Np * dz * (Ntime+1))
    
    midpoints = bins[:-1] + (bins[1]-bins[0]) / 2
    hist_Cross = hist_Cross / (np.sum(hist_Cross)*(midpoints[1]-midpoints[0]))
    hist_Land = hist_Land / (np.sum(hist_Land)*(midpoints[1]-midpoints[0]))

    return hist_, hist_Cross, hist_Land
```

This will be the engine to power the most WMC test in my master thesis. However, the reader may found that I have small modification for each file. It is ecentially just make it even faster, or let it to adapt to the mirrored domain. 


