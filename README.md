<h2>Rotation Invariant Householder Parameterization for Bayesian PCA</h2>

For more details see our paper published at the ICML 2019: [http://proceedings.mlr.press/v97/nirwan19a.html](http://proceedings.mlr.press/v97/nirwan19a.html)

We recommend to run the code in a virtualenv with packages specified in requirements.txt.


### Following improvements made (in addition to the paper) - in "improved" files 
1. Sampling of vs: instead of `v ~ normal(0,1)`, we put a prior on the radius of the unit vector
`sqrt(dot_self(v)) ~ gamma(100,100)` and correct for the Jacobian `-log(r)*(D-q)` for each $ v \in \mathbb{S}^{q-1}$.   
   There are some issues when sampling from a unit sphere in Stan if the dimension of the vector is too low. For more details, see [Stan Forum](https://discourse.mc-stan.org/t/divergence-treedepth-issues-with-unit-vector/8059/3). This point deals with that.   
2. Initialize the parameters in HMC in Stan with the classical PCA solution (leads to faster convergence). 


### Todo:
1. Householder parameterization via vs has a discontinuity, which sometimes causes divergences in the HMC trajectory. Explanation will follow soon.
