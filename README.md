# r_forecasts
Tensor-to-scalar forecasting with PyMC3

Used in validation of NUTS component separation code. Also useful for general MCMC forecasting of next-gen CMB experiments.

Core script is pymc3_BB.py, which defines the B-mode likelihoods class, and defines sampling and plotting routines. pseudo_Cl.py is also provided, which will perform power spectrum estimation given some recovered CMB Q and U maps. 

(Data used in running scripts in stored on Oxford Glamdring cluster).
