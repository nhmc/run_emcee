# first define the model in definitions.py.

# adjust the number of burn-in samples in emcee.cfg, and set Nmcmc=0
run_mcmc

# inspect the burn-in
plot_mcmc samples_burn.sav

# re-run with a longer burn-in if necessary. Otherwise set Nburn=0 and
# set the number of mcmc samples Nmcmc.
run_mcmc

# then look at the results using the final sample
plot_mcmc samples_mcmc.sav
