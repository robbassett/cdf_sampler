# cdf_sampler
Naive inverse cumulative distribution sampler

Creates a cumulative distribution naively from any function y(x). Then the function sample_n(N)
creates a sample of N uniformly sampled values between 0 and 1 (from numpy.random.uniform),
checks index where the cumulative distribution is closest to each value, then saves the x value
associated with the selected index. In this way, a distribution of N values is built up that
has a distribution matched to the input function y. 

[Usage examples can be found in the jupyter notebook included in this repo](CDF_sampler_exampples.ipynb)
