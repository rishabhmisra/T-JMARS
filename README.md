# Implementation of JMARS
Jointly Modelling Aspects, Ratings and Sentiments for Movie Recommendation (JMARS)

http://www.andrew.cmu.edu/user/chaoyuaw/jmars_kdd2014.pdf

To run the code, run jmars.py as a python file:

python jmars.py

File Descriptions:

constants.py - Contains constants and global variables

indexer.py - Contains code to read imdb data and extract relevant information

optimize.py - Contains code to run optimization needed in the M-Step

sampler.py - Contains code to run Gibbs Sampling needed in the E-Step

jmars.py - Contains main code which uses the other modules and runs Gibbs Expectation-Maximization to output predicted ratings
