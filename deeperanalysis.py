from MLMODELING import target_vector , model_predictions
import numpy as np 

mean_GPA = np.mean(target_vector)

# Now I want to calcualte the Residual Sum of Squares which is 
# Summation (y_real - y_predicted) ^2 -  the square happens for every single data 

sumOfResidaualSquares  = np.sum(((target_vector - model_predictions )**2) )

print(f"The sum of residual Squares Aka the total Error of the model: {sumOfResidaualSquares}")

# Next Is that we are going to calculate the Base line total error - we assume that the base line strategy 
# for a model trying to reduce its error is to predict the mean for all predictions so we see what will be the error
# in that case 

total_baseline_error = np.sum(((target_vector - mean_GPA )**2))

print(f"This will be the total Error of the model if it simply predicted The mean for every single Value\n{total_baseline_error}")

# Now as you can see 
# The total Error for the model ( Residual sum of squares) is 107.5k ~ GPA points all over 8 million samples
# Now the total Error for the Baseline Model - the mean ( every prediction is mean is)  ~ 211.1K over 8 million samples 

# so basically error per student is like 0.013 GPA Points for the trained model
# and error per student for the dumb model is 0.026375 GPA points for the dumb model 

# R^2 which is the coefficient of determinnance basiicaly captures - the amount by which 
# the model is able to correctly catch the wind - meaning 
# do the weights and bias values that we have result to predictions that move in the right direction ( up down ) when 
# a feature value is moved up or down - and does the prediction value move by the right amount 

# Formula for R^2 = 1 - (Residaul_sum_of_squares/ total_baseline_error)
# 