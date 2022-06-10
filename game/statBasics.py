# # following tutorial from 
# # https://machinelearningmastery.com/introduction-to-expected-value-variance-and-covariance/#:~:text=The%20covariance%20matrix%20can%20be,as%20one%20for%20each%20variable.

# #covering expected value, variance, covariance with NUMPY 




# ### MEAN EXAMPLE 

from numpy import array
# from numpy import mean
# v = array([1,2,3,4,5,6])
# print(v)
# result = mean(v)
# print(result)



# from numpy import array
# from numpy import mean
# M = array([[1,2,3,4,5,6],
#            [1,2,3,4,5,6]]) 
# ## calculate row and column means 

# col_mean = mean(M, axis=0)
# print(col_mean)
# row_mean = mean(M, axis=1)
# print(row_mean)


# ## variance example 

# from numpy import var
# v = array([1,2,3,4,5,6])
# result = var(v, ddof=1)
# print(result)


# # this calcs both row and column var 
# M = array([[1,2,3,4,5,6],[1,2,3,4,5,6]])

# col_var = var(M, ddof=1, axis=0)
# print(col_var)
# row_var = var(M, ddof=1, axis=1)
# print(row_var)



### COVARIANCE !! !
## cov is the measure of join prob between two random variables 
## eg cov(X,Y)


from numpy import cov
x = array([1,2,3,4,5,6,7,8,9])
y = array([9,8,7,6,5,4,3,2,1])
Sigma = cov(x,y)[0,1]
# Sigma = cov(x,y)
print(Sigma)



# The covariance can be normalized to a score between -1 and 1 to make the magnitude interpretable by dividing it by the standard deviation of X and
#  Y. The result is called the correlation of the variables, also called the Pearson correlation coefficient, named for the developer of the method.
# r = cov(X, Y) / sX sY

