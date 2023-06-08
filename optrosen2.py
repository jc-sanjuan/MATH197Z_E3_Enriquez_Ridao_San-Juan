"""
MATH 197 Z Exercise 3
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 13 June 2023
"""

import numpy as np 
from numpy import *
import math
import sys

def backtracking(fun, d, x, fx, gx, c1=1e-4, rho=0.5, alpha_in=1.0, maxback=30):
    """
	Parameters
	----------
        fun: callable
          objective function
		d:array
		  current direction
		x:array
		  current point
		fx:float
		  function value at x
		gx:array
		  gradient at x
        g:callable
		  gradient of the objective function
        ssc:string/int
          chosen stepsize criterion
		c1:float
		  armijo parameter (default value is 1e-4)
		rho:float 
		  backtracking parameter (default value is 0.5)
		alpha_in:float
		  initial step length (default value is 1.0) 
		maxback:int 
			max number of backtracking iterations (default is 30)

	Returns
	-------
		alpha: float
			steplength satisfying the chosen rule or the last steplength
	"""
    alpha = alpha_in
    q = np.dot(gx, d)
    j = 0
    armijo = fun(x + alpha*d)>fx+c1*alpha*q
    
    while armijo and j< maxback:
        alpha = rho*alpha
        j = j+1
        armijo = fun(x + alpha*d)>fx+c1*alpha*q
    return alpha 



def fletcher_reeves(fun, x, grad, tol=1e-6, maxit=50000):
    
    """
	Parameters
	----------
		fun:callable
			objective function
		x:array
			initial point
		gradfun:callable
			gradient of the objective function
		tol:float
			tolerance of the method (default is 1e-6)
		maxit:int
			maximum number of iterations

	Returns
	-------
		tuple(x,grad_norm,it)
			x:array
				approximate minimizer or last iteration
			grad_norm:float
				norm of the gradient at x
			it:int
				number of iteration
	"""


    f = fun(x)
    g = grad(x)
    grad_norm = np.linalg.norm(g)
    d = np.negative(g)
    it = 0
    while grad_norm>=tol and it<maxit:
        alpha = backtracking(fun,d,x,f,-d)
        
        grad_normPrev = grad_norm
        
        x = x + alpha*d
        g = grad(x)
    
        grad_norm = np.linalg.norm(grad(x))
        B = (grad_norm**2)/(grad_normPrev**2)
        d = np.negative(g) + np.dot(B,d)

        it = it + 1
        
    return x,grad_norm,it


def hager_zhang(fun, x, grad, tol=1e-6, maxit=50000):
    
    """
	Parameters
	----------
		fun:callable
			objective function
		x:array
			initial point
		gradfun:callable
			gradient of the objective function
		tol:float
			tolerance of the method (default is 1e-6)
		maxit:int
			maximum number of iterations

	Returns
	-------
		tuple(x,grad_norm,it)
			x:array
				approximate minimizer or last iteration
			grad_norm:float
				norm of the gradient at x
			it:int
				number of iteration
	"""


    f = fun(x)
    g = grad(x)
    grad_norm = np.linalg.norm(g)
    d = np.negative(g)
    it = 0
    while grad_norm>=tol and it<maxit:
        alpha = backtracking(fun,d,x,f,-d)
        
        grad_normPrev = grad_norm
        g_old = g
        x = x + alpha*d
        g = grad(x)
        grad_norm = np.linalg.norm(grad(x))
        if it > 0:

            y_old = y
        
            y = g - g_old
            
            B = np.dot((y - (2*((np.linalg.norm(y)**2)/np.dot(d,y_old))*d)),(g/np.dot(d,y_old)))
            d = np.negative(g) + np.dot(B,d)
        else:
            y = g - g_old
            d = np.negative(g)
            
        it = it + 1
        
    return x,grad_norm,it

def barzilai_borwein(fun, x, grad, tol=1e-6, maxit=50000):
    
    """
	Parameters
	----------
		fun:callable
			objective function
		x:array
			initial point
		gradfun:callable
			gradient of the objective function
		tol:float
			tolerance of the method (default is 1e-6)
		maxit:int
			maximum number of iterations

	Returns
	-------
		tuple(x,grad_norm,it)
			x:array
				approximate minimizer or last iteration
			grad_norm:float
				norm of the gradient at x
			it:int
				number of iteration
	"""


    g_old = grad(x)
    x_old = x
    x = x_old - g_old
    g = grad(x)
    grad_norm = np.linalg.norm(g)
    it = 1
    
    while grad_norm>=tol and it<maxit:
        S = x - x_old
        y = g - g_old
        #y_norm = np.linalg.norm(y)
        #alpha = (np.dot(S,y))/((y_norm)**2)
        alpha = ((np.linalg.norm(S))**2)/(np.dot(y, S))
        
        x_old = x
        x = x - alpha*g
        
        grad_norm = np.linalg.norm(g)
        g_old = g
        g = grad(x)
        
        it = it + 1
        
    return x_old,grad_norm,it