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
        
        if(it%2 == 0):
            alpha = ((np.linalg.norm(S))**2)/(np.dot(y, S))
        else:
            y_norm = np.linalg.norm(y)
            alpha = (np.dot(S,y))/((y_norm)**2)
        
        x_old = x
        x = x - alpha*g
        
        grad_norm = np.linalg.norm(g)
        g_old = g
        g = grad(x)
        
        it = it + 1
        
    return x_old,grad_norm,it



def broyden_fgs(fun, x, grad, tol=1e-6, maxit=50000, nr=1e-8, na=1e-12):
    
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
			tolerance of the method (default is 1e-10)
		maxit:int
			maximum number of iterationd

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
    I = np.identity(100)
    B = I
    it = 0
    while grad_norm>=tol and it<maxit:
        d = np.dot(np.negative(B),g)
        alpha = backtracking(fun,d,x,f,-d)
        
        
        s = alpha*d
        g_old = g
        x = x + s
        g = grad(x)
        grad_norm = np.linalg.norm(grad(x))
        y = g - g_old
        rho = np.dot(s,y)
        
        grad_s = np.linalg.norm(s)
        grad_y = np.linalg.norm(y)
        
        #grad_normPrev = grad_norm
        C = (np.outer(s,y)/rho)
        D = (np.outer(y,s)/rho)
        E = (np.outer(s,s)/rho)
        F = I - C
        G = I - D
        
        if rho < nr*np.dot(grad_s,grad_y) or rho < na:
            B = I  
        else:
            
            B = F*B*G + E
        it = it + 1
        
    return x,grad_norm,it


def dogleg(fun, x, grad, tol=1e-6, M=50000, E=0.1):
    
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
    delta = min(grad_norm, 100)
    NewPointFlag = True
    it = 0
    
    while grad_norm>=tol and it<M:
        #print(it)
        it = it + 1
        
        if NewPointFlag:
            H = HessianApprox(f, x, g, grad, E)
            
            
        mu = np.dot(g, np.dot(H, g))
        mu1 = grad_norm**2
        
        TrialPointType = "Cauchy"
        
        BoundaryPointFlag = False
        
        if mu <= 0:
            xcauchy = x-(delta*g)/grad_norm
            BoundaryPointFlag = True
        else:
            o = min((delta/grad_norm),(mu1/mu))
            
            if mu1/mu >= delta/grad_norm:
                BoundaryPointFlag = True
                
            xcauchy = x - o*g
            Hinv = np.linalg.inv(H)
            dnewton = np.dot(Hinv,g)
            
            if np.dot(dnewton,g) < 0:
                xnewton = x + dnewton
                
                if np.linalg.norm(dnewton) <= delta:
                    TrialPointType = "Newton"
                    
                    if np.linalg.norm(dnewton) >= delta - 1e-6:
                        BoundaryPointFlag = True
                else:
                    if not BoundaryPointFlag:
                        dcauchy = -o*g
                        d = dnewton - dcauchy
                        (a, b, c) = (np.linalg.norm(d)**2, 2*np.dot(dcauchy, d), (np.linalg.norm(dcauchy)**2)-(delta**2))
                        E = (-b + np.sqrt((b**2)-4*a*c))/(2*a)
                        xcauchy = xcauchy + E*(xnewton-xcauchy)
                        BoundaryPointFlag = True
                        TrialPointType = "Dogleg"
        
        if TrialPointType == "Newton":
            xtrial = xnewton
        else:
            xtrial = xcauchy
        
        dtrial = xtrial - x
        ftrial = fun(xtrial)
        actual_reduction = ftrial - f
        predicted_reduction = np.dot(g,dtrial)+((1/2)*np.dot(dtrial, np.dot(H, dtrial)))
        ratio = actual_reduction/ predicted_reduction
        
        if ratio < 1e-4:
            delta = (1/2)*min(delta, np.linalg.norm(dtrial))
            NewPointFlag = False
        else:
            x = xtrial
            NewPointFlag = True
        
        if ratio > 0.75 and BoundaryPointFlag:
            delta = min(2*delta, 1000)
            
        if NewPointFlag:
            f = fun(x)
            g = grad(x)
            grad_norm = np.linalg.norm(g)
        
        
    return x,grad_norm,it

def HessianApprox(f, x, g, grad, E):
    n = 100
    H = np.zeros((n, n))
    
    for j in range(0,99) :
        e = np.zeros(n)
        e[j] = 1
        H[:, j] = HessianAction(f, x, e, g, grad, E)
    
    return ((H + np.transpose(H))/2)

def HessianAction(f, x, d, g, grad, E):
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        z = np.zeros(100)
    else:
        gE = grad((x+E*d)/d_norm)
        z = (gE - g)/E
    
    return z
    
