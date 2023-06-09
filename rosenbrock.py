"""
MATH 197 Z Exercise 3
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 13 June 2023
"""
import numpy as np
from optrosen2 import fletcher_reeves, hager_zhang, broyden_fgs, barzilai_borwein, dogleg
import sys

def rosenbrock(x):
    """
	Parameter
	---------
 		x: array
			input n dimensional array

	Returns
	-------
		a:float
            rosenbrock function
	"""

    a = 0
    i = 0
    while i<99:
        a = a + 100.0*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
        i=i+1

    return a

def grad_rosenbrock(x):
    """
	Parameter
	---------
		x:array
		  input n dimensional array

	Returns
	-------
		dx:2d vector
          gradient
	"""
    dx = np.zeros(100)	

    i = 1
    dx[0] = 400.0*x[0]**3 - 400.0*x[0]*x[1] + 2*x[0] - 2
    
    while i < 99:
        dx[i] = dx[i] + 400.0*x[i]**3 - 400.0*x[i+1]*x[i] + 2*x[i] - 2 + 200.0*(x[i]-x[i-1]**2)
        i = i+1
    
    dx[i]= 200.0*(x[i]-x[i-1]**2)
    
    return dx


if __name__ == "__main__":
    choice = input("Choose a method:\n 1. Fletcher-Reeves NLCG\n 2. Hager-Zhang NLCG\n 3. Nonlinear Barzilai–Borwein Gradient\n 4. BFGS Quasi-Newton\n 5. Dogleg Trust Region\n Input name/number: ")
    x = np.zeros(100)
    if choice == 'Fletcher-Reeves NLCG' or choice == '1':
        x, grad_norm, it = fletcher_reeves(rosenbrock, x, grad_rosenbrock)
    elif choice == 'Hager-Zhang NLCG' or choice == '2':		
        x, grad_norm, it = hager_zhang(rosenbrock, x, grad_rosenbrock)
    elif choice == 'Nonlinear Barzilai–Borwein Gradient' or choice == '3':		
        x, grad_norm, it = barzilai_borwein(rosenbrock, x, grad_rosenbrock)
    elif choice == 'BFGS Quasi-Newton' or choice == '4':		
        x, grad_norm, it = broyden_fgs(rosenbrock, x, grad_rosenbrock)
    elif choice == 'Dogleg Trust Region' or choice == '5':		
        x, grad_norm, it = dogleg(rosenbrock, x, grad_rosenbrock)
    else:
        print("Please input a valid number or the exact method name.")
        sys.exit()
        
    print("Approximate Minimizer: {}" .format(x))
    print("Gradient Norm 		: {}" .format(grad_norm))
    print("Number of Iterations	: {}" .format(it))
    print("Function Value		: {}" .format(rosenbrock(x)))
