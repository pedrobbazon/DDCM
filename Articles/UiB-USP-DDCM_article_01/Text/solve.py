 # Test function
 dw = TestFunction(W) 

 # Optimallity conditions
 dL = derivative(L, w, dw)

 # Jacobian
 Jac = derivative(dL, w)

 # Non-linear solve (see solver parameters above)
 solve(dL == 0, w, bcs=bc_lst, J=Jac)
