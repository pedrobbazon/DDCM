 # Test function
 dw = TestFunction(W) 
 # Optimallity conditions
 dL = derivative(L, w, dw)
 # Jacobian
 Jac = derivative(dL, w)
 # solve problem
 bclist = [DirichletBC(W.sub(1), Constant(0.0), 'on_boundary'), 
           DirichletBC(W.sub(3), Constant(0.0), 'on_boundary')]
 solve(dL == 0, w, bcs=bclist, J=Jac)
