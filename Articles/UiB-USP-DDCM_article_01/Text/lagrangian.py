 # Lagrangian:
 if(formulation == 'Natural'):
   SF = FunctionSpace(msh, family='CG', degree=1)
   VF = VectorFunctionSpace(msh, family='DG', degree=0)
 elif(formulation == 'Equal-Order'):
   SF = FunctionSpace(msh, family='CG', degree=1)
   VF = VectorFunctionSpace(msh, family='CG', degree=1)

 # Mixed space:
 #   s    u    e    lam  mu
 W = VF * SF * VF * SF * VF

 # Unknown field
 w = Function(W)
 s, u, e, lam, mu = split(w)

 kap = pars['kappa'] # assert units in objective function
 # Proximity function
 dgrad, dflux = g-gt, s-st
 L = 0.5*inner(dgrad,dgrad)*dx + 0.5*kap*inner(dflux,dflux)*dx

 # Balance and Compatibility constraint
 L -= (inner(s, grad(lam)) + inner(pars['zeda']*u - q(x), lam))*dx
 L += inner(e - grad(u), mu)*dx
    
 # Stabilization terms
 if(pars['formulation'] == 'Equal-Order'): 
   CT = e - grad(u)
   L += 0.5*pars['alpha']*inner(CT,CT)*dx

   ET = e + mu - gt
   L -= 0.5*pars['gamma']*inner(ET,ET)*dx

   ST = kap*(s-st) - grad(lam)
   L -= (0.5*pars['eta']/kap)*inner(ST,ST)*dx

   BT = div(s) + pars['zeda']*u - q(x)
   L += (0.5*pars['theta']*kap*pars['ls']**2)*inner(BT,BT)*dx

   MT = div(mu) + pars['zeda']*lam
   L -= (0.5*pars['beta']*pars['lmu']**2)*inner(MT,MT)*dx
