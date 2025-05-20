 # Lagrangian:
 if(formulation == 'Natural'):
   SF = FunctionSpace(msh, family="CG", degree=1)
   VF = VectorFunctionSpace(msh, family="DG", degree=0)
 elif(formulation == 'Equal-Order'):
   SF = FunctionSpace(msh, family="CG", degree=1)
   VF = VectorFunctionSpace(msh, family="CG", degree=1)

 # Mixed space:
 #   s    u    e    lam  mu
 W = VF * SF * VF * SF * VF

 # Unknown field
 w = Function(W)
 s, u, e, lam, mu = split(w)

 kap = pars['kappa'] 
 # Proximity function
 dgrad, dflux = g-gt, s-st
 L = 0.5*inner(dgrad,dgrad)*dx + 0.5*kap*inner(dflux,dflux)*dx

 # Balance and Compatibility constraint
 L -= (inner(s, grad(lam)) + inner(pars['zeda']*u - q(x), lam))*dx
 L += inner(e - grad(u), mu)*dx
    
 # Stabilization terms (Equal-Order formulation)
 if(formulation == 'Equal-Order'): 
   resC = e - grad(u)
   L += 0.5*pars['alpha']*inner(resC,resC)*dx

   resE = e + mu - gt
   L -= 0.5*pars['gamma']*inner(resE,resE)*dx

   resK = kap*(s-st) - grad(lam)
   L -= (0.5*pars['eta']/kap)*inner(resK,resK)*dx

   resB = div(s) + pars['zeda']*u - q(x)
   L += (0.5*pars['theta']*kap*pars['ls']**2)*inner(resB,resB)*dx

   resM = div(mu) + pars['zeda']*lam
   L -= (0.5*pars['beta']*pars['lmu']**2)*inner(resM,resM)*dx
