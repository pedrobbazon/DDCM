 # Lagrangian:
 
 # Proximity function
 kappa = params['kappa']
 dg = g - gt
 ds = s - st
 L = 0.5*inner(dg, dg)*dx + 0.5*kappa*fd.inner(ds, dsa)*dx

 # Balance and Compatibility constraint
 L -= (inner(s, grad(lam)) + inner(zeda*u - q(x), lam))*dx
 L += inner(e - grad(u), mu)*dx
    
 # Stabilization terms
 resC = e - grad(u)
 L += 0.5*alpha*inner(resC,resC)*dx

 resE = e + mu - gt
 L  -= 0.5*gamma*inner(resE,resE)*dx

 resK = kappa*s - grad(lam) - kappa*st
 L  -= (0.5*eta/kappa)*inner(resK,resK)*dx

 resB = div(s) + zeda*u - q(x)
 L += (0.5*theta*kappa*ls**2)*inner(resB,resB)*dx

 resM = div(mu) + zeda*lam
 L -= (0.5*beta*lmu**2)*inner(resM,resM)*dx
