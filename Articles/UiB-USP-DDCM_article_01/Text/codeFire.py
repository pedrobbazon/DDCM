 # Lagrangian:
  
 x = fd.SpatialCoordinate(mesh)
 # Proximity function
 L = ObjFunc(e, graddata, s, fluxdata)

 # Balance and Compatibility constraint
 L -= (fd.inner(s, fd.grad(lam)) + fd.inner(zeda*u - q(x), lam)) * fd.dx
 L += fd.inner(e - fd.grad(u), mu) * fd.dx
    
 # Stabilization terms
 resC = e - fd.grad(u)
 L += 0.5*alpha*fd.inner(resC,resC) * fd.dx

 resE = e + mu - graddata
 L  -= 0.5*gamma*fd.inner(resE,resE) * fd.dx

 resK = kappa*s - fd.grad(lam) - kappa*fluxdata
 L  -= (0.5*eta/kappa)*fd.inner(resK,resK) * fd.dx

 resB = fd.div(s) + zeda*u - q(x)
 L += (0.5*theta*kappa*ls**2)*fd.inner(resB,resB) * fd.dx

 resM = fd.div(mu) + zeda*lam
 L -= (0.5*beta*lmu**2)*fd.inner(resM,resM) * fd.dx
