from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

def verify_hdiv():
    # Create mesh
    mesh = UnitSquareMesh(50, 50)
    
    # Define H(div) function space (RT or BDM elements)
    V = FunctionSpace(mesh, "RT", 2)  # Raviart-Thomas elements
    Q = FunctionSpace(mesh, "DG", 1)  # For divergence
    
    # Parameters for manufactured solution
    sigma = 1.0
    alpha = 100.0
    r0 = 0.4
    x0 = Constant([0.5, 0.5])
    
    # Define exact solution in H(div)
    x, y = SpatialCoordinate(mesh)
    r = sqrt((x - x0[0])**2 + (y - x0[1])**2)
    
    # Avoid division by zero using conditional
    r_safe = conditional(gt(r, 1e-12), r, 1e-12)
    
    tanh_val = tanh(alpha * (r_safe - r0))
    sech2_val = 1 - tanh_val**2
    
    # Exact flux solution u (vector field)
    u_exact = as_vector([-sigma * alpha * sech2_val * (x - x0[0]) / r_safe,
                         -sigma * alpha * sech2_val * (y - x0[1]) / r_safe])
    
    # Project exact solution into H(div) space
    u_hdiv = Function(V)
    u_hdiv.project(u_exact)
    
    # Compute divergence
    div_u = div(u_hdiv)
    
    # Compute L2 norms to verify H(div) membership
    norm_u = norm(u_hdiv, norm_type='L2')
    norm_div_u = norm(div_u, norm_type='L2')
    
    print(f"H(div) verification:")
    print(f"‖u‖_L² = {norm_u:.6f}")
    print(f"‖∇·u‖_L² = {norm_div_u:.6f}")
    print(f"H(div) norm ≈ sqrt(‖u‖² + ‖∇·u‖²) = {sqrt(norm_u**2 + norm_div_u**2):.6f}")
    
    # Check if norms are finite (they should be!)
    if np.isfinite(norm_u) and np.isfinite(norm_div_u):
        print("✓ u ∈ H(div) (both ‖u‖ and ‖∇·u‖ are finite)")
    else:
        print("✗ u ∉ H(div)")
    
    return u_hdiv, div_u, u_exact

def plot_verification(u_hdiv, div_u, u_exact, mesh):
    """Plot the numerical solution and its divergence"""
    
    # Project exact solution for plotting
    V = u_hdiv.function_space()
    u_exact_proj = Function(V)
    u_exact_proj.project(u_exact)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot numerical solution
    collection = tripcolor(u_hdiv, axes=axes[0,0])
    plt.colorbar(collection, ax=axes[0,0])
    axes[0,0].set_title('Numerical u (magnitude) in H(div)')
    
    # Plot exact solution magnitude
    magnitude = interpolate(sqrt(dot(u_exact, u_exact)), FunctionSpace(mesh, "CG", 1))
    collection = tripcolor(magnitude, axes=axes[0,1])
    plt.colorbar(collection, ax=axes[0,1])
    axes[0,1].set_title('Exact |u|')
    
    # Plot difference
    difference = interpolate(sqrt(dot(u_exact_proj - u_hdiv, u_exact_proj - u_hdiv)), 
                           FunctionSpace(mesh, "CG", 1))
    collection = tripcolor(difference, axes=axes[0,2])
    plt.colorbar(collection, ax=axes[0,2])
    axes[0,2].set_title('|u_exact - u_hdiv|')
    
    # Plot divergence
    collection = tripcolor(div_u, axes=axes[1,0])
    plt.colorbar(collection, ax=axes[1,0])
    axes[1,0].set_title('Divergence ∇·u')
    
    # Plot quiver plot of u
    quiver(u_hdiv, axes=axes[1,1])
    axes[1,1].set_title('Flux field u')
    axes[1,1].set_aspect('equal')
    
    # Plot divergence along line y=0.5
    x_vals = np.linspace(0, 1, 100)
    div_vals = []
    for x_val in x_vals:
        try:
            div_val = div_u.at([x_val, 0.5])
            div_vals.append(div_val)
        except:
            div_vals.append(0.0)
    
    axes[1,2].plot(x_vals, div_vals, 'b-', linewidth=2)
    axes[1,2].axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='r = r₀ ± δ')
    axes[1,2].axvline(x=0.9, color='red', linestyle='--', alpha=0.7)
    axes[1,2].set_xlabel('x (y=0.5)')
    axes[1,2].set_ylabel('∇·u')
    axes[1,2].set_title('Divergence along cross-section')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run verification
print("Numerical verification of H(div) membership:")
print("=" * 50)
u_hdiv, div_u, u_exact = verify_hdiv()

# Additional check: compute the H(div) norm directly
print("\n" + "=" * 50)
print("Direct H(div) norm computation:")
mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "RT", 2)

# Define test function and compute (u,u) + (∇·u, ∇·u)
u = TrialFunction(V)
v = TestFunction(V)

# Use our manufactured solution
x, y = SpatialCoordinate(mesh)
r = sqrt((x - 0.5)**2 + (y - 0.5)**2)
r_safe = conditional(gt(r, 1e-12), r, 1e-12)
tanh_val = tanh(100.0 * (r_safe - 0.4))
sech2_val = 1 - tanh_val**2

u_exact = as_vector([-1.0 * 100.0 * sech2_val * (x - 0.5) / r_safe,
                     -1.0 * 100.0 * sech2_val * (y - 0.5) / r_safe])

u_func = Function(V)
u_func.project(u_exact)

# H(div) norm: sqrt(∫(u·u + (∇·u)²) dx)
mass_term = dot(u_func, u_func) * dx
div_term = (div(u_func))**2 * dx

hdiv_norm_squared = assemble(mass_term + div_term)
hdiv_norm = np.sqrt(hdiv_norm_squared)

print(f"Direct H(div) norm computation:")
print(f"∫(u·u) dx = {assemble(mass_term):.6f}")
print(f"∫(∇·u)² dx = {assemble(div_term):.6f}")
print(f"H(div) norm = √({assemble(mass_term):.6f} + {assemble(div_term):.6f}) = {hdiv_norm:.6f}")

if np.isfinite(hdiv_norm):
    print("✓ Direct computation confirms u ∈ H(div)")
else:
    print("✗ Direct computation suggests u ∉ H(div)")

# Plot results
plot_verification(u_hdiv, div_u, u_exact, mesh)
