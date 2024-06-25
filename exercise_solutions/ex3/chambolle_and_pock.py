import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import Tuple
from mip.display.image import show, read, subplot_images

def grad(I: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of an image I.
    --------
    Parameters:
    I: np.ndarray
    --------
    Returns:
    grad_x: np.ndarray    
    """
    M, N = I.shape
    G = np.zeros((M, N, 2))
    G[:, 1:, 0] = I[:, 1:] - I[:, :-1]
    G[:, 0, 0] = I[:, 0]
    G[1:, :, 1] = I[1:, :] - I[:-1, :]
    G[0, :, 1] = I[0, :]
    return G


def divergence(G: np.ndarray) -> np.ndarray:
    """
    Computes the divergence of the gradient.
    --------
    Parameters:
    grad_x: np.ndarray
    grad_y: np.ndarray
    --------
    Returns:
    div: np.ndarray    
    """
    M, N, _ = G.shape
    p = np.zeros((M, N))
    p[:, :-1] += G[:, :-1, 0]
    p[:, 1:] -= G[:, :-1, 0]
    p[:, 0] = -G[:, 0, 0]
    p[:-1, :] += G[:-1, :, 1]
    p[1:, :] -= G[:-1, :, 1]
    p[0, :] = -G[0, :, 1]
    return p

def compute_energy(u, f, lambd):
    """ Compute the energy functional J(u). """
    G = grad(u)
    tv_norm = np.sqrt(G[:, :, 0]**2 + G[:, :, 1]**2).sum()
    data_fidelity = 0.5 * np.linalg.norm(u - f)**2
    return data_fidelity + lambd * tv_norm


def ROF_primal_dual(I, lambd, tau):
    """Perform the primal-dual minimization algorithm for ROF denoising."""
    sigma = 0.02
    theta = 1
    m, n = I.shape
    u = I.copy()
    p = np.zeros((m, n, 2))
    u_bar = u.copy()

    # Dual update
    grad_u_bar = grad(u_bar)
    p = (p + sigma * grad_u_bar) / (
        1 + sigma * np.sqrt(np.sum(grad_u_bar**2, axis=2, keepdims=True))
    )

    # Primal update
    div_p = divergence(p)
    u_old = u
    u = (u + tau * div_p + tau * lambd * I) / (1 + tau * lambd)

    # Relaxation
    u_bar = u + theta * (u - u_old)

    # Compute and store energy
    energy = compute_energy(u_bar, I, lambd)

    return u, energy


if __name__ == '__main__':
    img = read("../assets/cameraman_sp.png", "GRAY")
    images = []
    taus = [0.01, 0.1, 1]
    lambdas = [0.01, 0.1, 1]

    titles = []
    iterations = 120
    every = 120

    energy_plot_data = []

    for tau in taus:
        for lambda_ in lambdas:
            timg = img.copy()
            energies = []
            for i in range(iterations + 1):
                timg, energy = ROF_primal_dual(timg, lambda_, tau)
                energies.append(energy)
                if i > 0 and i % every == 0:
                    images.append(timg)
                    titles.append(f"λ={lambda_}, τ={tau}, iter={i}")
            energy_plot_data.append((lambda_, tau, energies))

    # Plot images
    subplot_images(
        images, titles=titles, fig_size=(20, 18), order=(-1, 3), dpi=100
    ).savefig("primal_dual.png")

    # Plot energy
    plt.figure(figsize=(10, 6))
    for lambda_, tau, energies in energy_plot_data:
        plt.plot(energies, label=f"λ={lambda_}, τ={tau}")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Energy over Iterations for Different λ and τ")
    plt.savefig("energy_plot_chambolle.png")
    plt.show()