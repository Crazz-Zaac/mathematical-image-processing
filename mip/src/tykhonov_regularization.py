import numpy as np

def grad(I: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of an image I.
    --------
    Parameters:
    I: np.ndarray
    --------
    Returns:
    G: np.ndarray
    """
    M, N = I.shape
    G = np.zeros((M, N, 2))
    G[:, 1:, 0] = I[:, 1:] - I[:, :-1]
    G[:, 0, 0] = I[:, 0]
    G[1:, :, 1] = I[1:, :] - I[:-1, :]
    G[0, :, 1] = I[0, :]
    return G


def div(G: np.ndarray) -> np.ndarray:
    """
    Computes the divergence of the gradient.
    --------
    Parameters:
    G: np.ndarray
    --------
    Returns:
    p: np.ndarray
    """
    M, N, _ = G.shape
    divergence = np.zeros((M, N))
    divergence[:, :-1] += G[:, :-1, 0]
    divergence[:, 1:] -= G[:, :-1, 0]
    divergence[:, 0] = -G[:, 0, 0]
    divergence[:-1, :] += G[:-1, :, 1]
    divergence[1:, :] -= G[:-1, :, 1]
    divergence[0, :] = -G[0, :, 1]
    return divergence


def laplacian(I: np.ndarray) -> np.ndarray:
    """
    Computes the laplacian of an image I.
    --------
    Parameters:
    I: np.ndarray
    --------
    Returns:
    lap: np.ndarray
    """
    lap = div(grad(I))
    return lap


# laplacian(np.random.randint(0, 255, (100, 100)))


def tykohonov_gradient(
    I: np.ndarray, lambda_: float, noisy_image: np.ndarray = None, tau: float = 0.1
) -> np.ndarray:
    """
    Computes the Tykohonov gradient of an image I.
    --------
    Parameters:
    I: np.ndarray
    lambda_: float
    noisy_image: np.ndarray
    tau: float
    --------
    Returns:
    tyk: np.ndarray
    """
    if noisy_image is None:
        noisy_image = I.copy()

    J_prime = I - noisy_image - lambda_ * laplacian(I)
    tyk = I - tau * J_prime
    return tyk


if __name__ == "__main__":
    from mip.display.image import read, subplot_images

    img = read(r"../assets/cameraman_sp.png", "GRAY")
    images = []
    timg = img.copy()
    taus = [0.01, 0.1, 1]
    lambdas = [0.01, 0.1, 1]

    titles = []
    iterations = 120
    every = 120
    for tau in taus:
        for lambda_ in lambdas:
            for i in range(iterations + 1):
                timg = tykohonov_gradient(timg, lambda_, img, tau)
                if i > 0 and i % every == 0:
                    images.append(timg)
                    titles.append(f"λ={lambda_}, τ={tau}, iter={i}")

    subplot_images(
        images, titles=titles, fig_size=(20, 18), order=(-1, 3), dpi=100
    ).savefig("tykohonov.png")
