import numpy as np


class StoneBSS:
    def __init__(self, long_h=9e5, short_h=1, max_mask_len=500, num_halflives=8):
        """Applies Stone's Blind Source Separation aglorithm.

        Usage (assuming x is an array of mixed signals):
        >>> from stone_bss import StoneBSS
        >>> bss = StoneBSS(
            long_h=9e5, short_h=1, max_mask_len=500, num_halflives=8)
        >>> bss.fit(x)
        >>> y = bss(x)

        References:
        Stone, James. (2001). Blind Source Separation Using Temporal Predictability. Neural computation. 13. 1559-74. 10.1162/089976601750265009. 

        Args:
            long_h (int): half-life of the long-term exponentially-weighted moving average
            short_h (int): half-life of the short-term exponentially-weighted moving average
            max_mask_len (int): length of the mask used to compute the EWMAs
            num_halflives (int): number of half-lives to use in the mask
        """

        # EWMA properties
        self.long_h = long_h
        self.short_h = short_h
        self.max_mask_len = max_mask_len
        self.num_halflives = num_halflives

        # Construct masks
        self._get_masks()

        # Initialize W
        W = None

    def _get_masks(self):
        """Computes the EWMA masks

        Returns:
            tuple[np.ndarray, np.ndarray]: short-term mask, long-term mask
        """
        # short mask
        ts = self.num_halflives * self.short_h
        lam_s = 2 ** (-1 / self.short_h)
        tau_s = np.arange(0, ts)
        s_mask = lam_s ** tau_s
        s_mask[0] = 0
        s_mask /= np.abs(s_mask).sum()
        s_mask[0] = -1

        # long mask
        tl = self.num_halflives * self.long_h
        tl = np.clip(tl, 1, self.max_mask_len)
        lam_l = 2 ** (-1 / self.long_h)
        tau_l = np.arange(0, tl)
        l_mask = lam_l ** tau_l
        l_mask[0] = 0
        l_mask /= np.abs(l_mask).sum()
        l_mask[0] = -1

        self.s_mask = s_mask
        self.l_mask = l_mask
        return s_mask, l_mask

    def _conv_rows(self, x, mask):
        """Performs 1D convolutions over the rows of `x` with kernel `mask`. Truncates output to same length as `x`. For FIR filters, has same behavior as MATLAB's `filter(mask, 1, x)`

        Args:
            x (np.ndarray): input array
            mask (np.ndarray): convolution kernel

        Returns:
            np.ndarray: result of applying the FIR filter `mask` to `x`
        """
        ewma = np.convolve(mask, x[:, 0]).reshape(-1, 1)
        for i in range(1, x.shape[1]):
            ewma_i = np.convolve(mask, x[:, i]).reshape(-1, 1)
            ewma = np.hstack((ewma, ewma_i))
        return ewma[:len(x)]

    def fit(self, x):
        """Computes the unmixing matrix W for the input mixed signals. To obtain umixed signals `y`, perform the operation

        `y = x @ W`

        Args:
            x (np.array): mixed signals

        Returns:
            np.array: unmixing matrix W
        """
        S = self._conv_rows(x, self.s_mask)
        L = self._conv_rows(x, self.l_mask)
        cov_s = S.T @ S
        cov_l = L.T @ L

        _, self.W = np.linalg.eig(
            np.linalg.solve(cov_s, cov_l))
        return self.W

    def __call__(self, x):
        assert self.W is not None
        assert x.shape[1] == self.W.shape[0]

        return x @ self.W


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # example source signals
    t = np.linspace(0, 10, 1000).reshape(1, -1)
    s1 = np.sin(2 * np.pi * t)
    s2 = np.cos(10 * np.pi * t)
    s3 = np.sin(t) * np.exp(-t**2)
    s = np.concatenate((s1, s2, s3), axis=0).T

    # scale to zero mean, unit variance
    s = (s - s.mean(axis=0)) / s.std(axis=0)

    # mix
    A = np.random.randn(3, 3)
    x = s @ A

    # instantiate source separator
    bss = StoneBSS(
        long_h=1000,
        short_h=1,
        max_mask_len=500,
        num_halflives=8
    )
    bss.fit(x)
    y = bss(x)

    # rescale
    y -= np.mean(y, axis=0)
    y /= np.std(y, axis=0)

    # plot
    plt.figure()
    for i in range(1, 4):
        plt.subplot(3, 3, 3*i - 2)
        plt.plot(x[:, i - 1])

        plt.subplot(3, 3, 3*i - 1)
        plt.plot(s[:, i - 1])

        plt.subplot(3, 3, 3*i)
        plt.plot(y[:, i - 1])

    plt.subplot(3, 3, 1), plt.title("Mixed")
    plt.subplot(3, 3, 2), plt.title("Sources")
    plt.subplot(3, 3, 3), plt.title("Unmixed")
    plt.show()
