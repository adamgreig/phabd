import numpy as np
import matplotlib.pyplot as plt

def add_noise(tx, snr):
    """ Add AGWN to *tx* so it has SNR=*snr*. Returns tx+noise. """

    tx_pwr = np.sum(np.abs(tx)**2)/tx.size
    noise = np.random.randn(tx.size)
    noise_pwr = np.sum(np.abs(noise)**2)/noise.size
    return tx + noise * np.sqrt((tx_pwr / noise_pwr) * (10.0 ** (-snr / 10.0)))

if __name__ == "__main__":
    N = 128
    t = np.arange(N)
    tx = np.sqrt(2.0) * np.sin(2.0*np.pi*t*(5.0/N))
    rx = add_noise(tx, 0.0)
    plt.plot(tx, color='g')
    plt.plot(rx, color='r')
    plt.show()
