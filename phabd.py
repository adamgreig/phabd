import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

NBITS = 2056
BD = 50.0
SR = 44100.0
F0 = 850.0
F1 = 1150.0

def generate_bitstream(n=NBITS):
    np.random.seed()
    return np.random.random_integers(0, 1, n)

def generate_pure_tones(n, phi=0, f0=F0, f1=F1, sr=SR, bd=BD):
    """
    Generate sinusoids at *f0* and *f1* with phase *phi*, *n* samples at *sr*.
    """
    t = np.linspace(0, 1.0/bd, n).reshape(-1, 1)
    s0 = np.sqrt(2.0) * np.sin(2.0 * np.pi * f0 * t + phi)
    s1 = np.sqrt(2.0) * np.sin(2.0 * np.pi * f1 * t + phi)
    return (s0, s1)

def fsk_samples(bits, phi=0, f0=F0, f1=F1, sr=SR, bd=BD):
    """
    FSK modulate *bits* with parameters. Returns sample vector.

    Currently starts each new bit at zero phase, which is not the same as
    selecting from two clock sources.
    """
    N = sr/bd
    s0, s1 = generate_pure_tones(N, phi, f0, f1, sr, bd)
    output = np.empty(bits.size * N).reshape(-1, 1)
    for idx, bit in enumerate(bits):
        output[idx*N:(idx+1)*N] = s1 if bit else s0
    return output

def add_noise(tx, snr):
    """ Add AGWN to *tx* so it has SNR=*snr*. Returns tx+noise. """
    tx_pwr = np.sum(np.power(2, np.abs(tx)))/tx.size
    noise = np.random.randn(tx.size).reshape(-1, 1)
    noise_pwr = np.sum(np.power(2, np.abs(noise)))/noise.size
    return tx + noise * np.sqrt((tx_pwr / noise_pwr) * np.power(10, -snr/10.0))

def bit_log_likelihood(rx, snr, phi=0, f0=F0, f1=F1, sr=SR, bd=BD):
    """
    Compute the log likelihood of the received baseband signal being 0 or 1,
    assuming that *rx* has *phi* phase and *snr* is the correct noise power.
    """
    # overconstrained on SR and BD, given rx.size.
    # choose to ignore BD.
    N = rx.size
    if abs(bd - sr/N) > (bd/20.0):
        print("W bit_log_likelihood: rx.size {0} not ok".format(rx.size))
    bd = sr/N
    ss = np.power(10.0, -snr / 10.0)
    k = -N/2.0 * np.log(2 * ss * np.pi)
    s0, s1 = generate_pure_tones(rx.size, phi, f0, f1, sr, bd)
    ll0 = k - 1.0/(2*ss) * np.sum(np.power(2, rx-s0))
    ll1 = k - 1.0/(2*ss) * np.sum(np.power(2, rx-s1))
    return (ll0, ll1)

def ll_to_bits(ll0, ll1):
    """Convert log likelihoods to a bitstream."""
    return (ll1 > ll0).astype(np.uint8)

def bits_to_rx(bits, snr, phi=0, f0=F0, f1=F1, sr=SR, bd=BD):
    """Modulate a bitstream to a received bytestream."""
    tx = fsk_samples(bits, np.pi/8, f0, f1, sr, bd)
    return add_noise(tx, snr)

def rx_to_bits(rx, snr, f0=F0, f1=F1, sr=SR, bd=BD):
    """Decode a sample stream to a bitstream."""
    N = SR/BD
    nbits = int(rx.size / N)
    ll0 = np.empty(nbits)
    ll1 = np.empty(nbits)
    for i in range(nbits):
        ll0[i], ll1[i] = bit_log_likelihood(rx[i*N:(i+1)*N], snr)
    return ll_to_bits(ll0, ll1)

def ber_for_snr(snr):
    """Find the BER at SNR=*snr*. Returns BER."""
    nbits = 0
    errs = 0
    while errs < 2 and nbits < 1024*1024:
        print("    snr={0:+0.02f} nbits={1}".format(snr, nbits))
        tx_bits = generate_bitstream(1024)
        rx = bits_to_rx(tx_bits, snr)
        rx_bits = rx_to_bits(rx, snr)
        errs += np.sum(np.abs(tx_bits - rx_bits))
        nbits += 1024
    ber = float(errs) / nbits
    print("*** snr={0:0.02f} ber={1:.2e}".format(snr, ber))
    return ber

if __name__ == "__main__":
    snrs = np.linspace(-15, -5, 11)
    pool = multiprocessing.Pool(processes=8)
    bers = pool.map(ber_for_snr, snrs)
    plt.plot(snrs, bers)
    plt.yscale('log')
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid()
    plt.show()
