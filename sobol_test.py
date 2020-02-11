import numpy as np
import matplotlib.pylab as plt
import sobol_seq



def primes_from_2_to(n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(dim, n_sample):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample


# Using the above nested radical formula for g=phi_d
# or you could just hard-code it.
# phi(1) = 1.61803398874989484820458683436563
# phi(2) = 1.32471795724474602596090885447809
def phi(d):
    x = 2.0000
    for i in range(10):
        x = pow(1 + x, 1 / (d + 1))
    return x


def plthist(x, y):
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()  # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    #axScatter.set_xlim((-lim, lim))
    #axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()

# Number of dimensions.
d = 2

# number of required points
n = 200

g = phi(d)
alpha = np.zeros(d)
for j in range(d):
    alpha[j] = pow(1 / g, j + 1) % 1
z = np.zeros((n, d))

# This number can be any real number.
# Common default setting is typically seed=0
# But seed = 0.5 is generally better.
seed = 0.5
for i in range(n):
    z[i] = seed + alpha * (i + 1)

z = z % 1
print(z)

s = sobol_seq.i4_sobol_generate(2, n)
h = halton(2, n)
r = np.random.rand(n, 2)



plt.scatter(s[:, 0], s[:, 1])
plt.scatter(h[:, 0], h[:, 1])
plt.scatter(z[:, 0], z[:, 1])
plt.scatter(r[:, 0], r[:, 1])


plt.show()


plthist(s[:, 0], s[:, 1])
plthist(h[:, 0], h[:, 1])
plthist(z[:, 0], z[:, 1])
plthist(r[:, 0], r[:, 1])
