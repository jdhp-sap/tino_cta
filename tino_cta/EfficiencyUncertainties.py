import numpy as np
from numpy import log, exp, pi

from scipy.optimize import minimize
from scipy.special import loggamma

from itertools import count

''' method from the paper
Calculating Efficiencies and Their
Uncertainties
Marc Paterno
FNAL/CD/CEPA/SLD
paterno@fnal.gov
May 5, 2003
'''





__all__ = ["get_efficiency_uncertainties"]


def log_gamma(end, start=1):
    '''
        calculating log(Γ(x)) as the sum of the logarithms
        of all integers in [1, x)
        a second argument can be used to set an alternative start
        e.g. if you have Γ(N) / Γ(k) -> Γ(N,k) sums all in [k,N)

        Parameters
        ----------
        end : integer
            (not included) end of the sum
        start : integer, default: 1
            start of the sum

        Notes
        -----
        Γ(x) = (x-1)!
        so `end` doesn't neet to be included in sum

        if `start` > `end`, return the negative instead
    '''

    return np.sign(end-start) * \
        np.sum(np.log(np.arange(min(start, end), max(start, end))))


def P_e(k, N, e):
    '''
    probability density function for the efficiency `e` of passing `k` out of `N` events

    Parameters
    ----------
    k: integer
        number of events passing selection
    N : integer
        total number of events
    e : float
        test-efficiency
        0 <= e <= 1

    Returns
    -------
    P : float
        the probability density to have a real efficiency of e
        while observing k out of N events passing a selection
    '''

    # has problems with high numbers ( Γ(100) gets HUUUUGE), use log-version instead
    # res1 = math.gamma(N+2) / (math.gamma(k+1)*math.gamma(N-k+1)) * e**k * (1-e)**(N-k)
    # return res1

    # protect from log(0) calls
    if e == 0:
        if k == 0:
            return 6
        else:
            return 0
    if e == 1:
        if k == N:
            return 6
        else:
            return 0
    if abs(e-.5) > .5:
        return 0

    # log-ed version
    #res2 = loggamma(N+2) + k*log(e) + (N-k)*np.log(1-e) - loggamma(N-k+1) - loggamma(k+1)
    #return exp(res2)

    # optimised version (sum_{0..N} - sum_{0..k} = sum_{k..N})
    res3 = log_gamma(N+2, k+1) + k*log(e) + (N-k)*log(1-e) - log_gamma(N-k+1)
    return exp(res3)


def get_b_from_a(a, k, N, conf=.68, de=.001, func=P_e):
    '''
    determines b from `a` so that the intervall between them covers an integral of `conf`

    integrates `func` starting from `a` in steps of `de` by trapezoid method and stops
    when either the integral is at least 0.68 or the right edge has been reached (b>1)

    Parameters
    ----------
    a : float
        starting point of integration
    k, N : integers
        passed / total number of events for efficiency calculation
    conf : float, optional (default: 0.68)
        confidence intervall
    de : float, optional (default: 0.001)
        stepsize in numerical integration
    func : callable, optional (default: `P_e`)
        function to be integrated

    Returns
    -------
    b : float
        right edge of the confidence intervall
    integral : float
        integral between `a` and `b` for cross-check -- should be close `conf`; might be
        smaller if right edge reaches 1
    '''
    b = a
    integral = 0
    for i in count():
        integral += (func(k, N, b) + func(k, N, b+de))*de/2.
        b = a + i*de
        if integral >= conf:
            return b
        if b+de > 1:
            return 9


def test_func(arg, k, N, conf=.68, de=.001, func=P_e):
    '''
    test function that calculates the
    difference between a and b while covering an intervall of `conf`

    Parameters
    ----------
    arg : array containing a float
        starting point of integration
    k, N : integers
        passed / total number of events for efficiency calculation
    conf : float, optional (default: 0.68)
        confidence intervall coverage
    de : float, optional (default: 0.001)
        stepsize in numerical integration
    func : callable, optional (default: `P_e`)
        function to be integrated
    '''

    a = arg[0]
    b = get_b_from_a(a, k, N, conf, de, func)
    if b <= 1:
        test_func.min_a = a
        test_func.min_b = b
        return b-a
    else:
        return b


def get_efficiency_uncertainties_minimize(k, N, conf=.68, de=0.0005, func=P_e):
    """
    calculates the mean and confidence intervall of the selection efficiency from `k`
    selected out of `N` total events by minimising the length of the confidence intervall
    while covering an integral of `conf`

    Parameters
    ----------
    k, N : integers or numpy arrays
        passed / total number of events for efficiency calculation
    conf : float, optional (default: 0.68)
        confidence intervall coverage
    de : float, optional (default: 0.0005)
        stepsize in numerical integration
    func : callable, optional (default: `P_e`)
        function to be integrated

    Returns
    -------
    results : shape (x,5) numpy array
        where x is the dimension of k;
        contains the result of the efficiency calculation for every dimension of k and N;
        what follows now is the content of `results` -- no additional return values
    mean : float
        the mean efficiency -- i.e. `k/N`
    lerr, herr : floats
        the lower and upper distance of the edges of the confidence intervall from
        the mean
    min_a, min_b : floats
        the lower and upper values of the edges of the confidence intervall
        Note:
            lerr = mean - min_a
            herr = min_b - mean
    """

    np_k = np.array(k, ndmin=1)
    np_N = np.array(N, ndmin=1)

    assert np_k.shape == np_N.shape, "k and N need to be of same dimension"

    results = np.zeros((len(np_k), 5))

    for i, (_k, _N) in enumerate(zip(np_k, np_N)):

        if _N == 0:
            results[i] = [0, 0, 0, 0, 0]
            continue

        minimize(test_func, [0], args=(_k, _N, conf, de, func), bounds=[(0, 1)],
                    method='L-BFGS-B', options={'disp': False, 'eps': 1e-3})

        mean = _k/_N

        # if no (all) events passed, the lower (upper) error is zero (one)
        if _k == 0.: test_func.min_a = 0
        if _k == _N: test_func.min_b = 1
        lerr = mean-test_func.min_a
        herr = test_func.min_b-mean

        # not sure if it's more useful to return errors as distance from mean or as
        # position on the axis... so do both
        results[i] = [mean, lerr, herr, test_func.min_a, test_func.min_b]

    return results


def get_efficiency_uncertainties_scan(k, N, conf=.68, de=0.0005, func=P_e):
    """
    same as `get_efficiency_uncertainties_minimize` just through scanning the values of a
    instead of minimisation -- not recommended, use the minimiser
    """
    if N == 0:
        return 0, 0, 0, 0, 0

    min_diff = 20.
    min_a = None
    min_b = None
    for i in count(0):
        a = i * de
        (b, integral) = get_b_from_a(a, k, N, conf, de, func)
        if b-a < min_diff:
            min_diff = b-a
            min_a = a
            min_b = b
        if b > 1:
            break

    mean = k/N
    if k == 0: min_a = 0
    if k == N: min_b = 1
    lerr = mean-min_a
    herr = min_b-mean

    return mean, lerr, herr, min_a, min_b


get_efficiency_uncertainties = get_efficiency_uncertainties_minimize


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        pars = [2, 5]
    else:
        pars = [int(sys.argv[1]), int(sys.argv[2])]

    from astropy.stats.funcs import binom_conf_interval as uncert

    print("get_efficiency_uncertainties_minimize({}, {}):".format(*pars))
    print(get_efficiency_uncertainties_minimize(*pars))

    print()
    print("astropy.stats.funcs.binom_conf_interval:")
    print(uncert(*pars))

    import numpy as np
    import matplotlib.pyplot as plt

    print()
    print("creating test distributions for N=5 and k = {0,...,5}")
    dt = 0.005
    t = np.arange(0.0, 1.+dt, dt)
    y0 = [P_e(0, 5, x) for x in t]
    y1 = [P_e(1, 5, x) for x in t]
    y2 = [P_e(2, 5, x) for x in t]
    y3 = [P_e(3, 5, x) for x in t]
    y4 = [P_e(4, 5, x) for x in t]
    y5 = [P_e(5, 5, x) for x in t]

    print("integral of test distributions (should be 1)")
    print("k=0: ", sum(y0) * dt)
    print("k=1: ", sum(y1) * dt)
    print("k=2: ", sum(y2) * dt)
    print("k=3: ", sum(y3) * dt)
    print("k=4: ", sum(y4) * dt)
    print("k=5: ", sum(y5) * dt)

    plt.figure(1)
    plt.plot(t, y0, 'bo', t, y1, 'ro', t, y2, 'yo', t, y3, 'go', t, y4, "bo", t, y5, "ro")
    plt.show()


from astropy.stats.funcs import binom_conf_interval as uncert
def test_astropy(pars=(500, 1000)):
    uncert(*pars)


def test_own(pars=(500, 1000)):
    get_efficiency_uncertainties_minimize(*pars)
