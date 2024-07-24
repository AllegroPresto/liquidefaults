
import numpy as np
from scipy.stats import t, norm, multivariate_normal, poisson
import matplotlib.pyplot as plt

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()






if __name__ == "__main__":


    # Parameters
    num_subjects = 1000
    time_horizon_years = 10
    time_horizon_months = time_horizon_years * 12
    prob_default_1yr = 0.05  # Annual default probability

    # Step 1: Transform annual default probability to monthly default rate
    lambda_annual = -np.log(1 - prob_default_1yr)
    lambda_monthly = lambda_annual / 12

    copula = 'gaussian'
    copula = 'clayton'
    copula = 'tstudent'


    if (copula == 'gaussian'):
        # Step 2: Creazione della matrice di correlazione (usiamo una correlazione media)
        rho = 0.1  # Supponiamo una correlazione media del 20%
        Sigma = np.full((num_subjects, num_subjects), rho) + np.eye(num_subjects) * (1 - rho)

        # Step 3: Simulazione della distribuzione normale multivariata
        mv_normal = multivariate_normal(mean=np.zeros(num_subjects), cov=Sigma)
        Z_sim = mv_normal.rvs(size=time_horizon_months)

        # Step 4: Trasformazione inversa nei quantili uniformi
        u_sim = norm.cdf(Z_sim)

        # Step 5: Determinazione del tasso di default mensile per ciascun soggetto
        default_sim = (u_sim <= norm.cdf(lambda_monthly)).astype(int)

    elif (copula == 'clayton'):

        # Step 2: Generate dependent uniform random variables using Clayton copula
        theta = 2.0  # Clayton copula parameter, controls the degree of dependency
        u = np.random.uniform(size=(time_horizon_months, num_subjects))
        v = np.random.uniform(size=(time_horizon_months, num_subjects))

        # Apply the inverse of the Clayton copula
        clayton_copula = (u ** (-theta) - 1 + v ** (-theta)) ** (-1 / theta)

        # Step 3: Transform to uniform quantiles
        default_sim = (clayton_copula <= norm.cdf(lambda_monthly)).astype(int)

    elif (copula == 'tstudent'):

        nu = 5  # Degrees of freedom for the t-Student copula

        # Step 2: Create correlation matrix (assuming average correlation)
        rho = 0.2  # 20% average correlation
        Sigma = np.full((num_subjects, num_subjects), rho) + np.eye(num_subjects) * (1 - rho)

        # Step 3: Simulate multivariate t-distribution
        L = np.linalg.cholesky(Sigma)
        Z = np.random.standard_t(df=nu, size=(time_horizon_months, num_subjects))
        Z = Z @ L.T  # Apply the Cholesky decomposition for the correlation structure

        # Step 4: Transform to uniform quantiles using the t-Student CDF
        u_sim = t.cdf(Z, df=nu)

        # Step 5: Determine monthly default events for each subject
        default_sim = (u_sim <= norm.cdf(lambda_monthly)).astype(int)


    else:

        print('Not available')
        FQ(99)

    # Step 4: Simulate default events using a Poisson process
    monthly_defaults = np.zeros(time_horizon_months)

    # Count defaults using a Poisson process
    for month in range(time_horizon_months):
        for i in range(num_subjects):
            if default_sim[month, i] == 1:
                # Simulate number of defaults for each subject in the month
                monthly_defaults[month] += poisson.rvs(lambda_monthly)

    # Step 5: Compute the times of all defaults
    default_times = np.where(monthly_defaults > 0)[0]

    # Step 6: Compute subsequent default times for each default event
    subsequent_default_times = []
    for i in range(len(default_times)):
        for j in range(i + 1, len(default_times)):
            subsequent_default_times.append(default_times[j] - default_times[i])

    # Step 7: Estimate g(t) - probability distribution of all subsequent default events
    time_bins = np.arange(1, time_horizon_months + 1, 2)
    g_t, _ = np.histogram(subsequent_default_times, bins=time_bins, density=True)

    time_bins_yy = time_bins/12.0
    #print('time_bins_yy: ', time_bins_yy)
    #FQ(99)
    # Step 10: Plot the distribution of subsequent default times (g(t))
    plt.figure(figsize=(12, 6))
    plt.bar(time_bins_yy[:-1], g_t, width=1.0/12, edgecolor='black', align='edge')
    plt.xlabel('Time [years] (since a Default event)')
    plt.ylabel('g(t) Pair Default probability')
    plt.title('g(t) Pair Default probability event distribution')
    plt.grid(True)
    plt.show()

    # Output the number of total defaults simulated
    num_defaults = np.sum(monthly_defaults)
    print(f"Number of defaults simulated over the next 10 years: {num_defaults}")

    time_bins_c = (time_bins_yy[:-1] + time_bins_yy[1:]) / 2.0
    plt.plot(time_bins_c, g_t, '--')
    plt.xlabel('Time [years] (since a Default event)')
    plt.ylabel('g(t) Pair Default probability')
    plt.title('g(t) Pair Default probability event distribution')
    plt.show()



    # Step 7: Visualizzazione della distribuzione temporale dei default
    time_bins = np.arange(0, time_horizon_months + 1, 1)
    #print('time_bins: ', time_bins)
    #FQ(99)

    plt.figure(figsize=(12, 6))
    plt.bar(time_bins[:-1], monthly_defaults, width=1, edgecolor='black', align='edge')
    plt.xlabel('Years')
    plt.ylabel('Number of Default event')
    plt.title('Time distribution of default in the next 10 years num. of Def. %s'%(num_defaults))
    plt.xticks(np.arange(0, time_horizon_months + 1, 12), labels=np.arange(0, time_horizon_years + 1))
    plt.grid(True)
    plt.show()

    # Output del numero totale di default simulati
    num_defaults = np.sum(monthly_defaults)
    print(f"Numero di default simulati nei prossimi 10 years: {num_defaults}")