
import numpy as np
from scipy.stats import t, norm, multivariate_normal, poisson
import matplotlib.pyplot as plt
from copulas.bivariate import Frank
import pandas as pd

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def get_par(params, copula):

    if (copula == 'gaussian'):
        par1 = params[copula]['rho']
        par2 = None
    elif (copula == 'clayton'):
        par1 = params[copula]['theta']
        par2 = None
    elif (copula == 'tstudent'):
        par1 = params[copula]['nu']
        par2 = params[copula]['rho']
    elif (copula == 'frank'):
        par1 = params[copula]['theta']
        par2 = None
    else:
        print('Tipologia di copula non trovata')
        FQ(77)
        # Generate joint samples using the Frank copula

    return par1, par2



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
    #copula = 'clayton'
    #copula = 'tstudent'
    #copula = 'frank'

    rho_val = 0.1
    theta_val = 2.0
    nu_val = 5
    rhot_val = 0.2

    gt_save = True
    #gt_save = False

    params = {'gaussian': {'rho': rho_val}, 'clayton': {'theta': theta_val},
              'tstudent': {'nu': nu_val, 'rho': rhot_val}, 'frank': {'theta': theta_val}}

    if (copula == 'gaussian'):
        # Step 2: Creazione della matrice di correlazione (usiamo una correlazione media)

        rho = params[copula]['rho']
        #rho = 0.1  # Supponiamo una correlazione media del 20%
        Sigma = np.full((num_subjects, num_subjects), rho) + np.eye(num_subjects) * (1 - rho)

        # Step 3: Simulazione della distribuzione normale multivariata
        mv_normal = multivariate_normal(mean=np.zeros(num_subjects), cov=Sigma)
        Z_sim = mv_normal.rvs(size=time_horizon_months)

        # Step 4: Trasformazione inversa nei quantili uniformi
        u_sim = norm.cdf(Z_sim)

        # Step 5: Determinazione del tasso di default mensile per ciascun soggetto
        default_sim = (u_sim <= norm.cdf(lambda_monthly)).astype(int)

    elif (copula == 'clayton'):

        theta = params[copula]['theta']
        # Step 2: Generate dependent uniform random variables using Clayton copula
        u = np.random.uniform(size=(time_horizon_months, num_subjects))
        v = np.random.uniform(size=(time_horizon_months, num_subjects))

        # Apply the inverse of the Clayton copula
        clayton_copula = (u ** (-theta) - 1 + v ** (-theta)) ** (-1 / theta)

        # Step 3: Transform to uniform quantiles
        default_sim = (clayton_copula <= norm.cdf(lambda_monthly)).astype(int)

    elif (copula == 'tstudent'):

        nu = params[copula]['nu']
        rho_t = params[copula]['rho']

        # Step 2: Create correlation matrix (assuming average correlation)
        #rho_t = 0.2  # 20% average correlation
        Sigma = np.full((num_subjects, num_subjects), rho_t) + np.eye(num_subjects) * (1 - rho_t)

        # Step 3: Simulate multivariate t-distribution
        L = np.linalg.cholesky(Sigma)
        Z = np.random.standard_t(df=nu, size=(time_horizon_months, num_subjects))
        Z = Z @ L.T  # Apply the Cholesky decomposition for the correlation structure

        # Step 4: Transform to uniform quantiles using the t-Student CDF
        u_sim = t.cdf(Z, df=nu)

        # Step 5: Determine monthly default events for each subject
        default_sim = (u_sim <= norm.cdf(lambda_monthly)).astype(int)

    elif (copula == 'frank'):

        theta = params[copula]['theta']
        frank_copula = Frank()
        data = np.random.uniform(size=(num_subjects, 2))
        frank_copula.fit(data)
        frank_copula.theta = theta  # Set the copula parameter directly

        # Generate dependent samples
        u_sim = np.random.uniform(size=(time_horizon_months, num_subjects))
        v_sim = np.zeros_like(u_sim)

        # Generate joint samples using the Frank copula
        for i in range(time_horizon_months):
            v_sim[i] = frank_copula.sample(num_subjects)[:, 1]

        # Step 3: Transform to uniform quantiles
        default_sim = (v_sim <= norm.cdf(lambda_monthly)).astype(int)

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
    w_data = []
    subsequent_default_times = []
    n_sample = len(default_times)
    for i in range(n_sample):
        for j in range(i + 1, len(default_times)):
            subsequent_default_times.append(default_times[j] - default_times[i])
            w_ = (n_sample - (j - i))
            w_data.append(1.0/w_)

    # Step 7: Estimate g(t) - probability distribution of all subsequent default events
    time_bins = np.arange(1, time_horizon_months + 1, 3)
    g_t, _ = np.histogram(subsequent_default_times, bins=time_bins, weights = w_data, density=True)
    time_bins_yy = time_bins/12.0

    if (gt_save):
        par1, par2 = get_par(params, copula)
        if (par2 == None):
            file_out = r'gt_via_copulas/gt_%s_par1_%s.csv' % (copula, par1)
        else:
            file_out = r'gt_via_copulas/gt_%s_par1_%s_par2_%s.csv' % (copula, par1, par2)

        g_t_df = pd.DataFrame({'Time (Months)': time_bins[:-1], 'Probability': g_t})
        g_t_df.to_csv(file_out, index=False)

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
    plt.title('Time distribution of default in the next 10 years num. of def. %s'%(int(num_defaults)))
    plt.xticks(np.arange(0, time_horizon_months + 1, 12), labels=np.arange(0, time_horizon_years + 1))
    plt.grid(True)
    plt.show()

    # Output del numero totale di default simulati
    num_defaults = np.sum(monthly_defaults)
    print(f"Numero di default simulati nei prossimi 10 years: {num_defaults}")