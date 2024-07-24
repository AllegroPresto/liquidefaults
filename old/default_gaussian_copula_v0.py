
import numpy as np
from scipy.stats import norm, multivariate_normal, poisson
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()






if __name__ == "__main__":


    # Parametri
    num_subjects = 1000
    time_horizon_years = 10
    time_horizon_months = time_horizon_years * 12
    prob_default_1yr = 0.05  # Probabilità di default a 1 anno

    # Step 1: Trasformazione della probabilità di default a 1 anno in probabilità di default mensile
    lambda_annual = -np.log(1 - prob_default_1yr)
    lambda_monthly = lambda_annual / 12


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

    # Step 6: Simulazione del tempo di default con distribuzione Poisson
    # Creiamo un array per il conteggio dei default mensili
    monthly_defaults = np.zeros(time_horizon_months)

    # Contiamo i default mensili usando un processo di Poisson
    for month in range(time_horizon_months):
        for i in range(num_subjects):
            if default_sim[month, i] == 1:
                # Simuliamo il numero di default per ciascun soggetto nel mese
                monthly_defaults[month] += poisson.rvs(lambda_monthly)

    # Step 7: Compute the times of all defaults
    default_times = np.where(monthly_defaults > 0)[0]

    # Step 8: Compute subsequent default times for each default event
    subsequent_default_times = []
    for i in range(len(default_times)):
        for j in range(i + 1, len(default_times)):
            subsequent_default_times.append(default_times[j] - default_times[i])

    # Step 9: Estimate g(t) - probability distribution of all subsequent default events
    time_bins = np.arange(1, time_horizon_months + 1,2)
    g_t, _ = np.histogram(subsequent_default_times, bins=time_bins, density=False)

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