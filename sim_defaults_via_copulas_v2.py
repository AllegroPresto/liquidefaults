import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon, t as student_t
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GaussianCopula, IndependenceCopula
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
    elif (copula == 't-student'):
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

# Funzione per generare campioni utilizzando diverse copule
def generate_copula_samples(copula_type, correlation, n_samples, n_assets, df=5):
    """
    Genera campioni di probabilità utilizzando diverse copule.

    Args:
    - copula_type: Tipo di copula ('gaussian', 't-student', 'clayton', 'frank', 'independence').
    - correlation: Coefficiente di correlazione (rho per Gaussiana, tau per Clayton e Frank).
    - n_samples: Numero di campioni da generare.
    - n_assets: Numero di asset nel portafoglio.
    - df: Gradi di libertà per la copula t-Student (valido solo se `copula_type='t-student'`).

    Returns:
    - copula_samples: Campioni di probabilità generati.
    """
    if copula_type == 'gaussian':
        # Copula Gaussiana
        cov_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(cov_matrix, 1)
        copula = GaussianCopula(corr=cov_matrix)
        copula_samples = copula.rvs(n_samples)

    elif copula_type == 't-student':
        # Copula t-Student multivariata
        cov_matrix = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(cov_matrix, 1)

        # Step 1: Genera campioni dalla distribuzione t-Student multivariata
        t_samples = student_t.rvs(df, size=(n_samples, n_assets))  # Generazione di n_samples x n_assets campioni t-Student
        # Step 2: Moltiplica per la matrice di correlazione (Cholesky Decomposition)
        t_samples = np.dot(t_samples, np.linalg.cholesky(cov_matrix).T)

        # Step 3: Trasforma i campioni t-Student in uniformi usando la CDF della distribuzione t-Student
        copula_samples = student_t.cdf(t_samples, df=df)  # Campioni uniformi su [0, 1] con dipendenza t-Student

    elif copula_type == 'clayton':
        copula = ClaytonCopula(correlation)
        copula_samples = copula.rvs(n_samples)

    elif copula_type == 'frank':
        copula = FrankCopula(correlation)
        copula_samples = copula.rvs(n_samples)


    elif copula_type == 'independence':

        n_samples = n_assets
        print('n_samples: ', n_samples)
        # Copula di indipendenza
        copula = IndependenceCopula()
        copula_samples = copula.rvs(n_samples)

        print('copula_samples: ', len(copula_samples))


        FQ(999999)


    else:
        raise ValueError(
            f"Copula sconosciuta: {copula_type}. Usa 'gaussian', 't-student', 'clayton', 'frank', o 'independence'.")


    # Se la copula è bivariata e il numero di asset è maggiore di 2, replicare le colonne
    if copula_type in ['clayton', 'frank']:
        copula_samples = np.tile(copula_samples, (1, n_assets // 2 + 1))[:, :n_assets]

    return copula_samples


def compute_gt(default_times, tot_n_credits):

    w_data = []
    subsequent_default_times = []
    n_sample = len(default_times)
    for i in range(n_sample):
        for j in range(i + 1, len(default_times)):
            subsequent_default_times.append(default_times[j] - default_times[i])
            w_ = (n_sample - (j - i))
            #w2_= num_credits[i]
            #w_data.append(1.0/w_/w2_)
            w_data.append(1.0/w_)

    return  subsequent_default_times, w_data


# Funzione per generare tempi di default con una copula specificata e margini specificati
def generate_default_times(n_assets, n_samples, horizon, correlation, marginal_type='uniform', copula_type='gaussian',df=5):
    """
    Genera i tempi di default per un portafoglio di asset con margini specificati e
    correlazione determinata da una copula specificata.

    Args:
    - n_assets: Numero di asset nel portafoglio.
    - n_samples: Numero di campioni da generare.
    - horizon: Orizzonte temporale per la trasformazione (valore massimo della distribuzione marginale).
    - correlation: Coefficiente di correlazione tra gli asset (valore tra -1 e 1).
    - marginal_type: Tipo di distribuzione marginale ('uniform' o 'exponential').
    - copula_type: Tipo di copula ('gaussian', 't-student', 'clayton', 'frank', 'independence').
    - df: Gradi di libertà per la copula t-Student.

    Returns:
    - default_times: Array di tempi di default generati (shape: [n_samples, n_assets]).
    """
    # Step 1: Generazione dei campioni della copula
    copula_samples = generate_copula_samples(copula_type, correlation, n_samples, n_assets, df)

    # Step 2: Calcolo dei tempi di default basati sul tipo di marginale
    if marginal_type == 'uniform':
        # Distribuzione marginale uniforme su [0, horizon]
        default_times = uniform.ppf(copula_samples) * horizon
    elif marginal_type == 'exponential':
        # Distribuzione marginale esponenziale con intensità lambda = 1/horizon
        default_times = expon.ppf(copula_samples, scale=horizon)
    else:
        raise ValueError(f"Margine sconosciuto: {marginal_type}. Usa 'uniform' o 'exponential'.")


    return default_times

def extract_defaut_time_diff(default_times, tot_n_credits):

    # Compute g(t)
    w_data = []
    subsequent_default_times = []
    n_sample = len(default_times)
    for i in range(n_sample):
        for j in range(i + 1, len(default_times)):
            subsequent_default_times.append(default_times[j] - default_times[i])
            w_ = (n_sample - (j - i))
            #w2_= num_credits[i]
            #w_data.append(1.0/w_/w2_)
            w_data.append(1.0/w_)

    return  subsequent_default_times, w_data

if __name__ == "__main__":


    # Parametri del portafoglio
    n_assets = 500# Numero di asset nel portafoglio
    n_samples = 1  # Numero di campioni da generare
    n_sampling = 100
    horizon = 10  # Orizzonte temporale in anni (massimo tempo di default per Uniforme o scala per Esponenziale)
    data_save = False
    #data_save = True


    # Tipo di marginale da utilizzare ('uniform' o 'exponential')
    #marginal_type = 'exponential'  # Cambia tra 'uniform' e 'exponential'
    marginal_type = 'uniform'  # Cambia tra 'uniform' e 'exponential'

    # Tipo di copula da utilizzare ('gaussian', 't-student', 'clayton', 'frank', 'independence')
    #copula_type = 't-student'  # Cambia tra 'gaussian', 't-student', 'clayton', 'frank', 'independence'
    #copula_type = 'clayton'  # Cambia tra 'gaussian', 't-student', 'clayton', 'frank', 'independence'
    #copula_type = 'frank'  #
    copula_type = 'gaussian'
    #copula_type = 'independence'

    correlation = 0.2  # Correlazione tra gli asset

    theta_val = +20.0  # +5, -5, +2, -2, +0.5, -0.5
    theta_val = +1.  # +5, -5, +2, -2, +0.5, -0.5
    df = 4  # Gradi di libertà per la copula t-Student

    rho_val = correlation
    nu_val = df

    params = {'gaussian': {'rho': rho_val}, 'clayton': {'theta': theta_val},
              't-student': {'nu': nu_val, 'rho': rho_val}, 'frank': {'theta': theta_val}}


    # Generazione dei tempi di default
    tot_n_credits = n_assets
    time_horizon_months = horizon*12.0
    default_times = []
    common_bins = np.linspace(0, horizon,int(time_horizon_months))


    cumulative_g_t = np.zeros(len(common_bins) - 1)
    cumulative_def_t = []
    for i in range(0, n_sampling):

        default_times = generate_default_times(n_assets, n_samples, horizon, correlation, marginal_type, copula_type, df)
        all_default_times = default_times.flatten()  # Appiattisce la matrice in un vettore
        all_default_times.sort()

        sub_def_times, w_data = extract_defaut_time_diff(all_default_times, tot_n_credits)

        g_t, bin_edges = np.histogram(sub_def_times, bins=common_bins, weights=w_data, density=True)

        print('n. sample: ', i)
        cumulative_g_t += g_t
        cumulative_def_t = np.append(cumulative_def_t, all_default_times)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    par1, par2 = get_par(params, copula_type)
    print('par1: ', par1)

    if (par2 == None):
        file_out = r'gt_via_copulas/gt_%s_%s_par1_%s.csv' % (copula_type, marginal_type, par1)
        plot_file = r'density_%s_%s_par1_%s.png' % (copula_type, marginal_type, par1)

    else:
        file_out = r'gt_via_copulas/gt_%s_%s_par1_%s_par2_%s.csv' % (copula_type, marginal_type, par1, par2)
        plot_file = r'density_%s_%s_par1_%s_par2_%s.png' % (copula_type, marginal_type, par1, par2)



    plt.figure(figsize=(12, 6))
    width_histo = (bin_edges[1] - bin_edges[0])
    plt.bar(bin_centers, cumulative_g_t, width=width_histo, edgecolor='black', align='edge')

    plt.xlabel('Time [mnth] (since a Default event)')
    plt.ylabel('g(t) Pair Default probability')
    plt.title('g(t), N. Credits %s, rho: %s' % (str(tot_n_credits), str(correlation)))
    plt.grid(True)
    plt.show()

    if (data_save):

        g_t_df = pd.DataFrame({'Time (Months)': bin_centers, 'Probability': cumulative_g_t})
        g_t_df.to_csv(file_out, index=False)




    # Visualizzazione: Istogramma aggregato dei tempi di default
    plt.figure(figsize=(10, 6))
    plt.hist(cumulative_def_t, bins='auto', alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title(f"Distribuzione Tempi di Default ({copula_type.capitalize()} Copula - Marginale: {marginal_type.capitalize()} - rho: {rho_val})")
    #plt.title(f"Distribuzione Tempi di Default ({copula_type.capitalize()} Copula - Marginale: {marginal_type.capitalize()} - nu: {theta_val})")

    plt.xlabel('Tempo di Default [months]')
    plt.ylabel('Frequenza')
    plt.grid(axis='y')

    if (data_save):
        print('plot_file: ', plot_file)
        plt.savefig('graph\%s' % (plot_file))

    plt.show()




