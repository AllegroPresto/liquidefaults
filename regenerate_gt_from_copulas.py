import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon, t as student_t
from scipy.interpolate import interp1d

from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GaussianCopula, IndependenceCopula
import pandas as pd

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def extract_prob_g_r(data, perc_n):

    data = data.sort_values(by='TimeToDefault').reset_index(drop=True)
    time_to_default = data['TimeToDefault'].values
    num_credits = data['NumberOfCredits'].values
    N_sample = len(time_to_default)
    n_bins_ref_ = np.minimum(int(2.0 * np.sqrt(N_sample)), 30)


    time_diff_matrix = (time_to_default[None, :] - time_to_default[:, None]) / 365.0

    # Filtra solo le differenze superiori a zero (considera solo tempi successivi)
    i_upper, j_upper = np.triu_indices(N_sample, k=1)
    subsequent_default_times = time_diff_matrix[i_upper, j_upper]

    # Calcolo dei pesi vettorialmente
    w_ = N_sample - (j_upper - i_upper)
    w2_ = num_credits[i_upper]  # Pesi dei crediti associati a `i`
    w_data = 1.0 / w_ / w2_

    #time_bins = np.arange(delta_bin/2.0, time_mat + 1.0*delta_bin/2, delta_bin)
    hist_, bin_edges_ = np.histogram(subsequent_default_times, bins=n_bins_ref_, weights = w_data, density=True)


    plot_to_chk = False
    if (plot_to_chk):

        plt.hist(subsequent_default_times, bins=n_bins_ref_, weights=w_data, edgecolor='black', density=True)
        # Adding titles and labels
        plt.title('Weighted Histogram')
        plt.xlabel('Data')
        plt.ylabel('Frequency')
        plt.show()


    bin_centers_w_ = (bin_edges_[:-1] + bin_edges_[1:]) / 2.0

    bin_widths = np.diff(bin_centers_w_)  # Differenza tra i centri consecutivi
    bin_widths = np.append(bin_widths, bin_widths[-1])  # Imposta la larghezza dell'ultimo bin uguale al precedente

    # Passaggio 2: Calcolo dell'area totale
    total_area = np.sum(hist_ * bin_widths)

    # Passaggio 3: Normalizzazione delle altezze dell'istogramma
    normalized_hist = hist_ / total_area

    return  normalized_hist, bin_centers_w_, N_sample



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
    elif marginal_type == 'gaussian':
        # Gaussian marginal distribution truncated to [0, horizon]
        mean_shift = horizon/3.0
        gaussian_values = norm.ppf(copula_samples) + mean_shift
        default_times = np.clip(gaussian_values, 0, horizon)
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
            w_data.append(1.0/w_)

    return  subsequent_default_times, w_data

def generate_ptf_default(n_assets, horizon, n_sampling, correlation, marginal_type, copula_type, df):
    tot_n_credits = n_assets
    common_bins = np.linspace(0, horizon,int(horizon*12.0))

    n_samples = 1
    cumulative_g_t = np.zeros(len(common_bins) - 1)
    cumulative_def_t = []
    for i in range(0, n_sampling):

        default_times = generate_default_times(n_assets, n_samples, horizon, correlation, marginal_type, copula_type, df)
        all_default_times = default_times.flatten()  # Appiattisce la matrice in un vettore
        all_default_times.sort()

        sub_def_times, w_data = extract_defaut_time_diff(all_default_times, tot_n_credits)
        g_t, bin_edges = np.histogram(sub_def_times, bins=common_bins, weights=w_data, density=True)

        cumulative_g_t += g_t
        cumulative_def_t = np.append(cumulative_def_t, all_default_times)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return cumulative_g_t, cumulative_def_t, bin_centers

def save_data_and_plot(data_save, params, copula_type, marginal_type, cum_g_t, plt):

    par1, par2 = get_par(params, copula_type)

    if (par2 == None):
        file_out = r'gt_via_copulas/gt_%s_%s_par1_%s.csv' % (copula_type, marginal_type, par1)
        plot_file = r'density_%s_%s_par1_%s.png' % (copula_type, marginal_type, par1)

    else:
        file_out = r'gt_via_copulas/gt_%s_%s_par1_%s_par2_%s.csv' % (copula_type, marginal_type, par1, par2)
        plot_file = r'density_%s_%s_par1_%s_par2_%s.png' % (copula_type, marginal_type, par1, par2)

    if (data_save):

        g_t_df = pd.DataFrame({'Time (Months)': bin_gt, 'Probability': cum_g_t})
        g_t_df.to_csv(file_out, index=False)

    if (data_save):
        print('plot_file: ', plot_file)
        plt.savefig('graph\%s' % (plot_file))


def save_resampled_gt(data_save, params, copula_type, marginal_type, bin_gt_sim, gt_sim, bin_gt_res, gt_res):

    par1, par2 = get_par(params, copula_type)

    if (par2 == None):
        file_sim_out = r'gt_resampled/gt_sim_%s_%s_par1_%s.csv' % (copula_type, marginal_type, par1)
        file_res_out = r'gt_resampled/gt_res_%s_%s_par1_%s.csv' % (copula_type, marginal_type, par1)

    else:
        file_sim_out = r'gt_resampled/gt_sim_%s_%s_par1_%s_par2_%s.csv' % (copula_type, marginal_type, par1, par2)
        file_res_out = r'gt_resampled/gt_res_%s_%s_par1_%s_par2_%s.csv' % (copula_type, marginal_type, par1, par2)


    if (data_save):

        g_t_sim_df = pd.DataFrame({'Time (Months)': bin_gt_sim, 'Probability': gt_sim})
        g_t_res_df = pd.DataFrame({'Time (Months)': bin_gt_res, 'Probability': gt_res})
        g_t_sim_df.to_csv(file_sim_out, index=False)
        g_t_res_df.to_csv(file_res_out, index=False)

def save_gt_plot(data_save, copula_type, marginal_type, par1, par2, plt):

    plot_file_gt = r'gt_%s_%s_%s_%s.png'%(copula_type, marginal_type, par1, par2)


    if (data_save):
        plt.savefig('graph\%s' % (plot_file_gt))


def save_rho_plot(data_save, copula_type, marginal_type, par1, par2, plt):

    plot_file_rho = r'rho_%s_%s_%s_%s.png'%(copula_type, marginal_type, par1, par2)

    if (data_save):
        plt.savefig('graph\%s' % (plot_file_rho))





if __name__ == "__main__":

    import time
    a0 = time.time()

    # Parametri del portafoglio
    n_assets = 500# Numero di asset nel portafoglio
    #n_samples = 1  # Numero di campioni da generare
    n_sampling = 200
    num_resamples = 2000
    n_samp = 20
    horizon = 10  # Orizzonte temporale in anni (massimo tempo di default per Uniforme o scala per Esponenziale)
    #data_save = False
    data_save = True
    visualize_plot = False


    # Tipo di marginale da utilizzare ('uniform' o 'exponential')
    marginal_type = 'exponential'  # Cambia tra 'uniform' e 'exponential'
    marginal_type = 'uniform'  # Cambia tra 'uniform' e 'exponential'
    #marginal_type = 'gaussian'  # Cambia tra 'uniform' e 'exponential'

    #copula_type = 't-student'  # Cambia tra 'gaussian', 't-student', 'clayton', 'frank', 'independence'
    copula_type = 'gaussian'
    #copula_type = 'independence'

    correlation = 0.5  # Correlazione tra gli asset

    theta_val = +20.0  # +5, -5, +2, -2, +0.5, -0.5
    theta_val = +1.  # +5, -5, +2, -2, +0.5, -0.5
    df = 4  # Gradi di libertà per la copula t-Student

    rho_val = correlation
    nu_val = df

    params = {'gaussian': {'rho': rho_val}, 'clayton': {'theta': theta_val},
              't-student': {'nu': nu_val, 'rho': rho_val}, 'frank': {'theta': theta_val}}

    print('1) Start generating defaults')
    common_bins = np.linspace(0, horizon,int(horizon*12.0))
    cum_g_t, cum_def_t, bin_gt = generate_ptf_default(n_assets, horizon, n_sampling, correlation, marginal_type, copula_type, df)

    plt.figure(figsize=(12, 6))
    width_histo = (bin_gt[1] - bin_gt[0])

    tot_area = np.sum(cum_g_t * width_histo)
    cum_g_t = cum_g_t / tot_area

    #plt.bar(bin_gt, cum_g_t, width=width_histo, edgecolor='black', align='edge')
    def_hist, bin_edges_def = np.histogram(cum_def_t, bins=common_bins, density=True)
    bin_centers_def = (bin_edges_def[:-1] + bin_edges_def[1:]) / 2
    b0 = time.time()
    #plt.show()
    #
    print('Time to generate the Sim: %.2f'%(b0- a0))

    if (visualize_plot):
        # PLOT g(t)
        plt.xlabel('Time [mnth] (since a Default event)')
        plt.ylabel('g(t) Pair Default probability')
        plt.title('g(t), N. Credits %s, rho: %s' % (str(n_assets), str(correlation)))
        plt.grid(True)
        plt.show()


        # PLOT Density
        def_hist, bin_edges_def = np.histogram(cum_def_t, bins=common_bins, density=True)
        bin_centers_def = (bin_edges_def[:-1] + bin_edges_def[1:]) / 2
        width_histo_d = (bin_edges_def[1] - bin_edges_def[0])
        plt.bar(bin_centers_def, def_hist, width=width_histo_d, edgecolor='black', align='edge')
        plt.title(f"Distribuzione Tempi di Default ({copula_type.capitalize()} Copula - Marginale: {marginal_type.capitalize()} - rho: {rho_val})")
        plt.xlabel('Tempo di Default [months]')
        plt.ylabel('Frequenza')
        plt.show()

    #-----------------------------------------------------
    # RESAMPLING from the empirical distribution
    #-----------------------------------------------------
    print('2) Start resanmpling default')
    a = time.time()

    histogram_values = def_hist
    bin_edges = bin_centers_def

    # Normalizzare l'istogramma per ottenere una distribuzione di probabilità
    histogram_values = histogram_values / histogram_values.sum()
    cdf_values = np.cumsum(histogram_values)  # CDF cumulativa

    # Chk che `bin_edges` e `cdf_values` abbiano la stessa lunghezza
    if len(cdf_values) == len(bin_edges):
        bin_edges_extended = bin_edges
    else:
        bin_edges_extended = bin_edges[:len(cdf_values)]  # Troncamento a lunghezza corretta

    # Interpolazione inversa della CDF (usando gli array con la stessa lunghezza)
    inverse_cdf = interp1d(cdf_values, bin_edges_extended, bounds_error=False, fill_value=(0, 10))
    resampled_defaults = np.array([])


    tot_n_credits = 2000
    cum_res_g_t = np.zeros(len(common_bins) - 1)
    cum_res_rho_t = np.zeros(len(common_bins) - 1)



    for i in range(n_samp):
        print('i: ', i)
        uniform_samples = np.random.uniform(0, 1, num_resamples)
        resampled_defaults_ = inverse_cdf(uniform_samples)
        #resampled_defaults = np.append(resampled_defaults, resampled_defaults_)


        resampled_defaults = resampled_defaults_.flatten()  # Appiattisce la matrice in un vettore
        resampled_defaults.sort()

        sub_def_times, w_data = extract_defaut_time_diff(resampled_defaults, tot_n_credits)
        g_t, bin_edges = np.histogram(sub_def_times, bins=common_bins, weights=w_data, density=True)
        rho_t, bin_rho = np.histogram(resampled_defaults, bins=common_bins, density=True)

        cum_res_g_t += g_t
        cum_res_rho_t += rho_t

        # = np.append(cumu_res_def_t, resampled_defaults)



    #--------------------------------
    #   COMPUTE THE G(t)
    #--------------------------------
    #resampled_defaults.sort()
    #sub_def_times, w_data = extract_defaut_time_diff(resampled_defaults, 500)
    #g_t_res_y, bin_edges = np.histogram(sub_def_times, bins=common_bins, weights=w_data, density=True)

    g_t_res_x = (bin_edges[:-1] + bin_edges[1:])/ 2.0
    rho_res_x = g_t_res_x

    g_t_sim_x = bin_gt
    rho_sim_x = bin_gt
    rho_res_x = bin_gt

    g_t_sim_y = cum_g_t
    g_t_res_y = cum_res_g_t
    rho_res_y = cum_res_rho_t
    rho_sim_y = def_hist

    bin_widths_sim = g_t_sim_x[2] - g_t_sim_x[1]
    bin_widths_res = bin_edges[2] - bin_edges[1]

    tot_area_gt_sim = np.sum(g_t_sim_y * bin_widths_sim)
    tot_area_gt_res = np.sum(g_t_res_y * bin_widths_res)

    tot_area_rho_sim = np.sum(rho_sim_y * bin_widths_res)
    tot_area_rho_res = np.sum(rho_res_y * bin_widths_res)

    g_t_sim_y = g_t_sim_y / tot_area_gt_sim
    g_t_res_y = g_t_res_y / tot_area_gt_res

    rho_sim_y = rho_sim_y / tot_area_rho_sim
    rho_res_y = rho_res_y / tot_area_rho_res

    #save_data_and_plot(False, params, copula_type, marginal_type, cum_g_t, plt)

    b = time.time()

    print('Time to resample %.2f: '%(b - a))
    par1, par2 = get_par(params, copula_type)



    plt.plot(g_t_sim_x, g_t_sim_y)
    plt.plot(g_t_res_x, g_t_res_y)
    plt.xlabel('Time (Years)')
    plt.ylabel('g(t) Frequency')
    plt.title(f"Simlated vs Resampled g(t): Copula %s, Marginal %s, rho =  %s."%(copula_type, marginal_type, correlation))
    plt.legend(['Simulated', 'Resampled'])

    save_gt_plot(data_save, copula_type, marginal_type, par1, par2, plt)
    plt.show()


    plt.plot(rho_sim_x, rho_sim_y)
    plt.plot(rho_res_x, rho_res_y)
    plt.xlabel('Time (Years)')
    plt.ylabel('Density Frequency')
    plt.title(f"Simlated vs Resampled rho(t): Copula %s, Marginal %s, rho =  %s"%(copula_type, marginal_type, correlation))
    plt.legend(['Simulated', 'Resampled'])

    save_rho_plot(data_save, copula_type, marginal_type, par1, par2, plt)
    plt.show()





