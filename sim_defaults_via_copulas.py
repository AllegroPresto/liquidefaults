
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


def clean_subsequent_default(default_sim):

    rows, cols = default_sim.shape
    for col in range(cols):
        one_found = False
        for row in range(rows):
            if default_sim[row, col] == 1:
                if one_found:
                    default_sim[row, col] = 0
                else:
                    one_found = True

    return  default_sim


def set_default_time(default_sim):

    rows, cols = default_sim.shape
    default_times = []

    for row in range(rows):
        for col in range(cols):
            if default_sim[row, col] == 1:
                event_value = (row + 1)
                default_times.append(event_value)

    return  default_times

def plot_default_density(default_events, time_horizon_months, num_subjects, prob_1y, rho_val, flag_save, output_file):

    default_counts = np.sum(default_events, axis=1)
    x_bar = range(1, time_horizon_months + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(x_bar, default_counts, color='blue', edgecolor='black', alpha=0.75)
    plt.xlabel('N. months')
    plt.ylabel('Number of Default events')
    plt.title('T=%s years, n. items %s, n. def. %s, Prob. def.(1Y): %s, rho = %s'% (
    time_horizon_years, num_subjects, int(num_defaults), prob_1y, str(rho_val)))
    plt.grid(True)

    # Save the plot to a file
    if (flag_save):
        plt.savefig('graph\%s'%(output_file))

    plt.show()


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

def compute_gt(default_times, tot_n_credits):

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


def def_via_gaussian_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type):
    # Creazione della matrice di correlazione (usiamo una correlazione media)
    rho = params[copula]['rho']
    Sigma = np.full((num_subjects, num_subjects), rho) + np.eye(num_subjects) * (1 - rho)

    # simulazione della distribuzione normale multivariata
    mv_normal = multivariate_normal(mean=np.zeros(num_subjects), cov=Sigma)
    z_sim = mv_normal.rvs(size=time_horizon_months)

    # Trasformazione inversa nei quantili uniformi
    u_sim = norm.cdf(z_sim)

    # Step 5: Determinazione degli eventi di default: Bernoulli
    default_events = (u_sim <= lambda_monthly).astype(int)

    lambda_exp = lambda_monthly*12.0
    time_to_def_exp = -np.log(1.0 - u_sim) / lambda_exp
    default_threshold = 1.0/12.0  # We assume this threshold as 1 year

    if distrib_type == 'uniform':

        default_events = (time_to_def_exp <= default_threshold).astype(int)

    elif distrib_type == 'exp':

        time_to_default_exponential = time_to_def_exp
        default_events = np.zeros((time_horizon_months, num_subjects), dtype=int)
        time_to_default_collected = np.zeros((time_horizon_months, num_subjects))

        # Iterate over each month to determine if a default occurs
        for month in range(time_horizon_months):
            current_time_years = month / 12.0  # Convert month index to years
            # Determine if a default event occurs for each subject
            #print('time_to_default_exponential: ', time_to_default_exponential[0])

            defaults_this_month = (time_to_default_exponential[month] <= current_time_years)

            # Update default events and collect time to default
            default_events[month] = defaults_this_month.astype(int)
            time_to_default_collected[month] = time_to_default_exponential[month]

    else:

        print('Distrib type not available!!')
        FQ(99)

    return  default_events


def def_via_clayton_copula(params, num_subjects, time_horizon_months, lambda_monthly):
    # Creazione della matrice di correlazione (usiamo una correlazione media)
    theta = params[copula]['theta']
    # Step 2: Generate dependent uniform random variables using Clayton copula
    u = np.random.uniform(size=(time_horizon_months, num_subjects))
    v = np.random.uniform(size=(time_horizon_months, num_subjects))

    # Apply the inverse of the Clayton copula
    clayton_copula = (u ** (-theta) - 1 + v ** (-theta)) ** (-1 / theta)

    # Step 3: Transform to uniform quantiles
    default_events = (clayton_copula <= norm.cdf(lambda_monthly)).astype(int)

    return  default_events


def def_via_tstudent_copula(params, num_subjects, time_horizon_months, lambda_monthly):
    nu = params[copula]['nu']
    rho_t = params[copula]['rho']

    # Step 2: Create correlation matrix (assuming average correlation)
    # rho_t = 0.2  # 20% average correlation
    Sigma = np.full((num_subjects, num_subjects), rho_t) + np.eye(num_subjects) * (1 - rho_t)

    # Step 3: Simulate multivariate t-distribution
    L = np.linalg.cholesky(Sigma)
    Z = np.random.standard_t(df=nu, size=(time_horizon_months, num_subjects))
    Z = Z @ L.T  # Apply the Cholesky decomposition for the correlation structure

    # Step 4: Transform to uniform quantiles using the t-Student CDF
    u_sim = t.cdf(Z, df=nu)

    # Step 5: Determine monthly default events for each subject
    default_events = (u_sim <= norm.cdf(lambda_monthly)).astype(int)

    return  default_events



def def_via_frank_copula(params, num_subjects, time_horizon_months):
    # Creazione della matrice di correlazione (usiamo una correlazione media)
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
    default_events = (v_sim <= norm.cdf(lambda_monthly)).astype(int)

    return  default_events



if __name__ == "__main__":


    # Parameters
    num_subjects = 1000
    time_horizon_years = 5.0
    time_horizon_months = int(time_horizon_years * 12)
    prob_default_1yr = 0.05 # Annual default probability

    # Step 1 Transform annual default probability to monthly default rate
    lambda_annual = -np.log(1 - prob_default_1yr)
    lambda_monthly = lambda_annual/ 12.0
    print('lambda_monthly: ', lambda_monthly)

    copula       = 'gaussian'
    distrib_type = 'uniform'
    distrib_type = 'exp'

    #copula = 'clayton'
    #copula = 'tstudent'
    #copula = 'frank'

    rho_val = 0.8

    #theta_val = 0.1 #0.1, 0.5, 1.5, 2.5, 4, 5, 7
    theta_val = 20.0

    #theta_val = -10.0 #+5, -5, +2, -2, +0.5, -0.5
    #theta_val = +2.0 #+5, -5, +2, -2, +0.5, -0.5
    #theta_val = -0.5 #+5, -5, +2, -2, +0.5, -0.5

    nu_val = 5
    rhot_val = 0.001


    #flag_density_plot = True
    flag_density_plot = True
    flag_gt_plot = True

    #gt_save = True
    gt_save = False
    flag_d_save = False



    params = {'gaussian': {'rho': rho_val}, 'clayton': {'theta': theta_val},
              'tstudent': {'nu': nu_val, 'rho': rhot_val}, 'frank': {'theta': theta_val}}

    if (copula == 'gaussian'):
        default_events = def_via_gaussian_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type)

    elif (copula == 'clayton'):
        default_events = def_via_clayton_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type)

    elif (copula == 'tstudent'):
        default_events = def_via_tstudent_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type)

    elif (copula == 'frank'):
        default_events = def_via_frank_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type)


    else:

        print('Not available')
        FQ(99)

    # From default event to default time
    default_events = clean_subsequent_default(default_events)
    default_times = set_default_time(default_events)

    tot_n_credits = num_subjects
    sub_def_times, w_data = compute_gt(default_times, tot_n_credits)


    # Histogram g(t)
    time_bins = np.arange(1, time_horizon_months + 1, 3)
    g_t, _ = np.histogram(sub_def_times, bins=time_bins, weights = w_data, density=True)
    time_bins_yy = time_bins/12.0

    if (gt_save):
        par1, par2 = get_par(params, copula)
        if (par2 == None):
            file_out = r'gt_via_copulas/gt_%s_%s_par1_%s.csv' % (copula, distrib_type, par1)
        else:
            file_out = r'gt_via_copulas/gt_%s_%s_par1_%s_par2_%s.csv' % (copula, distrib_type, par1, par2)

        g_t_df = pd.DataFrame({'Time (Months)': time_bins[:-1], 'Probability': g_t})
        g_t_df.to_csv(file_out, index=False)

    num_defaults = len(default_times)

    if (flag_gt_plot):
        plt.figure(figsize=(12, 6))
        plt.bar(time_bins_yy[:-1], g_t, width=1.0/12, edgecolor='black', align='edge')
        plt.xlabel('Time [years] (since a Default event)')
        plt.ylabel('g(t) Pair Default probability')
        plt.title('g(t), N. Credits %s, Prob. def.(1Y) %s, rho: %s'%(str(tot_n_credits), str(prob_default_1yr), str(rho_val)))
        plt.grid(True)
        plt.show()

        # Output the number of total defaults simulated
        print(f"Number of defaults simulated over the next 10 years: {num_defaults}")


        max_y = 1.2*max(g_t)
        time_bins_c = (time_bins_yy[:-1] + time_bins_yy[1:]) / 2.0
        plt.plot(time_bins_c, g_t, '--')
        plt.ylim(0, max_y)
        plt.xlabel('Time [years] (since a Default event)')
        plt.ylabel('g(t) Pair Default probability')
        plt.title('g(t), N. Credits %s, Prob. def.(1Y) %s, rho: %s'%(str(tot_n_credits), str(prob_default_1yr), str(rho_val)))
        plt.show()


    if (flag_density_plot):

        par1, par2 = get_par(params, copula)
        if (par2 == None):
            d_file = r'density_%s_%s_par1_%s.png' % (copula, distrib_type, par1)
        else:
            d_file = r'density_%s_%s_par1_%s_par2_%s.png' % (copula, distrib_type, par1, par2)

        plot_default_density(default_events, time_horizon_months, num_subjects, prob_default_1yr, rho_val, flag_d_save, d_file)


