
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

def plot_default_density(cumulative_def, time_horizon_months, num_subjects, prob_1y, rho_val, flag_save, output_file):


    #print('default_times: ', default_times)
    plt.hist(cumulative_def, bins='auto', edgecolor='black')

    #print('default_events: ', default_events)
    #default_counts = np.sum(default_events, axis=1)
    #print('default_counts: ', default_counts)

    #x_bar = range(1, time_horizon_months + 1)

    #plt.figure(figsize=(10, 6))
    #plt.bar(x_bar, default_counts, color='blue', edgecolor='black', alpha=0.75)
    plt.xlabel('N. months')
    plt.ylabel('Number of Default events')
    plt.title('T=%s month, n. items %s, n. def. %s, Prob. def.(1Y): %s, rho = %s'% (
    time_horizon_years, num_subjects, int(num_defaults), prob_1y, str(rho_val)))
    #plt.grid(True)

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


def extract_prob_g_t(data, perc_n):

    data = data.sort_values(by='TimeToDefault').reset_index(drop=True)
    time_to_default = data['TimeToDefault'].values
    num_credits = data['NumberOfCredits'].values
    N = len(time_to_default)
    N_sample = int(N*perc_n)


    n_bins_ref_ = np.minimum(int(2.0 * np.sqrt(N_sample)), 30)


    N_sample = len(time_to_default)
    print('N_sample: ', N_sample)

    # Step 6: Compute subsequent default times for each default event
    subsequent_default_times = []
    subsequent_default_times_ = []
    w_data = []

    for i in range(N_sample):

        if (i % 1000 == 0):
            print('i: %s/%s'%(i, N_sample))

        for j in range(i + 1, N_sample):
            delta_ = (time_to_default[j] - time_to_default[i]) / 365.0
            subsequent_default_times.append(delta_)

            if (i < int(N_sample/2.0) - 1) and ((j - i)< int(N_sample/2.0)):
                subsequent_default_times_.append(delta_)

            w_ = (N_sample - (j - i))
            w2_= num_credits[i]
            w_data.append(1.0/w_/w2_)
            #w_data.append(1.0)


    #FQ(77)
    #time_bins = np.arange(delta_bin/2.0, time_mat + 1.0*delta_bin/2, delta_bin)
    #hist_, bin_edges_ = np.histogram(subsequent_default_times, bins=n_bins_ref_, weights = w_data, density=True)
    hist_, bin_edges_ = np.histogram(subsequent_default_times, bins='auto', weights = w_data, density=True)

    #hist_2, bin_edges_2 = np.histogram(subsequent_default_times_, bins=n_bins_ref_, density=True)

    plot_to_chk = False
    if (plot_to_chk):


        plt.hist(subsequent_default_times, bins=n_bins_ref_, weights=w_data, edgecolor='black', density=False)

        plt.title('Weighted Histogram')
        plt.xlabel('Data')
        plt.ylabel('Frequency')
        plt.legend(['Weight', 'No weight'])
        plt.show()
        FQ(99)


    bin_centers_w_ = (bin_edges_[:-1] + bin_edges_[1:]) / 2.0

    return  hist_, bin_centers_w_, N_sample


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

def  def_events_from_coupulas(distrib_type, time_horizon_months, num_subjects, lambda_monthly, u_sim):


    default_events = np.zeros((time_horizon_months, num_subjects), dtype=int)

    if distrib_type == 'uniform':

        time_to_default_sim = u_sim * time_horizon_months

    elif distrib_type == 'exp':

        lambda_exp = lambda_monthly * 12.0
        time_to_def_exp = -np.log(1.0 - u_sim) / lambda_exp

        time_to_default_sim = time_to_def_exp
        #time_to_default_collected = np.zeros((time_horizon_months, num_subjects))
    else:

        print('Distrib type not available!!')
        FQ(99)

    # Iterate over each month to determine if a default occurs
    for month in range(time_horizon_months):
        current_time_years = month / 12.0  # Convert month index to years

        defaults_this_month = (time_to_default_sim[month] <= current_time_years)

        # Update default events and collect time to default
        default_events[month] = defaults_this_month.astype(int)
        #time_to_default_collected[month] = time_to_default_exponential[month]



    return  default_events



def def_via_gaussian_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type):
    # Creazione della matrice di correlazione (usiamo una correlazione media)
    rho = params[copula]['rho']
    Sigma = np.full((num_subjects, num_subjects), rho) + np.eye(num_subjects) * (1 - rho)

    # simulazione della distribuzione normale multivariata n. emittenti ptf vs n. month
    mv_normal = multivariate_normal(mean=np.zeros(num_subjects), cov=Sigma)
    z_sim = mv_normal.rvs(size=time_horizon_months)

    # Trasformazione inversa nei quantili uniformi
    u_sim = norm.cdf(z_sim)


    default_events = def_events_from_coupulas(distrib_type, time_horizon_months, num_subjects, lambda_monthly, u_sim)

    return  default_events


def def_via_clayton_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type):
    # Creazione della matrice di correlazione (usiamo una correlazione media)
    theta = params[copula]['theta']
    # Step 2: Generate dependent uniform random variables using Clayton copula
    u = np.random.uniform(size=(time_horizon_months, num_subjects))
    v = np.random.uniform(size=(time_horizon_months, num_subjects))

    # Apply the inverse of the Clayton copula
    u_sim = (u ** (-theta) - 1 + v ** (-theta)) ** (-1 / theta)

    default_events = def_events_from_coupulas(distrib_type, time_horizon_months, num_subjects, lambda_monthly, u_sim)


    return  default_events


def def_via_tstudent_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type):

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

    default_events = def_events_from_coupulas(distrib_type, time_horizon_months, num_subjects, lambda_monthly, u_sim)


    return  default_events



def def_via_frank_copula(params, num_subjects, time_horizon_months, lambda_monthly, distrib_type):
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

    default_events = def_events_from_coupulas(distrib_type, time_horizon_months, num_subjects, lambda_monthly, v_sim)


    return  default_events



if __name__ == "__main__":


    # Parameters
    num_subjects = 1000
    time_horizon_years = 1.0
    time_horizon_months = int(time_horizon_years * 12)
    prob_default_1yr = 0.05 # Annual default probability

    # Step 1 Transform annual default probability to monthly default rate
    lambda_annual = -np.log(1 - prob_default_1yr)
    lambda_monthly = lambda_annual/ 12.0

    copula       = 'gaussian'
    distrib_type = 'uniform'
    #distrib_type = 'exp'

    #copula = 'clayton'
    #copula = 'tstudent'
    #copula = 'frank'

    rho_val = 0.0005
    #rho_val = 0.05
    #rho_val = 0.1
    #rho_val = 0.2
    #rho_val = 0.5
    #rho_val = 0.7
    #rho_val = 0.8

    #theta_val = 0.1 #0.1, 0.5, 1.5, 2.5, 4, 5, 7
    #theta_val = 20.0

    theta_val = +20.0 #+5, -5, +2, -2, +0.5, -0.5
    #theta_val = +2.0 #+5, -5, +2, -2, +0.5, -0.5
    #theta_val = -0.5 #+5, -5, +2, -2, +0.5, -0.5

    nu_val = 1
    rhot_val = 0.1


    #flag_density_plot = True
    flag_density_plot = True
    flag_gt_plot = True

    gt_save = False
    flag_d_save = False

    #gt_save = True
    #flag_d_save = True
    n_sampling = 1


    params = {'gaussian': {'rho': rho_val}, 'clayton': {'theta': theta_val},
              'tstudent': {'nu': nu_val, 'rho': rhot_val}, 'frank': {'theta': theta_val}}

    default_times = []
    common_bins = np.linspace(0, time_horizon_months, time_horizon_months)
    cumulative_g_t = np.zeros(len(common_bins) - 1)
    cumulative_def = []

    for i in range(0, n_sampling):
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


        #print('default_events: ', default_events)
        default_events = clean_subsequent_default(default_events)

        #print('default_events: ', default_events)
        default_times_ = set_default_time(default_events)

        default_times_.sort()

        tot_n_credits = num_subjects
        sub_def_times, w_data = extract_defaut_time_diff(default_times_, tot_n_credits)

        g_t, bin_edges = np.histogram(sub_def_times, bins= common_bins, weights  = w_data, density=True)

        cumulative_g_t += g_t
        cumulative_def += default_times_

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


    if (gt_save):
        par1, par2 = get_par(params, copula)
        if (par2 == None):
            file_out = r'gt_via_copulas/gt_%s_%s_par1_%s.csv' % (copula, distrib_type, par1)
        else:
            file_out = r'gt_via_copulas/gt_%s_%s_par1_%s_par2_%s.csv' % (copula, distrib_type, par1, par2)

        g_t_df = pd.DataFrame({'Time (Months)': bin_centers, 'Probability': cumulative_g_t})
        g_t_df.to_csv(file_out, index=False)

    num_defaults = len(default_times)

    if (flag_gt_plot):
        plt.figure(figsize=(12, 6))
        width_histo = (bin_edges[1] - bin_edges[0])
        plt.bar(bin_centers, cumulative_g_t, width=width_histo, edgecolor='black', align='edge')

        plt.xlabel('Time [mnth] (since a Default event)')
        plt.ylabel('g(t) Pair Default probability')
        plt.title('g(t), N. Credits %s, Prob. def.(1Y) %s, rho: %s'%(str(tot_n_credits), str(prob_default_1yr), str(rho_val)))
        plt.grid(True)
        plt.show()

        # Output the number of total defaults simulated
        #print(f"Number of defaults simulated over the next 10 years: {num_defaults}")


        max_y = 1.2*max(cumulative_g_t)
        plt.plot(bin_centers, cumulative_g_t, '--')
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


        plot_default_density(cumulative_def, time_horizon_months, num_subjects, prob_default_1yr, rho_val, flag_d_save, d_file)


