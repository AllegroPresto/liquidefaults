
import numpy as np
from scipy.stats import norm, multivariate_normal, poisson
import matplotlib.pyplot as plt

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()






if __name__ == "__main__":

    import numpy as np
    from scipy.stats import t, norm, poisson
    import matplotlib.pyplot as plt

    # Parameters
    num_subjects = 1000
    time_horizon_years = 10
    time_horizon_months = time_horizon_years * 12
    prob_default_1yr = 0.05  # Annual default probability
    nu = 5  # Degrees of freedom for the t-Student copula

    # Step 1: Transform annual default probability to monthly default rate
    lambda_annual = -np.log(1 - prob_default_1yr)
    lambda_monthly = lambda_annual / 12

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

    # Step 6: Simulate default events using a Poisson process
    monthly_defaults = np.zeros(time_horizon_months)

    # Count defaults using a Poisson process
    for month in range(time_horizon_months):
        for i in range(num_subjects):
            if default_sim[month, i] == 1:
                # Simulate number of defaults for each subject in the month
                monthly_defaults[month] += poisson.rvs(lambda_monthly)

    # Step 7: Compute the times of all defaults
    default_times = np.where(monthly_defaults > 0)[0]

    # Step 8: Compute subsequent default times for each default event
    subsequent_default_times = []
    for i in range(len(default_times)):
        for j in range(i + 1, len(default_times)):
            subsequent_default_times.append(default_times[j] - default_times[i])

    # Step 9: Estimate g(t) - probability distribution of all subsequent default events
    time_bins = np.arange(1, time_horizon_months + 1)
    g_t, _ = np.histogram(subsequent_default_times, bins=time_bins, density=True)

    # Step 10: Plot the distribution of subsequent default times (g(t))
    plt.figure(figsize=(12, 6))
    plt.bar(time_bins[:-1], g_t, width=1, edgecolor='black', align='edge')
    plt.xlabel('Months since a Default Event')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Time to Subsequent Default Events (g(t))')
    plt.grid(True)
    plt.show()

    # Output the number of total defaults simulated
    num_defaults = np.sum(monthly_defaults)
    print(f"Number of defaults simulated over the next 10 years: {num_defaults}")
