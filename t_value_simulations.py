import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp

if __name__ == "__main__":
    t_values = []

    n_samples = 6
    popmean = 0

    for _ in range(100000):
        data = np.random.normal(0, size=n_samples)

        # t_value = (np.mean(data) - popmean) / (np.std(data) / np.sqrt(n_samples-1))

        # degrees_of_freedom = len(data) - 1
        # print(degrees_of_freedom)

        t_value = ttest_1samp(data, popmean=popmean, alternative="greater")[0]

        t_values.append(t_value)

    print("X1")
    print("95th quantile: ", np.quantile(t_values, 0.95))
    print("99th quantile: ", np.quantile(t_values, 0.99))

    plt.hist(t_values, bins=100, alpha=0.5)

    t_values = []
    for _ in range(100000):
        data = np.random.normal(0, size=n_samples)

        # t_value = (np.mean(data) - popmean) / (np.std(data) / np.sqrt(n_samples-1))

        data_2 = np.random.normal(0, size=n_samples)

        data_min = np.min((data, data_2), axis=0)
        # t_value = (np.mean(data_min) - popmean) / (np.std(data_min) / np.sqrt(n_samples-1))

        # print(t_value)

        # degrees_of_freedom = len(data) - 1
        # print(degrees_of_freedom)

        # t_value = ttest_1samp(data, popmean=popmean, alternative="greater")[0]
        t_value = ttest_1samp(data_min, popmean=popmean, alternative="greater")[0]

        t_values.append(t_value)

    print("min(X1, X2):")
    x_quantile_95 = np.quantile(t_values, 0.95)
    x_quantile_99 = np.quantile(t_values, 0.99)
    print("95th quantile: ", x_quantile_95)
    print("99th quantile: ", x_quantile_99)

    plt.hist(t_values, bins=100, alpha=0.8)
    plt.axvline(x=x_quantile_95, color='r', linestyle='--')
    plt.axvline(x=x_quantile_99, color='r', linestyle=':')

    plt.show()



















    # print(np.mean(t_values))
    # print(np.mean([(t > 2.015) for t in t_values]))

    # df = n_samples - 1
    # p_values = [scipy.special.stdtr(df, -np.mean(t_value)) for t_value in t_values]
    # print("mean p values: ", np.mean(p_values))
    #
    # t_value = scipy.stats.t.ppf(0.95, df)
    # p_value = scipy.stats.t.cdf(2.015, 5)
    # scipy.stats.t.sf(np.abs(2.015), 5)
    #
    #
    #
    # print("p value for 0.95 = ", scipy.special.stdtr(df, np.percentile(t_values, 95)))
    # print("p value for 0.95 = ", np.percentile([scipy.special.stdtr(df, t_value) for t_value in t_values]))


    # data_2 = np.random.random(100) - 0.43
    #
    # print("min:")
    # data_min = np.min((data, data_2), axis=0)
    # # print(data_min)
    # t_value = (np.mean(data_min) - popmean) / (np.std(data_min) / np.sqrt(len(data_min)))
    # print(t_value)
    # print(ttest_1samp(data_min, popmean=popmean, alternative="greater"))
