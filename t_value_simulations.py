import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp

if __name__ == "__main__":

    X = np.linspace(-5, 5, 100000)

    def cdf_5(t):
        return 1/2 + (1/np.pi) * ((t/(np.sqrt(5)*(1+((t**2)/5))))*(1+(2/(3*(1+(t**2)/5))))+np.arctan(t/np.sqrt(5)))

    Y = [cdf_5(x) for x in X]
    closest = min(Y, key=lambda x:abs(x-0.95))
    print(f"x={X[np.argwhere(Y == closest)[0][0]]}")

    plt.plot(X, Y)
    plt.ylim((0, 1))
    plt.axhline(0.95, color="r")

    def cdf_5_min(t):
        return 1 - (1-cdf_5(t))**2

    Y = [cdf_5_min(x) for x in X]
    closest = min(Y, key=lambda x:abs(x-0.95))
    print(f"x={X[np.argwhere(Y == closest)[0][0]]}")
    # Y_round = np.array([cdf_5_min(x) for x in X])
    plt.plot(X, Y)
    plt.ylim((0, 1))
    plt.axhline(0.95, color="r")
    plt.show()

    #x=0.824

    #
    # t_values = []
    #
    # n_samples = 6
    # popmean = 0
    #
    # for r in range(10000):
    #     data = np.random.normal(0, size=n_samples)
    #
    #     # t_value = (np.mean(data) - popmean) / (np.std(data) / np.sqrt(n_samples))
    #     t_value = ttest_1samp(data, popmean=popmean, alternative="greater")[0]
    #
    #     t_values.append(t_value)
    #
    # print("X1")
    # print("95th quantile: ", np.quantile(t_values, 0.95))
    # print("99th quantile: ", np.quantile(t_values, 0.99))
    #
    # plt.hist(t_values, bins=100, alpha=0.5)
    #
    # t_values = []
    # for r in range(10000):
    #     data = np.random.normal(0, size=n_samples)
    #
    #     # t_value = (np.mean(data) - popmean) / (np.std(data) / np.sqrt(n_samples))
    #     data_2 = np.random.normal(0, size=n_samples)
    #
    #     data_min = np.min((data, data_2), axis=0)
    #     # t_value = (np.mean(data_min) - popmean) / (np.std(data_min) / np.sqrt(n_samples))
    #     t_value = ttest_1samp(data_min, popmean=popmean, alternative="greater")[0]
    #
    #     t_values.append(t_value)
    #
    # print("min(X1, X2):")
    # x_quantile_95 = np.quantile(t_values, 0.95)
    # x_quantile_99 = np.quantile(t_values, 0.99)
    # print("95th quantile: ", x_quantile_95)
    # print("99th quantile: ", x_quantile_99)
    #
    # plt.hist(t_values, bins=100, alpha=0.8)
    # plt.axvline(x=x_quantile_95, color='r', linestyle='--')
    # plt.axvline(x=x_quantile_99, color='r', linestyle=':')
    #
    # plt.show()



















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
