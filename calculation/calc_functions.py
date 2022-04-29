from collections import Counter
import scipy.stats

import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool


cache = {}


def cached(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        if func_name in cache:
            return cache[func_name]
        result = func(*args, **kwargs)
        cache[func_name] = result
        return result
    return wrapper


@cached
def get_mean(vector: list):
    return sum(vector) / len(vector)


@cached
def get_median(vector: list):
    n = len(vector)
    mid = n // 2
    vector_sorted = sorted(vector)

    if n % 2 == 0:
        return get_mean([vector_sorted[mid], vector_sorted[mid - 1]])
    return vector_sorted[mid]


@cached
def get_variance(vector: list):
    mean = get_mean(vector)
    n = len(vector)
    n = n - 1 if n <= 30 else n

    deviations_sum = 0
    for entry in vector:
        deviations_sum += (entry - mean) ** 2

    return deviations_sum / n


@cached
def get_standard_deviation(vector: list):
    return get_variance(vector) ** 0.5


def get_mode(vector: list):
    most_common = Counter(vector).most_common()
    frequency = most_common[0][1]

    medians = []
    for entry in most_common:
        if entry[1] == frequency:
            medians.append(entry[0])
        else:
            break

    if len(medians) == len(vector):
        return 'does not contain any duplicate numbers'
    return medians


@cached
def get_confidence_interval_variance(vector: list, gamma: float = 0.95) -> tuple:
    n = len(vector)
    deg_of_freedom = n - 1
    significance_lvl_left = round((1 + gamma) / 2, 2)
    significance_lvl_right = round((1 - gamma) / 2, 2)

    chi2_crit_val_left = scipy.stats.chi2.ppf(
        significance_lvl_left, deg_of_freedom
    )
    chi2_crit_val_right = scipy.stats.chi2.ppf(
        significance_lvl_right, deg_of_freedom
    )

    variance = get_variance(vector)

    deg_of_freedom_and_variance_prod = deg_of_freedom * variance

    return (
        deg_of_freedom_and_variance_prod / chi2_crit_val_left,
        deg_of_freedom_and_variance_prod / chi2_crit_val_right
    )


def get_confidence_interval_mean(vector: list, gamma: float = 0.95) -> tuple:
    n = len(vector)
    deg_of_freedom = n - 1
    alpha = 1-(1-gamma)/2
    t_crit_val = scipy.stats.t.ppf(alpha, deg_of_freedom)

    same_term = (t_crit_val * get_standard_deviation(vector)) / (n ** 0.5)

    mean = get_mean(vector)

    return mean - same_term, mean + same_term


def get_confidence_interval_st_dev(vector: list, gamma: float = 0.95) -> tuple:
    var_interval = get_confidence_interval_variance(vector, gamma)
    return var_interval[0] ** 0.5, var_interval[1] ** 0.5


def get_central_moment(vector: list, order: int = 3):
    mean = get_mean(vector)
    result = 0
    for entry in vector:
        result += (entry - mean) ** order
    return result / len(vector)


def get_values_mean_difference(vector: list, order: int = 3):
    mean = get_mean(vector)
    result = 0
    for entry in vector:
        result += (entry - mean) ** order
    return result


def get_skewness(vector: list):
    n = len(vector)
    return (
        (get_values_mean_difference(vector, order=3))
        / (get_standard_deviation(vector) ** 3)
        * (n / (n-1) / (n-2))
    )


def get_kurtosis(vector: list):
    n = len(vector)
    return (
        n
        * (n + 1)
        * get_values_mean_difference(vector, order=4)
        / get_standard_deviation(vector) ** 4
        / (n - 1)
        / (n - 2)
        / (n - 3)
        - (
                (3 * (n - 1)**2)
                / (n-2)
                / (n-3)
           )
    )


def process_column(
        vector: list,
        col_name: str,
        gamma_for_mean: float = 0.95,
        gamma_for_variance: float = 0.95,
        gamma_for_st_dev: float = 0.95
):

    conf_int_mean = get_confidence_interval_mean(vector, gamma_for_mean)
    conf_int_var = get_confidence_interval_variance(vector, gamma_for_variance)
    conf_int_st_dev = get_confidence_interval_st_dev(vector, gamma_for_st_dev)

    stats = {
        'mean': round(get_mean(vector), 3),
        'variance': round(get_variance(vector), 3),
        'standard deviation': round(get_standard_deviation(vector), 3),
        'mode': get_mode(vector),
        'median': round(get_median(vector), 3),
        'skewness': round(get_skewness(vector), 3),
        'kurtosis': round(get_kurtosis(vector), 3),
        f'confidence interval for mean, {gamma_for_mean}':
            (round(conf_int_mean[0], 3), round(conf_int_mean[1], 3)),
        f'confidence interval for variance, {gamma_for_variance}':
            (round(conf_int_var[0], 3), round(conf_int_var[1], 3)),
        f'confidence interval stand. dev., {gamma_for_st_dev}':
            (round(conf_int_st_dev[0], 3), round(conf_int_st_dev[1], 3)),
    }
    clear_cache()

    display_plot_column(vector, col_name)

    return stats


def process_data(
        data: dict,
        gamma_for_mean: float = 0.95,
        gamma_for_variance: float = 0.95,
        gamma_for_st_dev: float = 0.95
):
    statistics = {}
    for column_name, column in data.items():
        statistics[column_name] = process_column(
            column,
            column_name,
            gamma_for_mean,
            gamma_for_variance,
            gamma_for_st_dev
        )
        clear_cache()
    return statistics


def process_data_maternal_risks(data: dict) -> dict:
    return process_data({
        'Age': data['Age'],
        'SystolicBP': data['SystolicBP'],
        'DiastolicBP': data['DiastolicBP'],
        'BS': data['BS'],
        'BodyTemp': data['BodyTemp'],
        'HeartRate': [entry for entry in data['HeartRate'] if entry >= 10],
        'RiskLevel': [risk_lvl_mapping[value] for value in data['RiskLevel']]
    })


def display_plot_column(vector: list, col_name: str):
    pd.DataFrame(vector).hist(bins=20)
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.show()


def clear_cache():
    global cache
    cache = {}




risk_lvl_mapping = {
    'low risk': 0,
    'mid risk': 1,
    'high risk': 2,
}
