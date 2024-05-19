from scipy.stats import binom
import numpy

def get_stat_power(N, mu_h0, mu_alternative, alpha):
    """Вычисляет статистическую мощность критерия для биномиального распределения

    Параметры:
        N - количество бернуллиевских экспериментов (размер выборки)
        mu_h0 - вероятность успеха в нулевой гипотезе
        mu_alternative - предполагаемая вероятность успеха в эксперименте
        alpha - уровень значимости критерия
    """
    binom_h0 = binom(n=N, p=mu_h0)
    binom_alternative = binom(n=N, p=mu_alternative)

    # вычисляем критическое значение
    critical_value = binom_h0.ppf(1 - alpha) + 1
    # вычисляем мощность по формуле
    return 1 - binom_alternative.cdf(critical_value - 1)


def binom_test_mde_one_sided(N, mu0, alpha=0.05, min_power=0.8):
    """Вычисляет MDE одностороннего критерия для проверки гипотезы mu = mu0 в задаче

    Параметры:
        N (int) - размер выборки
        mu0 (float) - вероятность успеха в нулевой гипотезе
        alpha (float) - уровень значимости критерия
        min_power (float) - желаемая мощность

    Возвращает:
        float - MDE"""
    delta_grid = numpy.linspace(0, 1 - mu0, 500)
    power = get_stat_power(N, mu0, mu0 + delta_grid, alpha=alpha)
    # выберем подходящие delta и вернем первую
    fit_delta = delta_grid[power >= min_power]
    return fit_delta[0]
