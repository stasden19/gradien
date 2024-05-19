from scipy.stats import binom


def make_binom_criterion(n, mu=0.5, alpha=0.05):
    """Строит статистический критерий

    Параметры:
        n: количество доставок в эксперименте
        mu: вероятность успеха в нулевой гипотезе
        alpha: уровень значимости критерия

    Возвращает:
        C для критерия S = {Q >= C}
    """
    binom_h0 = binom(n=n, p=mu)
    q = binom_h0.ppf(1 - alpha)
    return q + 1
print(f'ЕслиQ >=', make_binom_criterion(
    n=30,
    mu=0.5,
    alpha=0.05
), 'то мы не принимаем "заказ" в исполнение.')