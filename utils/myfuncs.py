# Reginaldo Ferreira
import numpy as np
import pandas as pd
from numpy import linalg as LA
from oct2py import octave


def generate_data(model:str, size=1000, noise_amplitude=0.01):

    def generate_input(size:int):
        u = np.ones(size) * 1
        for i in range(1, size):
            v1 = u[i-1]
            v2 = 1 - abs(v1)
            u[i] = np.random.choice([v1, v2], 1, p=[0.9, 0.1])
        return u

    u = generate_input(size)
    y = np.zeros(size)
    y[0:5] = np.random.rand(5)

    for k in range(4, size):
        y[k] = eval(model)

    y = y + noise_amplitude * np.random.rand(size)
    return pd.DataFrame({'u':u, 'y':y})


def generate_terms(terms: list[str]) -> str:

    """
    Esta função edita a equação (srt) de saída do octave paara ser utilizada no python
    """

    eq = []
    if len(terms) > 1:
        for term in terms:
            eq.append(term)
    else:
        eq.append(terms[0])

    eq = (',').join(eq).replace('(', '[').replace(')', ']').replace(',', '')
    for n in range(5, 1, -1):
        pattern1 = n * '['
        pattern2 = n * ']'
        eq = eq.replace(pattern1, '[').replace(pattern2, ']')
    eq = eq.replace('[y', 'y').replace('[u', 'u').replace('+ -', '- ')
    eq = eq.replace('y', 'y_est')
    eq = eq.replace(' * ', '*')
    return eq


def mse(y_est:pd.Series, y:pd.Series) -> int:
    return 1/y.size * np.sum((y - y_est)**2)


def generate_yest(input: pd.Series , output: pd.Series, model: dict):

    terms = model.Terms
    noiseTerms = model.NoiseTerms
    coefs = model.final_Coeff
    noiseCoeff = model.final_NoiseCoeff

    terms = generate_terms(terms)
    noiseTerms = generate_terms(noiseTerms)
    terms  = terms + ',' + noiseTerms

    coefs = np.array(coefs).reshape(-1)
    noiseCoefs = np.array(noiseCoeff).reshape(-1)
    coefs = np.append(coefs, noiseCoefs)

    print(terms)
    print(coefs)
    print('')

    size = input.size
    index = input.index
    u = input.values
    e = np.zeros(size)
    y_est = np.zeros(size)
    # print(f'index de y_est: {y_est.index}')
    y = output.values
    y_est[:10] = y[:10]
    # e[:5] = -y[:5]


    for k in range(5,size):
        val = np.array(eval(terms)) @ coefs
        y_est[k] = val
        # e[k] = sigmoid(y[k] - y_est[k])
        e[k] = (y[k] - y_est[k])
        # y_est[k-1] = y[k-1]

    y_est = pd.Series(y_est, index=index)
    e = pd.Series(e, index=index)
    err = rmse(y_est, output)
    return err, y_est


def gen_table(models: dict) -> pd.DataFrame:
    columns = ['mu', 'my', 'degree', 'coef', 'terms', 'mse']
    df = pd.DataFrame(columns=columns)
    keys = models.keys()
    for key in keys:
        coefs = models[key].final_Coeff
        noiseCoefs = models[key].final_NoiseCoeff
        final_coefs = np.append(coefs, noiseCoefs)
        final_coefs = np.around(final_coefs, 3)

        terms = models[key].Terms
        noiseTerms = models[key].NoiseTerms
        final_terms = np.append(terms, noiseTerms)
        mse = models[key].mse

        mu, my, degree = key
        df.loc[len(df)] = mu, my, degree, final_coefs, final_terms, mse
    return df

def train_models(   y_train: pd.Series,
                    u_train: pd.Series,
                    mu_order: int,
                    max_degree: int,
                    pho: float, phoN: float,
                    dataLength: int, divisions: int, delta_i: int
                ) -> dict:
    models = {}
    for mu in range(1, mu_order):
        for my in range(1, mu+1):
            for degree in range(2, max_degree+1):
                print('ok até aqui:', degree)
                results = octave.return_coefs(  y_train.values.reshape(-1,1),
                                                u_train.values.reshape(-1,1),
                                                mu, my, degree,
                                                pho, phoN, dataLength, divisions, delta_i
                                            );

                models[mu, my, degree] = results

    return models