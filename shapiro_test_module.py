import numpy as np
import pandas as pd
from scipy.optimize import anderson
from scipy.stats import shapiro, zscore, f_oneway, chi2_contingency, chisquare, normaltest
import statsmodels.api as sm


def test_outlier_detection_and_removal(df, numeric_columns):
    # Aykırı değerleri tespit eder ve kaldırır, ardından sonuçları değerlendirir.
    cleaned_df = detect_and_remove_outliers(df, numeric_columns)

    if len(cleaned_df) < len(df):
        print("Aykırı değerler kaldırılmamış.")
    if cleaned_df[numeric_columns].isnull().sum().sum() == 0:
        print("NaN değerleri içeren satırlar bulunuyor.")


def detect_and_remove_outliers(df, numeric_columns):
    # Z-skor kullanarak aykırı değerleri tespit eder ve kaldırır.
    z_scores = np.abs((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std())
    outliers = (z_scores > 3).all(axis=1)
    df_cleaned = df[~outliers]

    print("Aykırı Değerlerin İndeksleri:")
    print(df[outliers].index)

    return df_cleaned


def detect_and_remove_outliers_zscore(df, numeric_columns, threshold=3):
    # Z-skor kullanarak aykırı değerleri tespit eder ve kaldırır.
    z_scores = np.abs(zscore(df[numeric_columns]))
    outlier_indices = np.where(z_scores > threshold)

    # Aykırı değerleri kaldır
    df_cleaned = df.drop(outlier_indices[0])

    return df_cleaned


def detect_and_remove_outliers_iqr(df, numeric_columns):
    # IQR (Interquartile Range) kullanarak aykırı değerleri tespit eder ve kaldırır.
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1

    # Aykırı değerleri kaldır
    df_cleaned = df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_cleaned


def shapiro_test(df, numeric_columns):
    # Shapiro-Wilk testi kullanarak normalite testi gerçekleştirir.
    for col in numeric_columns:
        stat, p_value = shapiro(df[col])
        print(f'Statistic for {col}: {stat}, p-value: {p_value}')

    alpha = 0.05
    if p_value > alpha:
        print("Örneklem normal bir dağılıma sahiptir (H0 reddedilemez)")
    else:
        print("Örneklem normal bir dağılıma sahip değildir (H0 reddedilir)")


def parametric_test_anova(df, numeric_columns):
    # Parametrik ANOVA testi örneği
    stat, p_value = f_oneway(df[numeric_columns[0]], df[numeric_columns[1]], df[numeric_columns[2]],
                             df[numeric_columns[3]])

    print(f"ANOVA Test İstatistiği: {stat}")
    print(f"P Değeri: {p_value}")

    if p_value < 0.05:
        print("Sıfır hipotez reddedildi. Ortalamalar arasında anlamlı bir fark vardır.")
    else:
        print("Sıfır hipotez kabul edildi. Ortalamalar arasında anlamlı bir fark yoktur.")


def non_parametric_test_chi_square(df, categorical_column, target_column):
    # Parametrik olmayan Ki-Kare testi örneği
    contingency_table = pd.crosstab(df[categorical_column], df[target_column])
    stat, p_value, _, _ = chi2_contingency(contingency_table)

    print(f"Ki-Kare Test İstatistiği: {stat}")
    print(f"P Değeri: {p_value}")

    if p_value < 0.05:
        print("Sıfır hipotez reddedildi. İki değişken arasında anlamlı bir bağımsızlık vardır.")
    else:
        print("Sıfır hipotez kabul edildi. İki değişken arasında anlamlı bir bağımsızlık yoktur.")


def one_sample_chi_square_test(observed_values, expected_values):
    # Tek örneklem Ki-Kare testi örneği
    stat, p_value = chisquare(f_obs=observed_values, f_exp=expected_values)

    print(f"Tek Örneklem Ki-Kare Testi İstatistiği: {stat}")
    print(f"P Değeri: {p_value}")

    if p_value < 0.05:
        print("Sıfır hipotez reddedildi. İki dağılım arasında anlamlı bir fark vardır.")
    else:
        print("Sıfır hipotez kabul edildi. İki dağılım arasında anlamlı bir fark yoktur.")


def d_agostino_test_normality(data):
    # D'Agostino'nun K^2 Testi kullanarak normalite testi
    stat, p_value = normaltest(data)
    print(f"D'Agostino'nun K^2 Test İstatistiği: {stat}")
    print(f"P Değeri: {p_value}")

    if p_value < 0.05:
        print("Veri normal dağılımdan gelmiyor. H0 reddedildi.")
    else:
        print("Veri normal dağılımdan gelmektedir. H0 kabul edildi.")


def anderson_test_normality(data):
    # Anderson-Darling Testi kullanarak normalite testi
    result = anderson(data)
    print(f"Anderson-Darling Test İstatistiği: {result.statistic}")
    print(f"P Değeri: {result.critical_values[2]}")

    if result.statistic > result.critical_values[2]:
        print("Veri normal dağılımdan gelmiyor. H0 reddedildi.")
    else:
        print("Veri normal dağılımdan gelmektedir. H0 kabul edildi.")


def multivariate_regression(data, dependent_variable, independent_variables):
    # Çok Değişkenli Regresyon Modeli
    X = sm.add_constant(data[independent_variables])
    print("X verisi")
    print(X.head())
    y = data[dependent_variable]
    print("veri bitti")
    model = sm.OLS(y, X).fit()

    # Print the model summary
    print(model.summary())
    print(model.k_constant)