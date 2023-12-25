import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from shapiro_test_module import test_outlier_detection_and_removal, shapiro_test, detect_and_remove_outliers_iqr, \
    detect_and_remove_outliers_zscore, parametric_test_anova, non_parametric_test_chi_square, \
    one_sample_chi_square_test, multivariate_regression


def load_and_clean_data(file_path):
    # CSV dosyasını oku
    df = pd.read_csv(file_path, delimiter=' ')

    # Veri önizleme
    print("Veri Önizleme:")
    print(df.head())

    # Sayısal sütunları seç
    numeric_columns = ['TDP Watt', 'MHz - Turbo', 'Cores / Threads', 'Perf. Rating',
                       'Cinebench R15 CPU Single 64Bit', 'Cinebench R15 CPU Multi 64Bit',
                       'Cinebench R23 Single Core', 'Cinebench R23 Multi Core',
                       'x265', 'Blender(-)', '7-Zip Single', '7-Zip',
                       'Geekbench 5.5 Single-Core', 'Geekbench 5.5 Multi-Core', 'WebXPRT 3']

    # Sayısal sütunları temizleme ve eksik değerleri ortalama ile doldurma
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    return df, numeric_columns


def build_and_evaluate_model(df, numeric_columns, target_column):
    # Model için özellikler (features) ve hedef değişkeni (target variable) seçme
    X = df[numeric_columns]
    y = df[target_column]

    # Veriyi eğitim ve test setlerine bölelim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli oluşturup eğitme
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    # Modelin performansını değerlendirme
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model


def show_data_summary(df):
    # Temizlenmiş veri setinin istatistiksel özetini görüntüleme
    print("\nVeri İstatistikleri:")
    print(df.describe())


def plot_histograms(df, numeric_columns):
    # Sayısal sütunların histogramlarını çizme
    df[numeric_columns].hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle('Sayısal Değerlerin Histogramları', y=1.02)
    plt.show()


def plot_boxplots(df, numeric_columns):
    # Sayısal sütunların kutu grafiğini çizme
    fig, axes = plt.subplots(nrows=(len(numeric_columns) + 1) // 2, ncols=2, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5)

    for i, col in enumerate(numeric_columns):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        try:
            print(f"Plotting boxplot for {col}")
            sns.boxplot(x=col, data=df, ax=ax)
            ax.set_title(f'Box Plot - {col}')
            ax.set_xlabel(col)
        except Exception as e:
            print(f"Error plotting boxplot for {col}: {e}")

    plt.show()


def plot_correlation_matrix(df, numeric_columns):
    # Sayısal sütunlar arasındaki korelasyon matrisini oluşturma
    correlation_matrix = df[numeric_columns].corr()

    # Korelasyon matrisini ısı haritası olarak görselleştirme
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.show()


def plot_pairplot(df, numeric_columns):
    # Sayısal sütunlar arasındaki ilişkiyi çizme
    sns.pairplot(df[numeric_columns])
    plt.show()


def plot_pairgrid(df, numeric_columns):
    # Sayısal sütunlar arasındaki ilişkiyi daha detaylı çizme
    g = sns.PairGrid(df[numeric_columns])
    g.map(plt.scatter)
    g.fig.suptitle('Pair Grid - Scatter Plots', y=1.02)  # Başlık ekleme
    plt.show()


def plot_violin_plots(df, numeric_columns):
    # Sayısal sütunların violin grafiğini çizme
    fig, axes = plt.subplots(nrows=len(numeric_columns) // 2, ncols=2, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5)

    for i, col in enumerate(numeric_columns):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        sns.violinplot(y=df[col], ax=ax)
        ax.set_title(f'Violin Plot - {col}')
        ax.set_ylabel(col)

    plt.show()


def plot_categorical_with_hue(df, categorical_column, hue_column):
    # Kategorik sütunu ve renklendirme sütununu kullanarak grafiği çizme
    sns.countplot(x=df[categorical_column], hue=df[hue_column])
    plt.title(f'Categorical Plot with Hue - {categorical_column} vs {hue_column}')
    plt.show()


def plot_bar_chart(df, categorical_columns):
    # Bar grafiğini çizme
    if isinstance(categorical_columns, str):
        # If a single column is provided
        sns.countplot(x=df[categorical_columns])
        plt.title(f'Bar Chart - {categorical_columns}')
        plt.show()
    elif isinstance(categorical_columns, list):
        # If multiple columns are provided
        for col in categorical_columns:
            sns.countplot(x=df[col])
            plt.title(f'Bar Chart - {col}')
            plt.show()
    else:
        raise ValueError(
            "Invalid input for categorical_columns. Pass either a single column name or a list of column names.")


def qq_plot(df, numeric_columns):
    # Quantile-Quantile (QQ) plots çizme
    for col in numeric_columns:
        qqplot(df[col], line='s')  # 's' for standardized line
        plt.title(f'QQ Plot - {col}')
        plt.show()


def scatter_plot(df, x_column, y_column):
    # Scatter plot çizme
    plt.scatter(df[x_column], df[y_column])
    plt.title(f'Scatter Plot - {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


if __name__ == "__main__":
    file_path = 'mobile_cpu.csv'
    categorical_column = 'Model'
    dependent_variable = 'Perf. Rating'
    independent_variables = ['TDP Watt', 'MHz - Turbo', 'Cores / Threads', 'Cinebench R15 CPU Single 64Bit',
                             'Cinebench R15 CPU Multi 64Bit', 'Cinebench R23 Single Core', 'Cinebench R23 Multi Core',
                             'x265', 'Blender(-)', '7-Zip Single', '7-Zip', 'Geekbench 5.5 Single-Core',
                             'Geekbench 5.5 Multi-Core', 'WebXPRT 3']

    df, numeric_columns = load_and_clean_data(file_path)
    print("Veri yüklendi ve temizlendi \n")

    model = build_and_evaluate_model(df, numeric_columns, dependent_variable)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(df[['Perf. Rating']].values.reshape(-1, 1))

    # Dışarıdan veri alıp tahmin yapma
    new_data = pd.DataFrame({
        'TDP Watt': [80],
        'MHz - Turbo': [2500],
        'Cores / Threads': [8],
        'Cinebench R15 CPU Single 64Bit': [100],
        'Cinebench R15 CPU Multi 64Bit': [800],
        'Cinebench R23 Single Core': [120],
        'Cinebench R23 Multi Core': [1000],
        'x265': [150],
        'Blender(-)': [500],
        '7-Zip Single': [400],
        '7-Zip': [2000],
        'Geekbench 5.5 Single-Core': [1200],
        'Geekbench 5.5 Multi-Core': [6000],
        'WebXPRT 3': [250],
        'Perf. Rating': [0]
    })

    new_data_features = new_data[numeric_columns]

    # Tahmin yapma
    prediction_scaled = model.predict(new_data_features)

    prediction_original_scale = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

    print(f"Tahmin Edilen Performans Değeri (Orjinal Ölçekte): {prediction_original_scale} \n")

    # Shapiro-Wilk normalite testini uygula
    shapiro_test(df, numeric_columns)
    print("Shapiro testi tamamlandı \n")

    # Z-puanına göre aykırı değerleri tespit et ve kaldır
    detect_and_remove_outliers_zscore(df, numeric_columns)
    print("Z-puanına göre aykırı veriler tespit edildi ve kaldırıldı \n")

    # IQR yöntemine göre aykırı değerleri tespit et ve kaldır
    detect_and_remove_outliers_iqr(df, numeric_columns)
    print("IQR yöntemine göre aykırı veriler tespit edildi ve kaldırıldı \n")

    # Aykırı değer tespiti ve kaldırma işlemini test et
    test_outlier_detection_and_removal(df, numeric_columns)
    print("Aykırı veriler tespit edildi ve kaldırıldı \n")

    # Çoklu değişken regresyon analizi
    multivariate_regression(df, dependent_variable, independent_variables)
    print("Çoklu değişkene göre regresyon analizi yapıldı \n")

    # Veri setinin istatistiksel özetini gösterme
    show_data_summary(df)
    print("Veri setinin istatistiksel özeti gösterildi \n")

    # Kategorik sütunlar için renkli grafik çizimi
    plot_categorical_with_hue(df, categorical_column, dependent_variable)
    print("Kategorik sütunlar için renkli grafik çizimi yapıldı \n")

    # Sayısal sütunların histogramlarını çizme
    plot_histograms(df, numeric_columns)
    print("Sayısal sütunların histogramları çizildi \n")

    # Sayısal sütunlar arasındaki ilişkiyi çizme
    plot_pairplot(df, numeric_columns)
    print("Sayısal sütunlar arasındaki ilişkiyi çizildi \n")

    # Belirli kategorik sütunlar için bar grafik çizimi
    plot_bar_chart(df, ['TDP Watt', 'MHz - Turbo'])
    print("Belirli kategorik sütunlar için bar grafik çizildi \n")

    # Sayısal sütunlar arasındaki ilişkiyi daha detaylı çizme
    plot_pairgrid(df, numeric_columns)
    print("Sayısal sütunlar arasındaki ilişkiyi daha detaylı çizildi \n")

    # Sayısal sütunların kutu grafiğini çizme
    # plot_boxplots(df, numeric_columns)
    # print("Sayısal sütunların kutu grafiğini çizildi \n")

    qq_plot(df, numeric_columns)

    scatter_plot(df, 'TDP Watt', 'Perf. Rating')
    # Sayısal sütunlar arasındaki korelasyon matrisini çizme
    plot_correlation_matrix(df, numeric_columns)
    print("Sayısal sütunlar arasındaki korelasyon matrisini çizildi \n")

