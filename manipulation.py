import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    fig, axes = plt.subplots(nrows=len(numeric_columns) // 2, ncols=2, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5)

    for i, col in enumerate(numeric_columns):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Box Plot - {col}')
        ax.set_xlabel(col)

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
        raise ValueError("Invalid input for categorical_columns. Pass either a single column name or a list of column names.")


if __name__ == "__main__":
    file_path = 'mobile_cpu.csv'

    # Veriyi yükle ve temizle
    df, numeric_columns = load_and_clean_data(file_path)

    # Veri özetini görüntüle
    show_data_summary(df)

    # Histogramları çiz
    plot_histograms(df, numeric_columns)

    plot_pairplot(df, numeric_columns)

    plot_bar_chart(df, ['TDP Watt', 'MHz - Turbo'])

    plot_pairgrid(df,numeric_columns)
    # Korelasyon matrisini görselleştir
    plot_correlation_matrix(df, numeric_columns)
