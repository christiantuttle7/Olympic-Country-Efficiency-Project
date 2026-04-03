"""Creating basic visualization using the dataset
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    """
    olympicdataframe = pd.read_csv('olympic_countries_efficiency.csv')
    sns.histplot(data=olympicdataframe[country -- "USA"], x="total_medals")
    plt.tight_layout()
    plt.savefig("hist_total_medals_USA.png", dpi=200)
    """
    olympicdataframe = pd.read_csv('olympic_countries_efficiency.csv')
    print(olympicdataframe)


if __name__ == "__main__":
    main()
