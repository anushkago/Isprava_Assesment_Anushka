import matplotlib.pyplot as plt
import seaborn as sns

def plot_category_bar(series):
    fig, ax = plt.subplots(figsize=(7, 7))
    bars = ax.bar(series.index, series.values, color="#3498db", edgecolor="black")
    ax.set_title("Spend by Category", fontsize=18, fontweight="bold")
    ax.set_ylabel("Amount", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)
    ax.set_xticklabels(series.index, rotation=30, fontsize=12)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{int(y)}" for y in ax.get_yticks()], fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Add value labels:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
    

def plot_category_pie(series):
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        series, 
        labels=series.index, 
        autopct="%1.1f%%",
        startangle=140,
        textprops={'fontsize': 13}
    )
    ax.set_title("Spend Share by Category", fontsize=18, fontweight="bold")
    plt.setp(autotexts, size=13, weight="bold", color="white")
    plt.setp(texts, size=14)
    ax.axis('equal')
    plt.tight_layout()
    return fig


