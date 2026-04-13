import pandas as pd
import matplotlib.pyplot as plt

def line_chart_MSE(data):
    data.plot(x="train_mse", y="epoch", title="Train Mean Squared Error over Epochs")

    plt.xlabel("Train MSE")
    plt.ylabel("Epochs")
    plt.show()

def line_chart_MAE(data):
    data.plot(x="train_mae", y="epoch", title="Train Mean Absolute Error over Epochs")
    plt.xlabel("Train MAE")
    plt.ylabel("Epochs")
    plt.show()


if __name__ == "__main__":
    DATE = "2026-04-08_17-18-03"
    data = pd.read_csv("training_log_" + DATE + ".csv")
    line_chart_MSE(data)
    line_chart_MAE(data)