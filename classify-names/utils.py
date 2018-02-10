import matplotlib.pyplot as plt

def graph_losses(losses, ewma_losses):  # (losses, x_indices, val_losses, val_x_indices):
    plt.figure(1)
    plt.plot(losses, "r", label="Training Loss")
    plt.legend(loc=1)

    plt.figure(2)
    plt.plot(ewma_losses, "g", label="EWMA Loss")
    plt.legend(loc=1)

    # plt.figure(3)
    # plt.plot(x_indices, losses, "r", label="Training Loss")
    # plt.plot(val_x_indices, val_losses, "g", label="Validation Loss")
    # plt.legend(loc=1)
    
    plt.show()
