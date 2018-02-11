import matplotlib.pyplot as plt

def losses_to_ewma(losses, alpha = 0.98):
    losses_ewma = []
    ewma = losses[0]
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma


def graph_losses(losses):  # (losses, x_indices, val_losses, val_x_indices):
    plt.figure(1)
    plt.plot(losses, "r", label="Training Loss")
    plt.legend(loc=1)

    ewma_losses = losses_to_ewma(losses, alpha=0.99)
    plt.figure(2)
    plt.plot(ewma_losses, "g", label="EWMA Loss")
    plt.legend(loc=1)

    # plt.figure(3)
    # plt.plot(x_indices, losses, "r", label="Training Loss")
    # plt.plot(val_x_indices, val_losses, "g", label="Validation Loss")
    # plt.legend(loc=1)
    
    plt.show()
