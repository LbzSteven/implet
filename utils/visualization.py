#based on TSInterpret Implementation

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.data_utils import convert_to_label_if_one_hot
from utils.utils import create_path_if_not_exists


def plot_attribution(item, exp, figsize=(6.4, 4.8), normalize_attribution=True, save_path=None, title=""):
    """
    Plots explanation on the explained Sample.

    Arguments:
        item np.array: instance to be explained,if `mode = time`->`(1,time,feat)`  or `mode = feat`->`(1,feat,time)`.
        exp np.array: explanation, ,if `mode = time`->`(time,feat)`  or `mode = feat`->`(feat,time)`.
        figsize (int,int): desired size of plot.
        heatmap bool: 'True' if only heatmap, otherwise 'False'.
        save str: Path to save figure.
    """
    if len(item[0]) == 1:
        test = item[0]
        # if only one-dimensional input
        fig, (axn, cbar_ax) = plt.subplots(
            len(item[0]), 2, sharex=False, sharey=False, figsize=figsize, gridspec_kw={'width_ratios': [40, 1]},
        )

        # Shahbaz: Set color pallete such that negative is red and positive is blue
        my_cmap = sns.diverging_palette(260, 10, as_cmap=True)

        #set min and max to have same absolute values
        extremum = np.max(abs(exp))

        # cbar_ax = fig.add_axes([.91, .3, .03, .4])
        axn012 = axn.twinx()
        if normalize_attribution:
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap=my_cmap,
                ax=axn,
                yticklabels=False,
                vmin=-1*extremum,
                vmax=extremum,
                cbar_ax=cbar_ax,
                # cbar_kws={"orientation": "vertical"},
            )
        else:
            sns.heatmap(
                exp.reshape(1, -1),
                fmt="g",
                cmap="viridis",
                ax=axn,
                yticklabels=False,
                vmin=0,
                vmax=1,
            )
        sns.lineplot(
            x=np.arange(0, len(item[0][0].reshape(-1))) + 0.5,
            y=item[0][0].flatten(),
            ax=axn012,
            color="black",
        )
        # plt.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.95, top=0.95, bottom=0.05)
        cbar_ax.tick_params(labelsize=10)
        plt.title(title)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    else:
        ax011 = []

        fig, axn = plt.subplots(
            len(item[0]), 1, sharex=True, sharey=True, figsize=figsize
        )
        cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

        for channel in item[0]:
            # print(item.shape)
            # ax011.append(plt.subplot(len(item[0]),1,i+1))
            # ax012.append(ax011[i].twinx())
            # ax011[i].set_facecolor("#440154FF")
            axn012 = axn[i].twinx()
            if normalize_attribution:

                sns.heatmap(
                    exp[i].reshape(1, -1),
                    fmt="g",
                    cmap="viridis",
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    ax=axn[i],
                    yticklabels=False,
                    vmin=np.min(exp),
                    vmax=np.max(exp),
                )
            else:
                sns.heatmap(
                    exp[i].reshape(1, -1),
                    fmt="g",
                    cmap="viridis",
                    cbar=i == 0,
                    cbar_ax=None if i else cbar_ax,
                    ax=axn[i],
                    yticklabels=False,
                    vmin=0,
                    vmax=1,
                )

            sns.lineplot(
                x=range(0, len(channel.reshape(-1))),
                y=channel.flatten(),
                ax=axn012,
                color="white",
            )
            plt.xlabel("Time", fontweight="bold", fontsize="large")
            plt.ylabel(f"Feature {i}", fontweight="bold", fontsize="large")
            i = i + 1
        fig.tight_layout(rect=[0, 0, 0.9, 1])
    if save_path is not None:
        path_only = save_path.rsplit("/", 1)[0] + "/"

        create_path_if_not_exists(path_only)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_multiple_images(data, one_hot, n, shape, figsize=(12, 6)):
    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
    axes = axes.flatten()

    colors = ['b', 'orange', 'r', 'c', 'm', 'y', 'k']  # Customize as needed

    label_to_color = {}
    labels = convert_to_label_if_one_hot(one_hot)
    for i in range(n):
        if i >= len(axes):
            break  # Stop if n exceeds the available subplots

        label = labels[i]
        if label not in label_to_color:
            label_to_color[label] = colors[len(label_to_color) % len(colors)]

        color = label_to_color[label]

        a = data[i].copy().reshape(1, 1, -1).flatten()
        axes[i].plot(range(len(a)), a, color=color)
        axes[i].set_title(f"Sample {i + 1}, Label: {label}")

    for ax in axes[n:]:
        ax.axis('off')

    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in label_to_color.values()]
    labels = [f'Label: {l}' for l in label_to_color.keys()]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

    # Or alternatively, to display the legend outside the grid:
    # fig.legend(handles, labels, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_multiple_images_with_attribution(test_x, pred_y, n, shape, figsize=(12, 6), use_attribution=False,
                                          attributions=None, normalize_attribution=True, title="", save_path=None,
                                          test_y=None):
    """
    Plots multiple subplots with an option to include attribution heatmaps.

    Arguments:
        test_x np.array: Input data for the images.
        test_y np.array: Labels for the images.
        n int: Number of images to display.
        shape tuple: Tuple indicating the number of rows and columns (rows, columns).
        use_attribution bool: Whether to plot attribution heatmaps.
        attributions np.array: The attributions if use_attribution is True (same shape as test_x).
        normalize_attribution bool: Whether to normalize attributions to a common scale.
        title str: Title of the figure.
        save_path: Save the image if given the path
        :param test_y the GT label for the input:
    """
    fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Define different colors for different labels
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Customize as needed
    label_to_color = {}

    # Create a color palette for the heatmap if attributions are used
    my_cmap = sns.diverging_palette(260, 10, as_cmap=True)
    if test_y is not None:
        test_y = convert_to_label_if_one_hot(test_y)
    pred_y_labels = convert_to_label_if_one_hot(pred_y)
    for i in range(n):
        if i >= len(axes):
            break
        label = pred_y_labels[i]
        if label not in label_to_color:
            label_to_color[label] = colors[len(label_to_color) % len(colors)]
        color = label_to_color[label]

        length = test_x.shape[-1]

        # Plot the data
        channel_data = test_x[i].copy().reshape(1, 1, -1).flatten()
        axes[i].plot(range(len(channel_data)), channel_data, color=color)
        # axes[i].set_xticks([])
        axes[i].tick_params(axis='x', rotation=90)
        # Add attribution heatmap if use_attribution is True
        if use_attribution and attributions is not None:
            axn = axes[i].twinx()  # Create a secondary axis for the heatmap
            extremum = np.max(abs(attributions[i])) if normalize_attribution else None
            sns.heatmap(
                attributions[i].reshape(1, -1),
                fmt="g",
                cmap=my_cmap if normalize_attribution else "viridis",
                ax=axn,
                yticklabels=False,
                vmin=-extremum if extremum else 0,
                vmax=extremum if extremum else 1,
                cbar=True,  # Suppress color bar in individual subplots
                alpha=0.5  # Add transparency to the heatmap
            )
            # axn.set_yticks([])
            # axn.set_xticks(np.arange(0, 100, 10))

        if test_y is not None:
            axes[i].set_title(f"Image {i + 1}, Predicted: {label}|GT:{test_y[i]}")
        else:
            axes[i].set_title(f"Image {i + 1}, Predicted: {label}")
    # Add a global color bar if attributions are used
    # if use_attribution:
    #     cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    #     sm = plt.cm.ScalarMappable(cmap=my_cmap if normalize_attribution else "viridis")
    # sm.set_array([])
    # fig.colorbar(sm, cax=cbar_ax)

    # Create a custom legend for labels
    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in label_to_color.values()]
    labels = [f'Label: {l}' for l in label_to_color.keys()]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.suptitle(title)
    if save_path is not None:
        path_only = save_path.rsplit("/", 1)[0] + "/"
        if not os.path.exists(path_only):
            os.makedirs(path_only)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()