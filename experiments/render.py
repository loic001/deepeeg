# class Renderer(object):
#     def __init(self, results):
#         self.results = results
#
#     def __call__(self):
#         raise NotImplementedError()
#
# class TrainResultRenderer(Renderer):
#     def __call__(self):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def build_train_figure(name, train_result, fontsize):
    fig = plt.figure()
    fig.suptitle(name, fontsize=fontsize)

    train_grid = gridspec.GridSpec(2, 4)
    history = train_result.get('net_history', None)
    if history:
        length_history = len(history[:, 'train_loss'])
        # len_history = len(history)
        train_loss = history[:, 'train_loss'], np.arange(1, length_history+1)
        valid_loss = history[:, 'valid_loss'], np.arange(1, length_history+1)
        roc_auc = history[:, 'roc_auc'], np.arange(1, len(history)+1)
        valid_acc = history[:, 'valid_acc'], np.arange(1, length_history+1)

        valid_loss_np = np.array(valid_loss[0])
        best_epoch = np.argmin(valid_loss_np) + 1

        train_loss_until_best = train_loss[0][:best_epoch], np.arange(1, best_epoch+1)
        valid_loss_until_best = valid_loss[0][:best_epoch], np.arange(1, best_epoch+1)
        valid_acc_until_best = valid_acc[0][:best_epoch], np.arange(1, best_epoch+1)
        roc_auc_until_best = roc_auc[0][:best_epoch], np.arange(1, best_epoch+1)


        train_loss_ax = fig.add_subplot(train_grid[0, 0])
        valid_loss_ax = fig.add_subplot(train_grid[0, 1])
        roc_auc_ax = fig.add_subplot(train_grid[0, 2])
        valid_acc_ax = fig.add_subplot(train_grid[0, 3])

        train_loss_until_best_ax = fig.add_subplot(train_grid[1, 0])
        valid_loss_until_best_ax = fig.add_subplot(train_grid[1, 1])
        roc_auc_until_best_ax = fig.add_subplot(train_grid[1, 2])
        valid_acc_until_best_ax = fig.add_subplot(train_grid[1, 3])

        # plot train results
        simple_plot(train_loss_until_best, 'Train Loss', x_label='epochs',
                    y_label='nll loss', ax=train_loss_ax, fontsize=fontsize)
        simple_plot(valid_loss_until_best, 'Valid Loss', x_label='epochs',
                    y_label='nll loss', ax=valid_loss_ax, fontsize=fontsize)
        simple_plot(roc_auc_until_best, 'Roc Auc', x_label='epochs',
                    y_label='', ax=roc_auc_ax, fontsize=fontsize)
        simple_plot(valid_acc_until_best, 'Valid Acc', x_label='epochs',
                    y_label='valid acc', ax=valid_acc_ax, fontsize=fontsize)

        # simple_plot(train_loss, 'Train Loss', x_label='epochs',
        #             y_label='nll loss', ax=train_loss_until_best_ax, fontsize=fontsize)
        # simple_plot(valid_loss, 'Valid Loss', x_label='epochs',
        #             y_label='nll loss', ax=valid_loss_until_best_ax, fontsize=fontsize)
        # simple_plot(roc_auc, 'Roc Auc', x_label='epochs', y_label='',
        #             ax=roc_auc_until_best_ax, fontsize=fontsize)
        # simple_plot(valid_acc, 'Valid Acc', x_label='epochs',
        #             y_label='valid acc', ax=valid_acc_until_best_ax, fontsize=fontsize)

        fig.subplots_adjust(wspace=0.5, hspace=0.5)

    return fig

def compute_spelling_test_acc(group_by_item_df):
    data = group_by_item_df
    nb_iterations = len(data['predicted_item_iteration'][0])
    accuracies = []
    for x_iter in range(nb_iterations):
        acc = np.mean(data['predicted_item_iteration'].apply(
            lambda x: x[x_iter]) == data['true_item'])
        accuracies.append({'after_x_rep': x_iter + 1, 'acc': acc})
    return accuracies

def build_spelling_test_figure(name, test_spelling_result, fontsize):
    fig = plt.figure()
    fig.suptitle(name, fontsize=fontsize)
    test_spelling_grid = gridspec.GridSpec(2, 2)

    acc_iter_ax = fig.add_subplot(test_spelling_grid[0, 0])
    grouped_by_item = test_spelling_result['group_by_item']
    group_by_item_df = pd.DataFrame(grouped_by_item)
    print(group_by_item_df[['predicted_item_iteration', 'true_item']])

    acc_arr = compute_spelling_test_acc(group_by_item_df)
    bar_plot_arr(acc_arr, 'Acc after N repetitions', x_key='after_x_rep',
                 y_key='acc', ax=acc_iter_ax, fontsize=fontsize)

    fig.subplots_adjust(wspace=0.7, hspace=0.8)
    fig.tight_layout(pad=5)
    return fig
    # data['on_first'] = data['predicted_item_iteration']
    # print(data[['predicted_item_iteration', 'true_item']])

    # print(data[['predicted_item_iteration', 'acc_after_0_iter']])

    # grouped_by_true_item = group_by_key(grouped_by_item, 'true_item')
    #
    # for item in grouped_by_item:
    #     print(item['true_item'])
    # for item, value in grouped_by_true_item.items():
    #     print('{} {}'.format(item, len(value)))
    # print(grouped_by_true_item.keys())

def build_test_figure(name, test_result, test_result_on_train, fontsize):
    fig = plt.figure()
    fig.suptitle(name, fontsize=fontsize)
    test_grid = gridspec.GridSpec(2, 2)
    confusion_matrix_ax = fig.add_subplot(test_grid[0, 0])
    confusion_matrix_norm_ax = fig.add_subplot(test_grid[0, 1])

    confusion_matrix_on_train_ax = fig.add_subplot(test_grid[1, 0])
    confusion_matrix_norm_on_train_ax = fig.add_subplot(test_grid[1, 1])

    confusion_matrix_data = test_result.get('confusion_matrix', None)
    if confusion_matrix_data is not None:
        heatmap = confusion_matrix_heatmap(confusion_matrix_data, '', [
                                           'Non-Target', 'Target'], ax=confusion_matrix_ax, fontsize=fontsize)
        heatmap_norm = confusion_matrix_heatmap(confusion_matrix_data, '', [
                                                'Non-Target', 'Target'], ax=confusion_matrix_norm_ax, fontsize=fontsize, norm=True)

    confusion_matrix_data_on_train = test_result_on_train.get(
        'confusion_matrix', None)
    if confusion_matrix_data_on_train is not None:
        heatmap = confusion_matrix_heatmap(confusion_matrix_data_on_train, '', [
                                           'Non-Target', 'Target'], ax=confusion_matrix_on_train_ax, fontsize=fontsize)
        heatmap_norm = confusion_matrix_heatmap(confusion_matrix_data_on_train, '', [
                                                'Non-Target', 'Target'], ax=confusion_matrix_norm_on_train_ax, fontsize=fontsize, norm=True)
    fig.subplots_adjust(wspace=0.7, hspace=0.8)
    fig.tight_layout(pad=5)
    return fig


def simple_plot(data, name, ax, x_label="", y_label="", fontsize=10):
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    if isinstance(data, tuple):
        X=data[1]
        y=data[0]
    else:
        X=np.arange(len(data))
        y=data
    ax.plot(X, y)


def bar_plot_arr(arr, name, x_key='x', y_key='y', x_label=None, y_label=None, ax=None, fontsize=10):
    return bar_plot_df(pd.DataFrame(arr), name, x_key, y_key, x_label, y_label, ax, fontsize)


def bar_plot_df(df, name, x_key='x', y_key='y', x_label=None, y_label=None, ax=None, fontsize=10):
    bar_plot = sns.barplot(x=x_key, y=y_key, data=df,
                           ax=ax, color=sns.xkcd_rgb["denim blue"])
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', rotation=0, xytext=(0, 2), textcoords='offset points', fontsize=4)
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)


def confusion_matrix_heatmap(confusion_matrix, name, class_names, ax, fontsize=10, norm=False):
    if norm:
        confusion_matrix = confusion_matrix.astype(
            'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    title = 'CM {}'.format(name)
    title = 'Normalized ' + title if norm else title
    ax.set_title(title, fontsize=fontsize)
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names)
    fmt = ".3f" if norm else "d"
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt,
                          ax=ax, annot_kws={"size": 5})
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
