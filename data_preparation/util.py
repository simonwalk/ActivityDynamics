from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import sys
import pandas as pd
import os
import datetime
import numpy as np
from graph_tool.all import *
import time
from signal import signal, alarm, SIGALRM


plt.rcParams['figure.figsize'] = 16, 9


def print_f(*kwargs):
    print datetime.datetime.now().replace(microsecond=0).time(),
    for i in kwargs:
        print str(i),
    print ''
    sys.stdout.flush()


def read_graph(filename, activity_threshold=None, largest_component=False, force_preprocessing=False):
    print_f('load weighted graph:', filename)
    stored_graph_name = get_graph_filename(filename, activity_threshold, largest_component)
    if os.path.isfile(stored_graph_name):
        try:
            print_f('graph already pre-processed. load from:', stored_graph_name)
            graph = load_graph(stored_graph_name)
        except:
            print_f('failed loading pre-processed graph from:', stored_graph_name)
            print_f('start new pre-processing...')
            return read_graph(filename, activity_threshold, largest_component, force_preprocessing=True)
    else:
        graph = load_graph(filename)
        print_f('orig graph contains', graph.num_vertices(), 'nodes')
        if activity_threshold is not None and activity_threshold > 1:
            print_f('filter activity >=', activity_threshold)
            graph = GraphView(graph, efilt=lambda e: graph.edge_properties["activity"][e] >= activity_threshold)
        if largest_component:
            print_f('filter largest component')
            l = label_largest_component(graph)
            graph = GraphView(graph, vfilt=l)
        if stored_graph_name != filename:
            print_f('store pre-processed graph to:', stored_graph_name)
            graph.purge_vertices()
            graph.purge_edges()
            graph.save(stored_graph_name)
    return graph


def get_graph_filename(filename, activity_threshold=None, largest_component=False):
    filename, ext = filename.rsplit('.', 1)
    if not activity_threshold is None:
        filename += '_a' + str(activity_threshold)
    if largest_component:
        filename += '_lc'
    return filename + '.' + ext


def read_log_to_df(file_name):
    if os.path.isfile(file_name + '.ser'):
        file_name += '.ser'
        print_f('read log from:', file_name)
        data_frame = pd.read_pickle(file_name)
    else:
        print_f('read log from:', file_name)
        data_frame = pd.read_csv(file_name, header=None, names=['timestamp', 'source', 'destination'],
                                 infer_datetime_format=True, parse_dates=['timestamp'], sep='\t', comment='#',
                                 dtype={'timestamp': datetime.datetime, 'source': np.float, 'destination': np.float})
        print_f('convert and sort timestamps')

        print data_frame
        data_frame.sort('timestamp', inplace=True)
        print_f('serialize data to:', file_name + '.ser')
        data_frame.to_pickle(file_name + '.ser')
    return data_frame


def plot(stats=None, users=None, logy=False, x_label=None, y_label=None, second_y_label=None, filename='plot.png', custom_labels=None, second_y_axis=False):
    stats = stats.copy()
    if isinstance(custom_labels, str):
        custom_labels = [custom_labels]
    fig, ax = plt.subplots()
    if not users is None:
        print_f('plot user lines')
        users.plot(legend=False, ax=ax, label=False, alpha=0.4, logy=logy)
        print_f('done')
        ax.legend().set_visible(False)
    if not stats is None:
        if isinstance(stats, pd.Series):
            stats = pd.DataFrame(columns=['values' if custom_labels is None else custom_labels[0]], data=stats)
        else:
            stats.columns = stats.columns if custom_labels is None else custom_labels[:len(stats.columns)] + (
                [] if len(stats.columns) < len(custom_labels) else list(stats.columns)[len(custom_labels):])
        if not isinstance(second_y_axis, bool) or second_y_axis is True:
            if isinstance(second_y_axis, str):
                second_y_axis = [second_y_axis]
            elif isinstance(second_y_axis, bool):
                second_y_axis = stats.columns[int((len(stats.columns) + 1) / 2):]
            first_y_axis = [i for i in stats.columns if not i in second_y_axis]
            ax = stats.plot(y=first_y_axis, linewidth=3, logy=logy)
            ax2 = stats.plot(y=second_y_axis, secondary_y=True, linewidth=3, ax=ax, logy=logy)
            ax2.legend(loc=1)
        else:
            stats.plot(linewidth=3, ax=ax, logy=logy)
        plot_labels = stats.columns
    else:
        plot_labels = users.columns if not users is None else []
    if not y_label is None:
        ax.set_ylabel(y_label)
    if not second_y_label is None:
        ax.right_ax.set_ylabel(second_y_label)
    if not x_label is None:
        ax.set_xlabel(x_label)
    filename = filename if filename.endswith('.png') else filename + '.png'
    handles, labels = ax.get_legend_handles_labels()
    # print 'orig labels', labels
    legend_labels = {j if (custom_labels is None or len(custom_labels) <= i) else custom_labels[i] for i, j in enumerate(plot_labels)}
    # print 'set labels', legend_labels
    handles, labels = zip(*[(i, j) for i, j in zip(handles, labels) if j in legend_labels])
    # print 'filtered labels', labels
    ax.legend(handles, labels)
    if not isinstance(second_y_axis, bool) or second_y_axis is True:
        ax.legend(loc=2)
    print_f('save fig:', filename)
    plt.savefig(filename, dpi=150)
    print_f('done')
    plt.close('all')


def print_dict_table(data, cols=None, rows=None, index_name='', shorten_values=-1, values_unit='', empty_val='-', title=None, cell_spacing_char=' ', spacing=4):
    if isinstance(data, dict) and all([isinstance(i, dict) for i in data.values()]):
        cols = sorted(data.keys()) if cols is None else cols
        cols = [str(i) for i in cols]
        rows = sorted(set(j for i in data.values() for j in i.keys())) if rows is None else rows
        col_width = {col_name: len(col_name) + spacing for col_name in cols}
        col_width[index_name] = max(max([len(i) + spacing for i in rows]), len(index_name) + spacing)
        table_data = []
        for col_name in cols:
            for row_id, row_name in enumerate(rows):
                try:
                    value = str(data[col_name][row_name])[:shorten_values] + values_unit
                except:
                    value = str(empty_val)
                col_width[col_name] = max(col_width[col_name], len(value) + spacing)
                if len(table_data) <= row_id:
                    table_data.append([])
                table_data[row_id].append(value)
        for col_id, col_name in enumerate(cols):
            for row_id, row_name in enumerate(rows):
                table_data[row_id][col_id] = str(table_data[row_id][col_id]).center(col_width[col_name], cell_spacing_char[0])
        table_width = sum(col_width.values()) + len(col_width) - 1
        if not title is None:
            print str(title).center(table_width, '=')
        print_line = index_name.center(col_width[index_name])
        for col_name in cols:
            print_line += '|' + col_name.center(col_width[col_name])
        print print_line
        print_line = '-' * col_width[index_name]
        for idx, col_name in enumerate(cols):
            print_line += '+'
            print_line += '-' * col_width[col_name]
        print print_line
        for row_idx, i in enumerate(table_data):
            print_line = rows[row_idx].center(col_width[index_name])
            for col_idx, colname in enumerate(cols):
                print_line += '|' + table_data[row_idx][col_idx]
            print print_line
        print '=' * table_width


def raise_prompt_timeout():
    raise PromptTimeout


def prompt_with_timeout(message, timeout=None, default_val=None, target_data_type=None):
    signal(SIGALRM, lambda x, y: raise_prompt_timeout())
    result = default_val
    try:
        if not timeout is None:
            alarm(timeout)
            print '\n\tInput Timeout:', timeout, 'seconds. Press ENTER to stop timeout.'
            print '\tdefault val:', default_val
            result = raw_input(message)
            alarm(0)
            if result == '':
                print '\n\ttimeout stopped...waiting for input...'
                result = raw_input(message)
    except PromptTimeout:
        result = default_val
        print ''
        print_f('\tinput timeout: Using default val:', default_val)
        time.sleep(3)
    if not target_data_type is None:
        try:
            result = target_data_type(result)
        except:
            print ''
            print_f('\tparsing input error. will use default val:', default_val)
            result = default_val
            time.sleep(3)
    return result


class PromptTimeout(Exception):
    pass


def add_rolling_means(data, winsize=None, col_name=0):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(columns=[col_name], data=data)
    non_nan_data = np.sum(np.invert(np.isnan(data[col_name])))
    if winsize is None:
        winsize = 0.1
    if not isinstance(winsize, list):
        winsize = [winsize]
    winsize = [int(i) if i >= 1 else int(round(non_nan_data * i)) for i in winsize]
    for i in winsize:
        data['rolling mean ' + str(i)] = pd.rolling_mean(data[col_name], window=i)
    return data


def add_poly_fit(data, deg=2, col_name=0, predict=10, best=2):
    prefix = 'poly fit '
    if isinstance(data, pd.Series):
        data = pd.DataFrame(columns=[col_name], data=data)
    value_idx = np.invert(np.isnan(data[col_name]))
    x = list(data.index[value_idx])
    y = list(data[col_name][value_idx])
    y_mean = np.mean(y)
    y_std = np.std(y)
    trans_y_back = lambda x: [(i * y_std) + y_mean for i in x]
    y = (np.array(y) - y_mean) / y_std
    if len(x) != len(y) or len(x) < 2:
        print_f('WARNING: poly fit got no input')
        print_f('len x:', len(x))
        print_f('len y:', len(y))
        return
    used_mapping = False
    data_index_set = set(data.index)
    if all((isinstance(i, datetime.date) for i in x)):
        used_mapping = True
        x = [i.toordinal() for i in x]

    diffs = [x[i + 1] - x[i] for i in xrange(len(x) - 1)]
    diff = np.min(diffs) / 2
    predict_idx = list(x)
    if isinstance(predict, int) and predict > 0:
        for i in xrange(predict * 2):
            predict_idx.append(predict_idx[-1] + diff)
    if not isinstance(deg, list):
        deg = [deg]
    aic_values = np.zeros(len(deg))
    aic = lambda rss, n, k: n * np.log(float(rss) / n) + 2 * k
    for deg_idx, current_degree in enumerate(deg):
        data_set_name = prefix + str(current_degree)
        if isinstance(best, int) and 0 < best < len(deg):
            fitted_function, residuals, rank, singular_values, rcond = np.polyfit(x, y, current_degree, full=True)
            aic_values[deg_idx] = aic((residuals).sum(), len(x), current_degree + 1)
        else:
            fitted_function = np.polyfit(x, y, current_degree)
        fitted_function = np.poly1d(fitted_function)
        predicted_y = trans_y_back(np.polyval(fitted_function, predict_idx))
        if used_mapping:
            tmp_idx = [datetime.date.fromordinal(int(round(i))) for i in predict_idx]
        else:
            tmp_idx = predict_idx
        fitted_frame = pd.Series(data=predicted_y, index=tmp_idx)
        all_index = sorted(data_index_set | set(tmp_idx))
        data = data.reindex(all_index)
        data[data_set_name] = fitted_frame
    if isinstance(best, int) and 0 < best < len(deg):
        print_f('aic_values', aic_values)
        filtered_deg, aic_values = zip(*filter(lambda x: not np.isinf(x[1]), zip(deg, aic_values)))
        aic_values = np.array(aic_values)
        print_f('filtered aic_values', aic_values)
        print_f('filteder deg:', filtered_deg)
        aic_trans = np.exp(-0.5 * aic_values)
        aic_prob = aic_trans / aic_trans.sum()
        print_f('aic_probs', aic_prob)
        sorted_deg, sorted_aic = zip(*sorted(zip(filtered_deg, aic_prob), key=lambda x: x[1], reverse=True))
        best_data = [data[prefix + str(i)] for i in sorted_deg[:best]]
        for i in deg:
            col_name_to_drop = prefix + str(i)
            data.drop(col_name_to_drop, axis=1, inplace=True)
        for idx, i in enumerate(sorted_deg[:best]):
            col_name = prefix + str(i)
            data[col_name] = best_data[idx]
    return data


if __name__ == '__main__':
    pass