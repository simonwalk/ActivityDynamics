__author__ = 'Simon Walk, Florian Geigl, Denis Helic, Philipp Koncar'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

import config.config as config

import os as os
import math
import random
from lib.network import *
from collections import defaultdict
import datetime
from scipy.stats import sem
import numpy as np
import scipy as sp
from decimal import *
from textwrap import wrap
import pandas as pd

DEBUG_LEVEL = 0

def get_network_details(graph_name, run=0, path=config.graph_binary_dir):
    graph = load_graph(path + "GT/" + graph_name + "/" + graph_name + "_run_" + str(run) + ".gt")
    ratios = graph.graph_properties["ratios"]
    deltataus = graph.graph_properties["deltatau"]
    deltapsi = graph.graph_properties["deltapsi"]
    ew = graph.graph_properties["top_eigenvalues"]
    random_inits = graph.graph_properties["runs"]
    store_iterations = graph.graph_properties["store_iterations"]
    try:
        a_cs = graph.graph_properties["a_cs"]
        cas = graph.graph_properties["activity_per_month"]
    except:
        debug_msg(" --> Could not load empirical data!")
        cas = []
        a_cs = []
    return graph, ratios, deltataus, deltapsi, graph_name, store_iterations, cas, ew, random_inits, a_cs


def prepare_folders():
    flist = [config.graph_binary_dir,
             config.graph_binary_dir + "GT/",
             config.graph_binary_dir + "ratios/",
             config.graph_binary_dir + "empirical_input/",
             config.graph_source_dir + "empirical_results/",
             config.graph_source_dir,
             config.graph_source_dir + "weights",
             config.graph_dir,
             config.plot_dir,
             config.plot_dir + "eigenvalues",
             config.plot_dir + "functions",
             config.plot_dir + "scatterplots",
             config.plot_dir + "weights_over_time",
             config.plot_dir + "average_weights_over_tau",
             config.plot_dir + "ratios_over_time",
             config.plot_dir + "empirical_results"]

    for fname in flist:
        create_folder(fname)


def create_folder(folder):
    if not os.path.exists(folder):
        debug_msg("Created folder: {}".format(folder))
        os.makedirs(folder)


def get_weights_fn(store_iterations, deltatau, run, graph_name, ratio):
    return config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + \
           "_" + str(store_iterations).replace(".", "") + "_" + \
           str(deltatau).replace(".", "") + "_" + str(ratio).replace(".", "") + "_run_" + str(run) + "_weights.txt"

def get_intrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratio):
    return config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + \
           "_" + str(store_iterations).replace(".", "") + "_" + \
           str(deltatau).replace(".", "") + "_" + str(ratio).replace(".", "") + "_run_" + str(run) + "_intrinsic.txt"

def get_extrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratio):
    return config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + \
           "_" + str(store_iterations).replace(".", "") + "_" + \
           str(deltatau).replace(".", "") + "_" + str(ratio).replace(".", "") + "_run_" + str(run) + "_extrinsic.txt"


def empirical_result_plot(graph_name, run=0):
    import subprocess
    import os

    source_path = os.path.abspath(config.graph_source_dir + "empirical_results/" + graph_name + ".txt")
    out_file = open(source_path, "wb")

    debug_msg("  >>> Creating empirical result plot", level=0)
    graph, ratios, deltatau, deltapsi, graph_name, store_iterations, \
    cas, ew, random_inits, a_cs = get_network_details(graph_name, run)

    pipe_vals = [str(deltatau), "%.4f" % deltapsi, "%.2f" % a_cs[0], "%.2f" % ew[0]]

    debug_msg("    ** Preparing Ratios", level=0)
    x_ratios = [(float(x)+1) for x in range(len(ratios))]
    y_ratios = ratios
    out_file.write(("\t").join(["%.8f" % x for x in x_ratios])+"\n")
    out_file.write(("\t").join(["%.8f" % x for x in y_ratios])+"\n")
    debug_msg("    ** Preparing Average Activities", level=0)
    fpath = get_weights_fn(store_iterations, deltatau, run, graph_name, ratios[0])
    epath = os.path.abspath(get_extrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratios[0]))
    ipath = os.path.abspath(get_intrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratios[0]))
    x_activity = []
    y_activity = []
    pf = open(fpath, "rb")
    for lidx, l in enumerate(pf):
        x_activity.append((float(lidx)+1)*deltatau/deltapsi*store_iterations)
        y_activity.append(sum([float(x) for x in l.split("\t")]))
    pf.close()
    out_file.write(("\t").join(["%.8f" % x for x in x_activity])+"\n")
    out_file.write(("\t").join(["%.8f" % x for x in y_activity])+"\n")

    debug_msg("    ** Preparing Real Activities", level=0)
    y_real_act = cas
    x_real_act = range(len(cas))
    out_file.write(("\t").join(["%.8f" % x for x in x_real_act])+"\n")
    out_file.write(("\t").join(["%.8f" % x for x in y_real_act])+"\n")
    out_file.close()
    debug_msg(" ---- Calling empirical_plots.R ---- ", level=0)
    r_script_path = os.path.abspath(config.r_dir + 'empirical_plots.R')
    wd = r_script_path.replace("DynamicNetworks/R Scripts/empirical_plots.R", "") + \
         "DynamicNetworksResults/plots/empirical_results/"
    subprocess.call(['/usr/bin/RScript', r_script_path, wd, source_path, pipe_vals[0], pipe_vals[1], pipe_vals[2],
                     pipe_vals[3], graph_name, ipath, epath])#, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
    debug_msg(" ---- Done ---- ", level=0)


def plot_empirical_ratios(graph_name, run=0):
    debug_msg("  >>> Drawing Ratios over Tau", level=0)
    graph, ratios, deltatau, deltapsi, graph_name, store_iterations, cas, ew, random_inits, a_cs = get_network_details(graph_name, run)
    storage_folder = config.plot_dir + "ratios_over_time/" + graph_name + "/"
    debug_msg("    ++ Processing graph = {}, deltatau = {}, deltapsi = {}".format(graph_name, deltatau, deltapsi),
              level=0)
    x_vals = [(float(x)+1) for x in range(len(ratios))]
    #x_vals.append((float(lidx)+1)*deltatau/deltapsi*store_iterations)
    #x_vals = [((float(x)+1)*store_iterations) for x in range(len(ratios))]
    y_vals = ratios
    plt.plot(x_vals, y_vals, alpha=0.5)
    max_x = max(x_vals)
    plt.plot([0, max_x], [ew[0], ew[0]], 'k-', lw=1)
    plt.title("Ratio " +
              r"$(\frac{\lambda}{\mu})$" +
              " over " +
              r"$\tau$" +
              "\n" +
              r"$\Delta\tau$"+"={}, ".format(deltatau) +
              r"$\Delta\psi$"+"={}, ".format("%.4f" % deltapsi) +
              r"$a_c$"+"={}".format("%.2f" % a_cs[0]) +
              r", $\kappa_1$={}".format("%.2f" % ew[0])
    )
    plt.xlabel(r"$\tau$ (in Months)")
    plt.ylabel("Ratio "+ r"$(\frac{\lambda}{\mu})$")
    ax = plt.axes()
    ax.grid(color="gray")
    plt.savefig(storage_folder + graph_name + "_{}_{}_empirical_ratios.png".format(store_iterations,
                                                                                   str(deltatau).replace(".", "")))
    plt.close("all")


def avrg_activity_over_tau_empirical(graph_name, run=0):
    debug_msg("  >>> Drawing Average Activity over Tau", level=0)

    graph, ratios, deltatau, deltapsi, graph_name, store_iterations, cas, \
    ew, random_inits, a_cs = get_network_details(graph_name, run)

    storage_folder = config.plot_dir + "average_weights_over_tau/" + graph_name + "/"
    debug_msg("    ++ Processing graph = {}, deltatau = {}, deltapsi = {}".format(graph_name, deltatau, deltapsi),
              level=0)
    fpath = get_weights_fn(store_iterations, deltatau, run, graph_name, ratios[0])
    plt.figure()
    x_vals = []
    y_vals = []
    pf = open(fpath, "rb")
    for lidx, l in enumerate(pf):
        x_vals.append((float(lidx)+1)*deltatau/deltapsi*store_iterations)
        print sum([float(x) for x in l.split("\t")])
        y_vals.append(sum([float(x) for x in l.split("\t")]))
    plt.plot(x_vals, y_vals, alpha=0.5)
    #plt.plot(range(len(cas)), cas, "ro", alpha=0.5)
    pf.close()
    plt.title("Activity in Network over "+r"$\tau$"+"\n"
              +r"$\Delta\tau$"+"={}, ".format(deltatau)
              +r"$\Delta\psi$"+"={}, ".format("%.4f" % deltapsi)
              +r"$a_c=${}".format("%.2f" % a_cs[0]))
    plt.xlabel(r"$\tau$ (in Months)")
    plt.ylabel("Average Activity per Network")
    ax = plt.axes()
    ax.grid(color="gray")
    plt.savefig(storage_folder + graph_name + "_" + str(store_iterations) +
                "_iterations_over_time_{}_{}_empirical.png".format(store_iterations, str(deltatau).replace(".", "")))
    plt.close("all")



def avrg_activity_over_tau(graph_name):
    debug_msg("  >>> Drawing Average Activity over Tau", level=0)
    graph, ratios, deltatau, deltapsi, graph_name, store_iterations, cas, \
    ew, random_inits, a_cs = get_network_details(graph_name)
    storage_folder = config.plot_dir + "average_weights_over_tau/" + graph_name + "/"
    for ratio in ratios:
        x_avrg = []
        y_avrg = []
        for run in xrange(random_inits):
            debug_msg("  ++ Processing ratio = {}, deltatau = {}".format(ratio, deltatau), level=0)
            fpath = get_weights_fn(store_iterations, deltatau, run, graph_name, ratio)
            weights_df = pd.read_csv(fpath, sep="\t", header=None)
            means = weights_df.mean(axis=1)
            if len(y_avrg) == 0:
                y_avrg = means[:]
                for lidx in xrange(len(y_avrg)):
                    x_avrg.append(float(lidx)*(deltatau/deltapsi)*store_iterations)
            else:
                y_avrg += means[:]
        plt.figure()
        pf = open(fpath, "rb")
        plt.plot(x_avrg, y_avrg/random_inits, alpha=0.5)
        pf.close()
        plt.title("Activity in Network over "+r"$\tau$"+"\n"+r"$\frac{\lambda}{\mu}$"+
                  "={},".format(ratio)+r"$\Delta\tau$"+"={}, ".format(deltatau)+
                  r"$\Delta\psi$"+"={}".format(deltapsi))
        plt.xlabel(r"$\tau$")
        plt.ylabel("Average Activity per Network")
        plt.subplots_adjust(bottom=0.15)
        ax = plt.axes()
        ax.grid(color="gray")
        plt.savefig(storage_folder + graph_name + "_" + str(store_iterations) +
                    "_iterations_over_time_{}_{}.png".format(str(ratio).replace(".", ""),
                                                             str(deltatau).replace(".", "")))
        plt.close("all")


def plot_weights_over_time(graph_name):
    import subprocess
    import os

    debug_msg("  >>> Drawing Weights over Time plot", level=0)
    graph, ratios, deltatau, deltapsi, graph_name, store_iterations, cas, ew, random_inits, a_cs = get_network_details(graph_name)

    for ratio in ratios:
        y_vals = []
        x_vals = [0.0]
        out_path = os.path.abspath(config.plot_dir + "weights_over_time/" + graph_name + "/"+graph_name+"_"+
                                   str(ratio).replace(".", "")+".txt")
        out_file = open(out_path, "wb")
        for run in xrange(random_inits):
            debug_msg("  ++ Processing ratio = {}, deltatau = {}".format(ratio, deltatau), level=0)
            fpath = get_weights_fn(store_iterations, deltatau, run, graph_name, ratio)
            if run == 0:
                y_vals = pd.read_csv(fpath, sep="\t", header=None)
            else:
                y = pd.read_csv(fpath, sep="\t", header=None)
                y_vals = y_vals.add(y).fillna(method="ffill", axis=1)
        for lidx in xrange(len(y_vals)):
            x_vals.append(float(lidx)*(deltatau/deltapsi)*store_iterations)
        y_vals = y_vals.div(random_inits).fillna(method="ffill", axis=1).as_matrix()
        out_file.write(("\t").join(["%.8f" % xn for xn in x_vals])+"\n")
        for seq in y_vals:
            out_file.write(("\t").join([str(x) for x in seq])+"\n")
        out_file.close()
        debug_msg(" -- Calling weights_over_time.R", level=0)
        r_script_path = os.path.abspath(config.r_dir + 'weights_over_time.R')
        wd = r_script_path.replace("DynamicNetworks/R Scripts/weights_over_time.R", "") + \
             "DynamicNetworksResults/plots/weights_over_time/"
        subprocess.call(['/usr/bin/RScript', r_script_path, wd, out_path, str(ratio), str(deltatau), graph_name,
                         "%.2f" % ew[0]], stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
        debug_msg(" -- Done", level=0)

        # plt.figure()
        # plt.plot(x_vals, y_vals, alpha=0.5)
        # plt.title("Activity per Node over "+r"$\tau$"+"\n("+r"$\frac{\lambda}{\mu}$"+"={},".format(ratio)+r"$\Delta\tau$"+"={})".format(deltatau))
        # plt.xlabel(r"$\tau$")
        # plt.ylabel("Activityratio per Node")
        # plt.subplots_adjust(bottom=0.15)
        # ax = plt.axes()
        # ax.grid(color="gray")
        # #plt.ylim(0, 100000)
        # plt.savefig(storage_folder + graph_name + "_" + str(store_iterations) +
        #             "_iterations_over_time_{}_{}.png".format(str(ratio).replace(".", ""), str(deltatau).replace(".", "")))
        # plt.close("all")



# def create_scatterplots(graph_name, graph_source_name, max_iter_ev=100000, max_iter_hits=100000,
#                         plot_labels=False, alpha=0.75):
#     if graph_source_name == "":
#         graph_source_name = graph_name
#
#     graph, ratios, deltatau, deltastep, graph_name, store_iterations, cas, ew, random_inits = get_network_details(graph_name)
#     nw = Network(10000, False, graph_name, plot_with_labels=True)
#     nw.create_folders()
#     gml_f = config.graph_binary_dir + "GT/" + graph_source_name + "/" + graph_source_name + "_run_0.gt"
#     debug_msg("  >>> Creating scatterplots for {}".format(graph_name), level=0)
#     debug_msg("Loading {}".format(gml_f), level=1)
#     nw.load_graph(gml_f)
#
#     debug_msg("\x1b[35m>> Loading Network Metrics\x1b[00m", level=1)
#
#     debug_msg("\x1b[33m  ++ Loading Degree Property Map\x1b[00m", level=1)
#     degree = nw.graph.vertex_properties["degree"]
#     node_size = degree.a
#     node_size_str = "(size=degree)"
#
#     debug_msg("\x1b[33m  ++ Loading PageRank\x1b[00m", level=1)
#     pr = nw.graph.vertex_properties["pagerank"]
#
#     debug_msg("\x1b[33m  ++ Loading Clustering Coefficient\x1b[00m", level=1)
#     clustering = nw.graph.vertex_properties["clustercoeff"]
#
#     debug_msg("\x1b[33m  ++ Loading Eigenvector Centrality\x1b[00m", level=1)
#     ev_centrality = nw.graph.vertex_properties["evcentrality"]
#
#     debug_msg("\x1b[33m  ++ Loading HITS\x1b[00m", level=1)
#     authorities = nw.graph.vertex_properties["authorities"]
#     hubs = nw.graph.vertex_properties["hubs"]
#
#     clist = []
#     cmap = plt.get_cmap('gist_rainbow')
#
#     try:
#         for v in nw.graph.vertex_properties["colors"].a:
#             clist.append(v)
#     except:
#         norm = matplotlib.colors.Normalize(vmin=0, vmax=nw.graph.num_vertices())
#         m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
#         for x in xrange(nw.graph.num_vertices()):
#             clist.append(m.to_rgba(x))
#
#     for a in a_vals:
#         for b in b_vals:
#             fpath = get_avg_weight_fn(a, b, graph_name, num_iterations)
#             debug_msg("  >>> creating scatterplots for a={}, b={}\x1b[00m".format(a, b), level=1)
#             f = open(fpath, "rb")
#             avrg_node_weights_d = defaultdict(float)
#             num_nodes = 0
#             for nidx, line in enumerate(f):
#                 avrg_node_weights_d[nidx] = np.mean(np.array(np.fromstring(line, sep='\t')))
#                 num_nodes += 1
#             f.close()
#             storage_folder = config.plot_dir + "scatterplots/" + graph_name + "/"
#             try:
#                 corr, pval = sp.stats.pearsonr(clustering.a, avrg_node_weights_d.values())
#                 if pval < 0.01:
#                     pval_desc = "p-val < 0.01"
#                 else:
#                     pval_desc = "p-val = %.2f" % pval
#
#                 debug_msg("  ** plotting: Average Activity vs. Cluster Coeff (corr={})".format(corr), level=1)
#                 plt.figure()
#                 plt.scatter(clustering.a, avrg_node_weights_d.values(), cmap=cmap, s=node_size, c=clist, alpha=alpha)
#                 plt.xlabel("Clustering Coefficient")
#                 plt.ylabel("Average Activity")
#                 plt.grid(color="gray")
#                 plt.title("Activity vs. Clustering (corr={}; {})".format("%.2f" % corr, pval_desc) + node_size_str)
#                 if plot_labels:
#                     labels = xrange(0, num_nodes, 1)
#                     for label, x, y in zip(labels, clustering.a, avrg_node_weights_d.values()):
#                         plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points',
#                                      ha='right', va='bottom',
#                                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
#                 plt.savefig(storage_folder + graph_name + "_" + str(num_iterations) + "_iterations_" +
#                             str(a).replace(".", "") + "_" + str(b).replace(".", "") +
#                             "_clustering_coefficient_scatterplot.png")
#                 plt.close("all")
#             except Exception, e:
#                 print e
#
#             try:
#                 corr, pval = sp.stats.pearsonr(pr.a, avrg_node_weights_d.values())
#                 if pval < 0.01:
#                     pval_desc = "p-val < 0.01"
#                 else:
#                     pval_desc = "p-val = %.2f" % pval
#                 debug_msg("  ** plotting: Average Activity vs. PageRank", level=1)
#                 plt.figure()
#                 plt.scatter(pr.a, avrg_node_weights_d.values(), cmap=cmap, s=node_size, c=clist, alpha=alpha)
#                 plt.xlabel("PageRank")
#                 plt.ylabel("Average Activity")
#                 plt.grid(color="gray")
#                 plt.title("Activity vs. PageRank (corr={}; {})".format("%.2f" % corr, pval_desc) + node_size_str)
#                 if plot_labels:
#                     labels = xrange(0, num_nodes, 1)
#                     for label, x, y in zip(labels, pr.a, avrg_node_weights_d.values()):
#                         plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right',
#                                      va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
#                 plt.savefig(storage_folder + graph_name + "_" + str(num_iterations) + "_iterations_" +
#                             str(a).replace(".", "") + "_" + str(b).replace(".", "") + "_pagerank_scatterplot.png")
#                 plt.close("all")
#             except Exception, e:
#                 print e
#
#             try:
#                 corr, pval = sp.stats.pearsonr(hubs.a, avrg_node_weights_d.values())
#                 if pval < 0.01:
#                     pval_desc = "p-val < 0.01"
#                 else:
#                     pval_desc = "p-val = %.2f" % pval
#                 debug_msg("  ** plotting: Average Activity vs. Hubs", level=1)
#                 plt.figure()
#                 plt.scatter(hubs.a, avrg_node_weights_d.values(), cmap=cmap, s=node_size, c=clist, alpha=alpha)
#                 plt.xlabel("Hubs")
#                 plt.ylabel("Average Activity")
#                 plt.grid(color="gray")
#                 plt.title("Activity vs. Hubs (corr={}; {})".format("%.2f" % corr, pval_desc) + node_size_str)
#                 if plot_labels:
#                     labels = xrange(0, num_nodes, 1)
#                     for label, x, y in zip(labels, hubs.a, avrg_node_weights_d.values()):
#                         plt.annotate(label,
#                                      xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom',
#                                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
#                         )
#                 plt.savefig(storage_folder + graph_name + "_" + str(num_iterations) + "_iterations_" +
#                             str(a).replace(".", "") + "_" + str(b).replace(".", "") + "_hubs_scatterplot.png")
#                 plt.close("all")
#                 corr, pval = sp.stats.pearsonr(authorities.a, avrg_node_weights_d.values())
#                 if pval < 0.01:
#                     pval_desc = "p-val < 0.01"
#                 else:
#                     pval_desc = "p-val = %.2f" % pval
#                 debug_msg("  ** plotting: Average Activity vs. Authorities", level=1)
#                 plt.figure()
#                 plt.scatter(authorities.a, avrg_node_weights_d.values(), cmap=cmap, s=node_size, c=clist, alpha=alpha)
#                 plt.xlabel("Authorities")
#                 plt.ylabel("Average Activity")
#                 plt.grid(color="gray")
#                 plt.title("Activity vs. Authorities (corr={}; {})".format("%.2f" % corr, pval_desc) + node_size_str)
#                 if plot_labels:
#                     labels = xrange(0, num_nodes, 1)
#                     for label, x, y in zip(labels, authorities.a, avrg_node_weights_d.values()):
#                         plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right',
#                                      va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
#                 plt.savefig(storage_folder + graph_name + "_" + str(num_iterations) + "_iterations_" +
#                             str(a).replace(".", "") + "_" + str(b).replace(".", "") + "_authorities_scatterplot.png")
#                 plt.close("all")
#             except Exception, e:
#                 print e
#
#             try:
#                 corr, pval = sp.stats.pearsonr(degree.a, avrg_node_weights_d.values())
#                 if pval < 0.01:
#                     pval_desc = "p-val < 0.01"
#                 else:
#                     pval_desc = "p-val = %.2f" % pval
#                 debug_msg("  ** plotting: Average Activity vs. Degrees", level=1)
#                 plt.figure()
#                 plt.scatter(degree.a, avrg_node_weights_d.values(), cmap=cmap, s=node_size, c=clist, alpha=alpha)
#                 plt.xlabel("Degree")
#                 plt.ylabel("Average Activity")
#                 plt.grid(color="gray")
#                 plt.title("Activity vs. Degree (corr={}; {})".format("%.2f" % corr, pval_desc) + node_size_str)
#                 if plot_labels:
#                     labels = xrange(0, num_nodes, 1)
#                     for label, x, y in zip(labels, degree.a, avrg_node_weights_d.values()):
#                         plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right',
#                                      va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
#                 plt.savefig(storage_folder + graph_name + "_" + str(num_iterations) + "_iterations_" +
#                             str(a).replace(".", "") + "_" + str(b).replace(".", "") + "_degrees_scatterplot.png")
#                 plt.close("all")
#             except Exception, e:
#                 print e
#
#             try:
#                 corr, pval = sp.stats.pearsonr(ev_centrality.a, avrg_node_weights_d.values())
#                 if pval < 0.01:
#                     pval_desc = "p-val < 0.01"
#                 else:
#                     pval_desc = "p-val = %.2f" % pval
#                 debug_msg("  ** plotting: Average Activity vs. Eigenvector Centrality", level=1)
#                 plt.figure()
#                 plt.scatter(ev_centrality.a, avrg_node_weights_d.values(), cmap=cmap, s=node_size, c=clist, alpha=alpha)
#                 plt.xlabel("Eigenvector Centrality")
#                 plt.ylabel("Average Activity")
#                 plt.grid(color="gray")
#                 plt.gray()
#                 plt.title("Activity vs. Eigenvector Centrality (corr={}; {})".format("%.2f" % corr, pval_desc) + node_size_str)
#                 if plot_labels:
#                     labels = xrange(0, num_nodes, 1)
#                     for label, x, y in zip(labels, ev_centrality.a, avrg_node_weights_d.values()):
#                         plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right',
#                                      va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
#                 plt.savefig(storage_folder + graph_name + "_" + str(num_iterations) + "_iterations_" +
#                             str(a).replace(".", "") + "_" + str(b).replace(".", "") + "_eigenvector_centrality_scatterplot.png")
#                 plt.close("all")
#             except Exception, e:
#                 print e


def debug_msg(msg, level=0):
    if level <= DEBUG_LEVEL:
        print "  \x1b[33m-UTL-\x1b[00m [\x1b[36m{}\x1b[00m] \x1b[35m{}\x1b[00m".format(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), msg)


# Methods for plotting eigenvalues summary
def get_highest_evs_array(evs_per_iter):
    highest_evs = []
    for x in evs_per_iter:
        highest_evs.append(x[0])
    return highest_evs


def get_average_evs_array(evs_per_iter):
    average_evs = []
    for x in evs_per_iter:
        average_evs.append(reduce(lambda x, y: x + y, x) / float(len(x)))
    return average_evs


def get_median_evs_array(evs_per_iter):
    median_evs = []
    for x in evs_per_iter:
        if len(x) % 2 == 1:
            median = x[len(x) / 2]
        else:
            median1 = x[len(x) / 2]
            median2 = x[(len(x) / 2) - 1]
            median = (median1 + median2) / 2
        median_evs.append(median)
    return median_evs


def plot_eigenvalues_summary(highest_evs, average_evs, median_evs, edges_per_iter, percent_per_iter, graph_name, title="No Title", subtitle="No Subtitle"):
    num_iter = len(edges_per_iter)
    x = np.arange(num_iter)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.075, 0.125, 0.65, 0.75])
    ax.plot(x, highest_evs, "-g^", label="highest eigenvalue")
    ax.plot(x, average_evs, "-bs", label="average eigenvalue")
    ax.plot(x, median_evs, "-ro", label="median eigenvalue")
    ax.set_xlabel('edges (% of connectivity)')
    ax.set_ylabel('eigenvalue')
    e_p_combined = []
    for i in range(0, len(edges_per_iter)):
        e_p_combined.append(str(edges_per_iter[i]) + "\n(" + str(percent_per_iter[i]) + "%)")
    plt.xticks(np.arange(num_iter), e_p_combined, rotation=30, fontsize=8)
    #plt.title(title)
    plt.figtext(0.5, 0.95, title, fontsize=18, ha="center")
    plt.figtext(0.5, 0.9, subtitle, fontsize=10, ha="center")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.savefig("plots/eigenvalues/" + graph_name + "_summary.pdf")
    plt.close("all")


def plot_eigenvalue_comparison(eigenvalues, edges_per_iter, percent_per_iter, labels, title="No Title", subtitle="No Subtitle", xlabel="edges (% of connectivity)", ylabel="eigenvalues", filesuffix=""):
    if len(eigenvalues) > 20:
        debug_msg("It is not possible to print that many different eigenvalues. Max possible = 20!")
        return None
    if len(eigenvalues) != len(labels):
        debug_msg("The number of labels is not matching the number of different eigenvalues!")
        return None

    # Check for unique values
    unique_eigenvalues = []
    unique_labels = []
    for index, evals in enumerate(eigenvalues):
        if evals not in unique_eigenvalues:
            unique_eigenvalues.append(evals)
            unique_labels.append(labels[index])
        else:
            for i, newevals in enumerate(unique_eigenvalues):
                if evals == newevals:
                    unique_labels[i] += ", " + labels[index]

    styles = ["-r+",
              "-b.",
              "-go",
              "-c*",
              "-mp",
              "-ys",
              "-kx",
              "-rD",
              "-bh",
              "-g^",
              "-.r+",
              "-.b.",
              "-.go",
              "-.c*",
              "-.mp",
              "-.ys",
              "-.kx",
              "-.rD",
              "-.bh",
              "-.g^"]

    wrapped_labels = []
    for l in unique_labels:
        wrapped_labels.append("\n".join(wrap(l, 20)))

    num_iter = len(edges_per_iter)
    x = np.arange(num_iter)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.075, 0.125, 0.65, 0.75])
    for index, values in enumerate(unique_eigenvalues):
        ax.plot(x, values, styles[index], label=wrapped_labels[index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    e_p_combined = []
    for i in range(0, len(edges_per_iter)):
        e_p_combined.append(str(edges_per_iter[i]) + "\n(" + str(percent_per_iter[i]) + "%)")
    plt.xticks(np.arange(num_iter), e_p_combined, rotation=30, fontsize=8)
    #plt.title(title)
    plt.figtext(0.5, 0.95, title, fontsize=16, ha="center")
    plt.figtext(0.5, 0.9, subtitle, fontsize=10, ha="center")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.savefig("plots/eigenvalues/" + title + filesuffix + ".pdf")
    plt.close("all")


def store_eigenvalue_summary(evs_per_iter, edges_per_iter, percent_per_iter, graph_name):
    highest_evs = get_highest_evs_array(evs_per_iter)
    average_evs = get_average_evs_array(evs_per_iter)
    median_evs = get_median_evs_array(evs_per_iter)
    # the following directory creation should be moved to the setup file!
    directory = "graph_sources/eigenvalues_summary/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    summary_file = open(directory + graph_name + ".txt", "w")
    for i in highest_evs:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in average_evs:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in median_evs:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in edges_per_iter:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in percent_per_iter:
        summary_file.write(str(i) + " ")
    summary_file.close()


def load_and_prepare_eigenvalue_summary(graph_name):
    directory = "graph_sources/eigenvalues_summary/"
    if not os.path.exists(directory):
        debug_msg("Summary files could not be found! Aborting...")
        return None
    summary_file = open(directory + graph_name + ".txt", "r")
    highest_evs = summary_file.readline().rstrip(" \n").split(" ")
    average_evs = summary_file.readline().rstrip(" \n").split(" ")
    median_evs = summary_file.readline().rstrip(" \n").split(" ")
    edges_per_iter = summary_file.readline().rstrip(" \n").split(" ")
    percent_per_iter = summary_file.readline().rstrip(" \n").split(" ")
    summary_file.close()
    return highest_evs, average_evs, median_evs, edges_per_iter, percent_per_iter


def store_eigenvalue_summary_average(graph_name, num_iter, delFiles = False):
    directory = "graph_sources/eigenvalues_summary/"
    if not os.path.exists(directory):
        debug_msg("Summary files could not be found! Aborting...")
        return None
    highest_evs_over_iter = []
    average_evs_over_iter = []
    median_evs_over_iter = []
    edges_per_iter = None
    percent_per_iter = None
    for i in range(0, num_iter):
        try:
            summary_file = open(directory + graph_name + "_" + str(i) + ".txt", "r")
        except Exception:
            debug_msg("Files with the given name could not be found! Aborting...")
            return None
        highest_evs = summary_file.readline().rstrip(" \n").split(" ")
        average_evs = summary_file.readline().rstrip(" \n").split(" ")
        median_evs = summary_file.readline().rstrip(" \n").split(" ")
        if edges_per_iter is None and percent_per_iter is None:
            edges_per_iter = summary_file.readline().rstrip(" \n").split(" ")
            percent_per_iter = summary_file.readline().rstrip(" \n").split(" ")
        for j in range(0, len(highest_evs)):
            if len(highest_evs_over_iter) < len(highest_evs):
                highest_evs_over_iter.append(0)
                average_evs_over_iter.append(0)
                median_evs_over_iter.append(0)
            highest_evs_over_iter[j] += float(highest_evs[j])
            average_evs_over_iter[j] += float(average_evs[j])
            median_evs_over_iter[j] += float(median_evs[j])
    highest_evs_over_iter = [x/num_iter for x in highest_evs_over_iter]
    average_evs_over_iter = [x/num_iter for x in average_evs_over_iter]
    median_evs_over_iter = [x/num_iter for x in median_evs_over_iter]
#     print highest_evs_over_iter
#     print average_evs_over_iter
#     print median_evs_over_iter
#     print edges_per_iter
#     print percent_per_iter
    if not os.path.exists(directory):
        os.makedirs(directory)
    summary_file = open(directory + graph_name + ".txt", "w")
    for i in highest_evs_over_iter:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in average_evs_over_iter:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in median_evs_over_iter:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in edges_per_iter:
        summary_file.write(str(i) + " ")
    summary_file.write("\n")
    for i in percent_per_iter:
        summary_file.write(str(i) + " ")
    summary_file.close()
    if delFiles is True:
        for i in range(num_iter):
            os.remove(directory + graph_name + "_" + str(i) + ".txt")


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
