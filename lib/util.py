__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
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

# set level of debug output
DEBUG_LEVEL = 0


def get_network_details_for_epochs(graph_name):
    graph = load_graph(config.graph_binary_dir + "GT/" + graph_name + "/" + graph_name + "_run_" + str(0) + ".gt")
    dtau = graph.graph_properties["deltatau"]
    store_itas = graph.graph_properties["store_iterations"]
    a_cs = graph.graph_properties["a_cs"]
    ratios = graph.graph_properties["ratios"]
    ratios = [np.nan] + ratios
    k1s = graph.graph_properties["k1_over_epochs"]
    gs = graph.graph_properties["g_over_epochs"]
    max_qs = graph.graph_properties["max_q_over_epochs"]
    mus = graph.graph_properties["mu_over_epochs"]
    num_epochs = len(a_cs)
    apm = graph.graph_properties["activity_per_month"]
    sapm = graph.graph_properties["simulated_activity_per_month"]
    fapm = graph.graph_properties["filtered_activity"]
    return dtau, store_itas, num_epochs, a_cs, ratios, k1s, gs, max_qs, mus, apm, sapm, fapm


def get_network_details(graph_name):
    graph = load_graph(config.graph_binary_dir + "GT/" + graph_name + "/" + graph_name + "_run_" + str(0) + ".gt")
    dtau = graph.graph_properties["deltatau"]
    store_itas = graph.graph_properties["store_iterations"]
    mu = round(graph.graph_properties["mu"], 2)
    ac = round(graph.graph_properties["a_c"], 2)
    ratios = graph.graph_properties["ratios"]
    ratios = ratios + [np.nan]
    k1 = round(graph.graph_properties["top_eigenvalues"][0], 2)
    apm = graph.graph_properties["activity_per_month"]
    sapm = graph.graph_properties["simulated_activity_per_month"]
    csapm = graph.graph_properties["corrected_simulated_activity_per_month"]
    k1s = graph.graph_properties["k1s"]
    aanu = graph.graph_properties["agg_act_new_users"]
    nvoi = graph.graph_properties["num_vertices_over_iter"]
    neoi = graph.graph_properties["num_edges_over_iter"]
    acoi = graph.graph_properties["a_c_over_iter"]
    mppdoi = graph.graph_properties["mppd_over_iter"]
    gpmoi = graph.graph_properties["gpm_over_iter"]
    mqoi = graph.graph_properties["mq_over_iter"]
    return dtau, store_itas, mu, ac, ratios, k1, apm, sapm, csapm, k1s, aanu, nvoi, neoi, acoi, mppdoi, gpmoi, mqoi


# retrieve parameters stored in binary graph file
# def get_network_details(graph_name, run=0, path=config.graph_binary_dir):
#     graph = load_graph(path + "GT/" + graph_name + "/" + graph_name + "_run_" + str(run) + ".gt")
#     ratios = graph.graph_properties["ratios"]
#     deltataus = graph.graph_properties["deltatau"]
#     deltapsi = graph.graph_properties["deltapsi"]
#     ew = graph.graph_properties["top_eigenvalues"]
#     random_inits = graph.graph_properties["runs"]
#     store_iterations = graph.graph_properties["store_iterations"]
#     try:
#         a_cs = graph.graph_properties["a_cs"]
#         cas = graph.graph_properties["activity_per_month"]
#     except:
#         debug_msg(" --> Could not load empirical data!")
#         cas = []
#         a_cs = []
#     return graph, ratios, deltataus, deltapsi, graph_name, store_iterations, cas, ew, random_inits, a_cs


def prepare_folders():
    flist = [config.graph_binary_dir,
             config.graph_binary_dir + "GT/",
             config.graph_binary_dir + "ratios/",
             config.graph_binary_dir + "empirical_data/",
             config.graph_source_dir + "empirical_results/",
             config.graph_source_dir,
             config.graph_source_dir + "weights",
             config.graph_dir,
             config.ds_source_dir,
             config.plot_dir,
             config.plot_dir + "eigenvalues",
             config.plot_dir + "functions",
             config.plot_dir + "weights_over_time",
             config.plot_dir + "average_weights_over_tau",
             config.plot_dir + "empirical_results"]

    for fname in flist:
        create_folder(fname)


def create_folder(folder):
    if not os.path.exists(folder):
        debug_msg("Created folder: {}".format(folder))
        os.makedirs(folder)


# helper functions to get filename
def get_weights_fn(store_iterations, deltatau, run, graph_name, ratio):
    return config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + \
           "_" + str(store_iterations).replace(".", "") + "_" + \
           str(deltatau).replace(".", "") + "_" + str(ratio).replace(".", "") + "_run_" + str(run) + "_weights.txt"


def get_abs_path(graph_name, suffix, store_iterations, ratio, deltatau=0.001, run=0):
    return os.path.abspath(config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + \
           "_" + str(store_iterations).replace(".", "") + "_" + \
           str(deltatau).replace(".", "") + "_run_" + str(run) + suffix + ".txt")


# helper function to get filename for intrinsic activity
def get_intrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratio):
    return config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + \
           "_" + str(store_iterations).replace(".", "") + "_" + \
           str(deltatau).replace(".", "") + "_" + str(ratio).replace(".", "") + "_run_" + str(run) + "_intrinsic.txt"


# helper function to get filename for extrinsic activity
def get_extrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratio):
    return config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + \
           "_" + str(store_iterations).replace(".", "") + "_" + \
           str(deltatau).replace(".", "") + "_" + str(ratio).replace(".", "") + "_run_" + str(run) + "_extrinsic.txt"


def empirical_result_plot_for_epochs(graph_name, mode, plot_fmt):
    debug_msg("*** Start plotting of empirical results ***")
    import subprocess
    import os
    output_path = os.path.abspath(config.graph_source_dir + "empirical_results/" + graph_name + "_" + mode +
                                  "_epochs.txt")
    debug_msg("--> Start collecting data of: " + graph_name)
    dtau, store_itas, num_epochs, a_cs, ratios, k1s, gs, max_qs, mus, apm, sapm, fapm = get_network_details_for_epochs(graph_name)
    debug_msg("--> Done with collecting data")
    real_act_y = apm
    real_act_x = range(len(apm))
    sim_act_y = sapm
    sim_act_x = range(len(apm))
    fil_act_y = fapm
    fil_act_x = range(len(fapm))
    debug_msg("--> Real activity data included")
    debug_msg("--> Starting combination of data")
    combined_data = [a_cs, ratios, k1s, gs, max_qs, mus, real_act_x, real_act_y, sim_act_x, sim_act_y, fil_act_x, fil_act_y]
    header = "a_cs\tratios\tk1s\tgs\tmax_qs\tmus\treal_act_x\treal_act_y\tsim_act_x\tsim_act_y\tfil_act_x\tfil_act_y"
    np.savetxt(output_path, np.array(combined_data).T, delimiter="\t", header=header, comments="")
    debug_msg("--> Data successfully combined")
    debug_msg("--> Getting weight file and tau file path")
    weights_path = get_abs_path(graph_name, "_weights", store_itas, ratios[1], deltatau=dtau)
    debug_msg("----> " + weights_path)
    taus_path = get_abs_path(graph_name, "_taus", store_itas, ratios[1], deltatau=dtau)
    debug_msg("----> " + taus_path)
    debug_msg("--> Calling empirical_plots_epochs.R")
    r_script_path = os.path.abspath(config.r_dir + 'empirical_plots_epochs.R')
    wd = r_script_path.replace("R Scripts/empirical_plots_epochs.R", "") + config.plot_dir + "empirical_results/"
    subprocess.call([config.r_binary_path, r_script_path, wd, output_path, weights_path, taus_path, graph_name, mode,
                     plot_fmt])
                    #stdout=open(os.devnull, 'wb'))#, stderr=open(os.devnull, 'wb'))
    debug_msg("*** Successfully plotted empirical results ***")


def empirical_result_plot(graph_name, mode, plot_fmt):
    debug_msg("*** Start plotting of empirical results ***")
    import subprocess
    import os
    output_path = os.path.abspath(config.graph_source_dir + "empirical_results/" + graph_name + "_" + mode + ".txt")
    output_table = os.path.abspath(config.graph_source_dir + "empirical_results/" + graph_name + "_table_" + mode + ".txt")
    debug_msg("--> Start collecting data of: " + graph_name)
    dtau, store_itas, mu, ac, ratios, k1, apm, sapm, csapm, k1s, aanu, nvoi, neoi, acoi, mppdoi, gpmoi, mqoi = get_network_details(graph_name)
    debug_msg("--> Done with collecting data")
    real_act_y = apm
    real_act_x = range(len(apm))
    debug_msg("--> Real activity data included")
    debug_msg("--> Starting combination of data")
    combined_data = [ratios, k1s, real_act_x, real_act_y, sapm, csapm]
    header = "ratios\tk1s\treal_act_x\treal_act_y\tsim_act_y\tcor_sim_act_y"
    np.savetxt(output_path, np.array(combined_data).T, delimiter="\t", header=header, comments="")
    debug_msg("--> Data successfully combined")
    debug_msg("--> Save Latex table")
    tabstring_nvoi = ""
    tabstring_neoi = ""
    tabstring_k1oi = ""
    tabstring_roi = ""
    tabstring_acoi = ""
    tabstring_mppdoi = ""
    tabstring_gpmoi = ""
    tabstring_mqoi = ""
    tabstring_aanu = ""
    tabstring_sad = ""
    sad_list = []
    for i in range(0, len(apm)):
        sad_list.append(round(csapm[i] - apm[i], 2))
    for i in range(0, len(nvoi)):
        tabstring_nvoi += "& $" + str(nvoi[i]) + "$ "
        tabstring_neoi += "& $" + str(neoi[i]) + "$ "
        tabstring_k1oi += "& $" + str(round(k1s[i], 2)) + "$ "
        tabstring_roi += "& $" + str(round(ratios[i], 2)) + "$ "
        tabstring_acoi += "& $" + str(round(acoi[i], 2)) + "$ "
        tabstring_mppdoi += "& $" + str(round(mppdoi[i], 2)) + "$ "
        tabstring_gpmoi += "& $" + str(round(gpmoi[i], 2)) + "$ "
        tabstring_mqoi += "& $" + str(round(mqoi[i], 2)) + "$ "
        tabstring_aanu += "& $" + str(aanu[i]) + "$ "
        tabstring_sad += "& $" + str(sad_list[i]) + "$ "
    tabfile = open(output_table, "w")
    tabfile.write(tabstring_nvoi + "\n")
    tabfile.write(tabstring_neoi + "\n")
    tabfile.write(tabstring_k1oi + "\n")
    tabfile.write(tabstring_roi + "\n")
    tabfile.write(tabstring_acoi + "\n")
    tabfile.write(tabstring_mppdoi + "\n")
    tabfile.write(tabstring_gpmoi + "\n")
    tabfile.write(tabstring_mqoi + "\n")
    tabfile.write(tabstring_aanu + "\n")
    tabfile.write(tabstring_sad + "\n")
    tabfile.close()
    debug_msg("--> Done.")
    #debug_msg("--> Getting weight file and tau file path")
    #weights_path = get_abs_path(graph_name, "_weights", store_itas, ratios[1], deltatau=dtau)
    #debug_msg("----> " + weights_path)
    #taus_path = get_abs_path(graph_name, "_taus", store_itas, ratios[1], deltatau=dtau)
    #debug_msg("----> " + taus_path)
    debug_msg("--> Calling empirical_plots.R")
    r_script_path = os.path.abspath(config.r_dir + 'empirical_plots.R')
    wd = r_script_path.replace("R Scripts/empirical_plots.R", "") + config.plot_dir + "empirical_results/"
    subprocess.call([config.r_binary_path, r_script_path, wd, output_path, graph_name, mode,
                     plot_fmt, str(dtau), str(mu), str(ac), str(k1)])
                    #stdout=open(os.devnull, 'wb'))#, stderr=open(os.devnull, 'wb'))
    debug_msg("*** Successfully plotted empirical results ***")


# # plot empirical activity (left y-axis) vs. observed activity (right y-axis) for empirical datasets
# def empirical_result_plot(graph_name, run=0):
#     import subprocess
#     import os
#     source_path = os.path.abspath(config.graph_source_dir + "empirical_results/" + graph_name + ".txt")
#     out_file = open(source_path, "wb")
#     debug_msg("  >>> Creating empirical result plot", level=0)
#     graph, ratios, deltatau, deltapsi, graph_name, store_iterations, \
#     cas, ew, random_inits, a_cs = get_network_details(graph_name, run)
#     pipe_vals = [str(deltatau), "%.4f" % deltapsi, "%.2f" % a_cs[0], "%.2f" % ew[0]]
#     debug_msg("   ** Preparing Ratios", level=0)
#     x_ratios = [(float(x)+1) for x in range(len(ratios))]
#     y_ratios = ratios
#     out_file.write(("\t").join(["%.8f" % x for x in x_ratios])+"\n")
#     out_file.write(("\t").join(["%.8f" % x for x in y_ratios])+"\n")
#     debug_msg("   ** Preparing Average Activities", level=0)
#     fpath = get_weights_fn(store_iterations, deltatau, run, graph_name, ratios[0])
#     epath = os.path.abspath(get_extrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratios[0]))
#     ipath = os.path.abspath(get_intrinsic_weights_fn(store_iterations, deltatau, run, graph_name, ratios[0]))
#     x_activity = []
#     y_activity = []
#     pf = open(fpath, "rb")
#     for lidx, l in enumerate(pf):
#         x_activity.append((float(lidx)+1)*deltatau/deltapsi*store_iterations)
#         y_activity.append(sum([float(x) for x in l.split("\t")]))
#     pf.close()
#     out_file.write(("\t").join(["%.8f" % x for x in x_activity])+"\n")
#     out_file.write(("\t").join(["%.8f" % x for x in y_activity])+"\n")
#
#     debug_msg("   ** Preparing Real Activities", level=0)
#     y_real_act = cas
#     x_real_act = range(len(cas))
#     out_file.write(("\t").join(["%.8f" % x for x in x_real_act])+"\n")
#     out_file.write(("\t").join(["%.8f" % x for x in y_real_act])+"\n")
#     out_file.close()
#     debug_msg("   ** Calling empirical_plots.R", level=0)
#     r_script_path = os.path.abspath(config.r_dir + 'empirical_plots.R')
#     wd = r_script_path.replace("R Scripts/empirical_plots.R", "") + config.plot_dir + "empirical_results/"
#     subprocess.call([config.r_binary_path, r_script_path, wd, source_path, pipe_vals[0], pipe_vals[1], pipe_vals[2],
#                      pipe_vals[3], graph_name, ipath, epath])#, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
#     debug_msg("   ** Done", level=0)


# plot simulated ratios and kappa 1 (for empirical datasets)
def plot_empirical_ratios(graph_name, run=0):
    debug_msg("  >>> Drawing Ratios over Tau", level=0)
    graph, ratios, deltatau, deltapsi, graph_name, store_iterations, cas, ew, random_inits, a_cs = get_network_details(graph_name, run)
    storage_folder = config.plot_dir + "ratios_over_time/" + graph_name + "/"
    debug_msg("    ++ Processing graph = {}, deltatau = {}, deltapsi = {}".format(graph_name, deltatau, deltapsi),
              level=0)
    x_vals = [(float(x)+1) for x in range(len(ratios))]
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


# plot average activity over tau (for empirical networks)
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


# plot average activity over tau (for synthetical datasets)
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


# function to call R Script for plotting weights over time
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
        debug_msg("  ** Calling weights_over_time.R", level=0)
        r_script_path = os.path.abspath(config.r_dir + 'weights_over_time.R')
        wd = r_script_path.replace("R Scripts/weights_over_time.R", "") + \
             "results/plots/weights_over_time/"
        subprocess.call([config.r_binary_path, r_script_path, wd, out_path, str(ratio), str(deltatau), graph_name,
                         "%.2f" % ew[0]], stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
        debug_msg("  ** Done", level=0)


def calc_random_itas_average(graph_name, scenario, step_value, rand_itas, store_itas, ratio, dtau, delFiles=False):
    debug_msg("*** Starting combination of random iterations ***")
    output_path = os.path.abspath(config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + "_" +
                                  str(store_itas) + "_" + str(float(dtau)).replace(".", "") + "_" +
                                  str(ratio).replace(".", "") + "_run_average_" + scenario + "_" + str(step_value) +
                                  ".txt")
    average = []
    for i in range(0, rand_itas):
        weight_path = get_abs_path(graph_name, "_Random_" + scenario + "_" + str(step_value) + "_weights", store_itas,
                                   ratio, deltatau=dtau, run=i)
        average.append(np.loadtxt(weight_path))
        if delFiles:
            os.remove(weight_path)
    np.savetxt(output_path, np.mean(average, axis=0))


def plot_scenario_results(graph_name, scenario, step_values, plot_fmt, rand_itas, legend_suffix):
    debug_msg("*** Starting plotting of scenario results ***")
    import subprocess
    import os
    output_path_data = os.path.abspath(config.graph_source_dir + "empirical_results/" + graph_name + "_scenarios.txt")
    output_path_weights = os.path.abspath(config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + "_" +
                                          scenario + "_combined.txt")
    output_path_results = os.path.abspath(config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + "_" +
                                          scenario + "_results.txt")
    debug_msg("--> Collection graph data...")
    dtau, dpsi, store_itas, mu, ac, ratios, k1, apm = get_network_details(graph_name)
    debug_msg("--> Done with collecting graph data.")
    real_act_y = apm
    real_act_x = range(len(apm))
    debug_msg("--> Real activity data included")
    debug_msg("--> Combining graph data...")
    combined_data = [ratios, real_act_x, real_act_y]
    header = "ratios\treal_act_x\treal_act_y"
    np.savetxt(output_path_data, np.array(combined_data).T, delimiter="\t", header=header, comments="")
    debug_msg("--> Graph data successfully combined")
    debug_msg("--> Combining weights and tau data...")
    combined_data = []
    weights_path = get_abs_path(graph_name, "_weights", store_itas, ratios[1], deltatau=dtau)
    debug_msg("----> " + weights_path)
    combined_data.append(np.loadtxt(weights_path))
    # taus_path = get_abs_path(graph_name, "_taus", store_itas, ratios[1], deltatau=dtau)
    # debug_msg("----> " + taus_path)
    # combined_data.append(np.loadtxt(taus_path))
    header = "sim_act"#\ttaus"
    len_tau = len(combined_data[0])
    for step_value in step_values:
        weights_path = get_abs_path(graph_name, "_" + scenario + "_" + str(step_value),
                                    store_itas, ratios[1], deltatau=dtau, run="average")
        debug_msg("----> " + weights_path)
        temp = np.loadtxt(weights_path)
        len_temp = len_tau - len(temp)
        temp_array = np.empty(len_temp)
        temp_array.fill(np.nan)
        temp = list(temp_array) + list(temp)
        for i, el in enumerate(temp):
            if el < 0:
                temp[i] = float(0)
        combined_data.append(temp)
        header += "\t" + scenario + "_" + str(step_value) + "_Random"
    for step_value in step_values:
        weights_path = get_abs_path(graph_name, "_Informed_" + scenario + "_" + str(step_value) + "_weights",
                                    store_itas, ratios[1], deltatau=dtau)
        debug_msg("----> " + weights_path)
        temp = np.loadtxt(weights_path)
        len_temp = len_tau - len(temp)
        temp_array = np.empty(len_temp)
        temp_array.fill(np.nan)
        temp = list(temp_array) + list(temp)
        for i, el in enumerate(temp):
            if el < 0:
                temp[i] = float(0)
        combined_data.append(temp)
        header += "\t" + scenario + "_" + str(step_value) + "_Informed"
    np.savetxt(output_path_weights, np.array(combined_data).T, delimiter="\t", header=header,
               comments="")
    temp_results = [[0 for x in range(0, len(combined_data))] for x in range(0, len(combined_data))]
    for i in range(0, len(combined_data)):
        for j in range(0, len(combined_data)):
            temp_results[i][j] = round(combined_data[i][-1] - combined_data[j][-1])
    np.savetxt(output_path_results, np.array(temp_results).T, delimiter="\t", header=header, comments="", fmt="%i")
    result_file = open(output_path_results, "a")
    result_file.write("Sim to obs: %i" % round(combined_data[0][-1] - real_act_y[-1]))
    result_file.close()
    debug_msg("--> Weights and tau data successfully combined")
    debug_msg("--> Preparing legend values...")
    legend_values = ""
    for step_value in step_values:
        if step_value is 1:
            used_legend_suffix = legend_suffix[:-1]
        else:
            used_legend_suffix = legend_suffix
        legend_values += str(step_value) + " " + used_legend_suffix + " (Random)" + ", "
    for step_value in step_values:
        if step_value is 1:
            used_legend_suffix = legend_suffix[:-1]
        else:
            used_legend_suffix = legend_suffix
        legend_values += str(step_value) + " " + used_legend_suffix + " (Informed)" + ", "
    legend_values = legend_values[:-2]
    pch_skip = ((dpsi / dtau) / store_itas)
    debug_msg("--> Done: " + legend_values)
    debug_msg("--> Calling empirical_scenarios_plots.R")
    r_script_path = os.path.abspath(config.r_dir + 'empirical_scenarios_plots.R')
    wd = r_script_path.replace("R Scripts/empirical_scenarios_plots.R", "") + config.plot_dir + "empirical_results/"
    subprocess.call([config.r_binary_path, r_script_path, wd, output_path_data, output_path_weights, graph_name,
                     scenario, plot_fmt, str(rand_itas), legend_values, str(pch_skip)])
    debug_msg("*** Successfully plotted scenario results ***")


# debug output
def debug_msg(msg, level=0):
    if level <= DEBUG_LEVEL:
        print "  \x1b[33m-UTL-\x1b[00m [\x1b[36m{}\x1b[00m] \x1b[35m{}\x1b[00m".format(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), msg)

