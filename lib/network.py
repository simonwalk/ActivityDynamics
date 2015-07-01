from __future__ import division
from time import sleep

__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

from lib.util import *
from config import config

import math
import random
import numpy as np
import pandas as pd
from numpy.lib.type_check import real, imag
import datetime

from graph_tool.all import *


import matplotlib.pyplot as plt
from random import shuffle, sample
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import sys


class Network:
    # Create Network Object with default values
    # @params:
    #   directed            =   Is graph directed?
    #   graph_name          =   Name of graph, mainly used for storing plots and figures
    #   run                 =   Number of random initializations for activity weights
    #   percentage          =   Percentage of nodes to randomly select for increasing/decreasing ratio
    #   converge_at         =   Minimum difference that has to be reached between two iterations to declare convergence
    #   ratios              =   The ratio for the activity dynamics.
    #   deltatau            =   The time that each iteration "represents".
    #   debug_level         =   The level of debug messages to be displayed.
    #   store_iterations    =   The interval for storing iterations (1 = all, 2 = every other, etc.)
    def __init__(self, directed, graph_name, run=1, converge_at=1e-16, deltatau=0.01, runs = 1,
                 deltapsi=0.0001, debug_level=1, store_iterations=1, ratios=[], ratio_index = 0, tau_in_days=30,
                 num_nodes=None):
        # default variables
        self.name_to_id = {}
        self.graph = Graph(directed=directed)
        # variables used to store and load files
        self.graph_name = graph_name
        self.run = run
        self.runs = runs
        self.num_nodes = num_nodes
        # variables used for activity dynamics modeling process
        self.cur_iteration = 0
        self.ratio_ones = [1.0] * self.graph.num_vertices()
        self.deltatau = deltatau
        self.deltapsi = deltapsi
        self.tau_in_days = tau_in_days
        self.converge_at = float(converge_at)
        self.store_iterations = store_iterations
        self.ratio = None
        self.tau_iter = 0
        self.sapm = []
        # variables used to specifically increase the ratio for certain nodes
        self.random_nodes = []
        # variable needed for adding and removing edges from graph
        self.edge_list = None
        # variable storing the eigenvalues for the network
        self.top_eigenvalues = None
        self.debug_level = debug_level
        # empirical network variables
        self.ratio_index = ratio_index
        self.ratios = ratios
        self.minimized_error_ratios = []
        # synthetic network helper variables
        self.converged = False
        self.diverged = False
        # user analysis variables
        self.agg_user_activity = None
        self.agg_emp_user_activity = None

    def reduce_collaboration_edges(self, k):
        self.debug_msg("Reducing to k = " + str(k) + " collaboration edges...", level=1)
        bool_map = self.graph.new_edge_property("bool")
        for e in self.graph.edges():
            if self.graph.edge_properties["collCount"][e] >= k:
                bool_map[e] = 1
            else:
                bool_map[e] = 0
        self.graph.set_edge_filter(bool_map)
        self.num_edges = self.graph.num_edges()
        self.debug_msg("Reduced network to " + str(self.graph.num_vertices()) + " vertices with " +
                       str(self.graph.num_edges()) + " collaboration edges.", level=1)

    def get_empirical_activity_per_user(self):
        if self.agg_emp_user_activity is None:
            self.agg_emp_user_activity = self.graph.new_vertex_property("double")
            self.graph.vertex_properties["agg_emp_user_activity"] = self.agg_emp_user_activity
        source_path = config.graph_binary_dir + "empirical_data/" + self.graph_name + "/"
        apm_per_user = pd.read_pickle(source_path + "user_df_apm.ser")
        apm_per_user = apm_per_user.sum(axis=0)
        for i in apm_per_user.index:
            v = find_vertex(self.graph, self.graph.vertex_properties["nodeID"], i)
            self.graph.vertex_properties["agg_emp_user_activity"][v[0]] = apm_per_user[i]

    def aggregate_user_activity(self):
        if self.agg_user_activity is None:
            self.agg_user_activity = self.graph.new_vertex_property("double")#[0 for x in range(0, self.num_vertices)]
            self.graph.vertex_properties["agg_user_activity"] = self.agg_user_activity
        self.graph.vertex_properties["agg_user_activity"].a += (self.graph.vertex_properties["activity"].a * self.a_c * self.graph.num_vertices())

    def get_ratio_colors(self, threshold=0):
        self.graph.vertex_properties["act_diff_cols"] = self.graph.new_vertex_property("vector<double>")
        for v in self.graph.vertices():
            if threshold is not 0:
                plus_minus = (self.agg_emp_user_activity[v] / 100) * threshold
            else:
                plus_minus = 0
            if self.agg_user_activity[v] < self.agg_emp_user_activity[v] - plus_minus:
                self.graph.vertex_properties["act_diff_cols"][v] = [1, 0, 0, 1]
            elif self.agg_user_activity[v] > self.agg_emp_user_activity[v] + plus_minus:
                self.graph.vertex_properties["act_diff_cols"][v] = [0, 1, 0, 1]
            else:
                self.graph.vertex_properties["act_diff_cols"][v] = [0, 0, 1, 1]
        #for v in self.graph.vertices():
        #    print self.graph.vertex_properties["act_diff_cols"][v]

    def plot_act_diff_graph(self, threshold=10, output_size=4000):
        out_path = config.graph_dir + self.graph_name + "_act_diff.png"
        v_size_prop_map = self.graph.vertex_properties["evcentrality"]
        val = math.sqrt(self.graph.num_vertices()) / self.graph.num_vertices() * (output_size / 4)
        mi = val
        ma = val * 2
        graph_draw(self.graph, vertex_fill_color=self.graph.vertex_properties["act_diff_cols"],
                   vertex_size=(prop_to_size(v_size_prop_map, mi=mi, ma=ma)), output=out_path,
                   output_size=(output_size, output_size))

    def calc_acs(self):
        self.a_cs = [(np.mean(self.replies) + np.mean(self.posts)) / self.num_vertices] * (len(self.replies))
        self.set_ac(0)

    def set_ac(self, index):
        self.a_c = self.a_cs[index]

    #def calc_ac(self, start_tau=0, end_tau=None, min_ac=40):
    #    replies = self.replies[start_tau:end_tau]
    #    posts = self.posts[start_tau:end_tau]
    #    return max((np.mean(replies) + np.mean(posts)) / self.num_vertices, min_ac)

    def calc_max_posts_per_day(self, start_tau=0, end_tau=None):
        return max(self.posts_per_user_per_day[start_tau:end_tau])

    def calc_g_per_month(self):
        return self.max_posts_per_day / (math.sqrt(self.a_c ** 2 + self.max_posts_per_day ** 2))

    def calc_max_q(self):
        return (self.max_posts_per_day * self.tau_in_days * self.num_vertices) / (2 * self.num_edges * self.g_per_month)

    def get_empirical_input(self, path, start_tau=0, end_tau=None, ac_per_taus=None):
        self.dx = []
        self.apm = []
        self.posts = []
        self.replies = []
        self.num_users = []
        self.init_users = []
        self.posts_per_user_per_day = []
        self.a_cs = []
        f = open(path, "rb")
        for ldx, line in enumerate(f):
            if ldx < 1:
                continue
            el = line.strip().split("\t")
            try:
              self.dx.append(float(el[1]))
            except:
                self.debug_msg(" !! Reached end of file = No dx !! ")
            self.apm.append(float(el[2]))
            self.posts.append(float(el[3]))
            self.replies.append(float(el[4]))
            try:
                self.init_users.append(el[6].split(","))
            except:
                self.init_users.append(["dummy"])
            num_users = float(el[5]) + 1
            self.num_users.append(num_users)
            self.posts_per_user_per_day.append(float(el[3])/num_users/self.tau_in_days)
        f.close()
        self.calc_acs()
        self.max_posts_per_day = self.calc_max_posts_per_day(start_tau, end_tau)
        self.g_per_month = self.calc_g_per_month()
        self.max_q = self.calc_max_q()
        self.mu = self.max_q / self.a_c
        self.deltapsi = self.mu
        self.debug_msg("max_q: {}".format(round(self.max_q, 3)), level=1)
        self.debug_msg("deltapsi: {}".format(round(self.deltapsi, 3)), level=1)
        self.debug_msg("max_posts_per_day: {}".format(round(self.max_posts_per_day, 3)), level=1)
        self.debug_msg("a_c: {}".format(round(self.a_c, 3)), level=1)
        self.debug_msg("kappa_1: {}".format(round(self.k1, 3)), level=1)

    # Creating all necessary folders for storing results, plots and figures
    def create_folders(self):
        folders = [config.graph_source_dir+"weights/"+self.graph_name+"/",
                   config.plot_dir + "weights_over_time/" + self.graph_name + "/",
                   config.plot_dir + "average_weights_over_tau/" + self.graph_name + "/",
                   config.plot_dir + "ratios_over_time/" + self.graph_name + "/"]
        try:
            for folder in folders:
                if not os.path.exists(folder):
                    self.debug_msg("Creating folder: {}".format(folder))
                    os.makedirs(folder)
        except Exception, e:
            self.debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))


    def get_binary_filename(self, source_name, bin_type="GT", run=0):
        if bin_type == "GT":
            return config.graph_binary_dir+"GT/"+source_name+"/"+source_name+"_run_"+str(run)+".gt"
        elif bin_type == "GML":
            return config.graph_binary_dir+"GML/"+source_name+"/"+source_name+"_run_"+str(run)+".gml"


    # Methods to manage the files to store the weights over all iterations
    def open_weights_files(self):
        folder = config.graph_source_dir + "weights/" + self.graph_name + "/"
        wname = self.graph_name + "_" + str(self.store_iterations) +"_"+\
                str(float(self.deltatau)).replace(".", "") + "_" + str(self.ratio).replace(".", "") + "_run_" + \
                str(self.run) + "_weights.txt"
        self.weights_file_path = folder+wname
        self.weights_file = open(self.weights_file_path, "wb")

    def open_taus_files(self):
        folder = config.graph_source_dir + "weights/" + self.graph_name + "/"
        wname = self.graph_name + "_" + str(self.store_iterations) +"_"+\
                str(float(self.deltatau)).replace(".", "") + "_" + str(self.ratio).replace(".", "") + "_run_" + \
                str(self.run) + "_taus.txt"
        self.taus_file_path = folder+wname
        self.taus_file = open(self.taus_file_path, "wb")

    def write_weights_to_file(self):
        self.weights_file.write(("\t").join(["%.8f" % float(x) for x in self.get_node_weights("activity")]) + "\n")

    def write_summed_weights_to_file(self):
        self.weights_file.write(str(sum(self.get_node_weights("activity")) * self.a_c * self.graph.num_vertices()) + "\n")

    def write_initial_tau_to_file(self):
        self.taus_file.write(str(float(0)) + "\n")

    def close_weights_files(self):
        self.weights_file.close()

    def close_taus_files(self):
        self.taus_file.close()

    def reduce_to_largest_component(self):
        fl = label_largest_component(self.graph)
        self.graph = GraphView(self.graph, vfilt=fl)


    def set_graph_property(self, type, property, name):
        a = self.graph.new_graph_property(type, property)
        self.graph.graph_properties[name] = a


    # Add calculated graph_properties to graph object
    def add_graph_properties(self):
        self.set_graph_property("object", self.deltatau, "deltatau")
        self.set_graph_property("object", self.deltapsi, "deltapsi")
        self.set_graph_property("float", self.cur_iteration, "cur_iteration")
        self.set_graph_property("string", self.graph_name, "graph_name")
        self.set_graph_property("int", self.store_iterations, "store_iterations")
        self.set_graph_property("object", self.top_eigenvalues, "top_eigenvalues")
        self.set_graph_property("object", self.ratios, "ratios")
        self.set_graph_property("int", self.runs, "runs")
        try:
            self.set_graph_property("object", self.apm, "activity_per_month")
            self.set_graph_property("object", self.dx, "delta_activity_per_month")
            self.set_graph_property("object", self.posts, "posts")
            self.set_graph_property("object", self.replies, "replies")
            self.set_graph_property("float", self.a_c, "a_c")
            self.set_graph_property("object", self.a_cs, "a_cs")
            self.set_graph_property("object", self.max_q, "max_q")
            self.set_graph_property("object", self.max_posts_per_day, "max_posts_per_day")
            self.set_graph_property("object", self.g_per_month, "g_per_month")
            self.set_graph_property("object", self.sapm, "simulated_activity_per_month")
        except:
            self.debug_msg("  -> INFO: Could not store empirical activities! ", level=1)


    # Reset attributes between iterations / runs
    def reset_attributes(self, ratio, temp_weights):
        self.graph.vertex_properties["activity"].a = temp_weights
        self.ratio = ratio
        self.converged = False
        self.diverged = False
        self.cur_iteration = 0

    # node weights getter
    def get_node_weights(self, name):
        return np.array(self.graph.vp[name].a)

    # fixed initial activity function
    def init_empirical_activity(self):
        initial_empirical_activity = self.apm[0] / self.graph.num_vertices() / self.a_c
        self.debug_msg("Init Activity: {}".format(initial_empirical_activity), level=1)
        initial_empirical_activity /= sum(self.graph.vp["evcentrality"].a)
        self.debug_msg("Init Activity per edge: {}".format(initial_empirical_activity), level=1)
        ta = initial_empirical_activity * (sum(self.graph.vp["evcentrality"].a)) * self.a_c * self.graph.num_vertices()
        ca = self.apm[0]
        self.debug_msg("Total Activity: {}".format(ta), level=1)
        self.debug_msg("Control Activity: {}".format(ca), level=1)
        for v in self.graph.vertices():
            self.graph.vertex_properties["activity"][v] = initial_empirical_activity * self.graph.vp["evcentrality"][v]
            try:
                self.graph.vertex_properties["weight_initialized"][v] = True
            except:
                pass
        self.debug_msg("Actual Activity: {}".format(np.sum(self.graph.vp["activity"].a)), level=1)

    # node weights setter
    def set_node_weights(self, name, weights):
        self.graph.vertex_properties[name].a = weights


    def update_node_weights(self, name, added_weight):
        self.graph.vertex_properties[name].a += added_weight


    def clear_all_filters(self):
        self.graph.clear_filters()
        self.num_vertices = self.graph.num_vertices()
        self.ratio_ones = [1.0] * self.num_vertices


    # creating random node weights
    def add_node_weights(self, min=0.0, max=0.1, distribution=[1,0,0]):
        self.debug_msg("Adding random weights between {} and {} to nodes.".format(min, max), level=0)
        num_nodes = int(self.graph.num_vertices())
        weights = self.graph.new_vertex_property("double")
        weights_list = [random.uniform(min, max) for x in xrange(num_nodes)]
        random.shuffle(weights_list)
        for ndx, n in enumerate(self.graph.vertices()):
            weights[n] = weights_list[ndx]
        self.graph.vertex_properties["activity"] = weights


    # creating random edge weights
    def add_edge_weights(self, min=0.0, max=0.1):
        self.debug_msg("Adding random weights between {} and {} to edges.".format(min, max), level=0)
        for edge in self.graph.edges():
            self.graph.edge_properties['activity'][edge] = random.uniform(min, max)


    # eigenvalues getter
    def get_eigenvalues(self):
        return np.asarray(self.graph.graph_properties['top_eigenvalues'])

    def calc_eigenvalues(self):
        self.A = adjacency(self.graph, weight=None)
        evals_large_sparse, evecs_large_sparse = largest_eigsh(self.A, 2, which='LM')
        evs = sorted([float(x) for x in evals_large_sparse], reverse=True)[0]
        self.k1 = evs
        self.top_eigenvalues = [evs]
        self.debug_msg("Calculated k1: " + str(self.k1), level=1)


    # store graph to gt
    def store_graph(self, run, postfix=""):
        self.debug_msg("Storing Graph")
        path = config.graph_binary_dir + "/GT/{}/".format(self.graph_name)
        try:
            if not os.path.exists(path):
                self.debug_msg("Created folder: {}".format(path))
                os.makedirs(path)
        except Exception as e:
            self.debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))
        self.graph.save(path + "{}_run_{}{}.gt".format(self.graph_name, run, postfix))


    # sample calculation of g(x)
    def gx(self, q, a, ac):
        return (q*a)/math.sqrt(ac**2+a**2)


    def fx(self, x, ratio):
        return -x*ratio

    # plot g(x) function for multiple values
    def plot_gx(self, min, max):
        x = []
        y = []
        y2 = []
        y3 = []
        y4 = []
        for weight in np.arange(min, max, 0.01):
            y.append(self.gx(1.0, weight, 0.5))
            y2.append(self.gx(1.0, weight, 2.0))
            y3.append(self.gx(2.0, weight, 0.5))
            y4.append(self.gx(2.0, weight, 2.0))
            x.append(weight)
        plt.figure()
        plt.plot(x, y, alpha=1, label="$a_c=0.5$, $q=1.0$")
        plt.plot(x, y2, alpha=1, label="$a_c=2.0$, $q=1.0$")
        plt.plot(x, y3, alpha=1, label="$a_c=0.5$, $q=2.0$")
        plt.plot(x, y4, alpha=1, label="$a_c=2.0$, $q=2.0$")
        ax = plt.axes()
        plt.xlabel("Node Activity ($a$)")
        plt.ylabel("Values for $g(a)$")
        plt.plot([-6, 6], [0, 0], 'k-', lw=0.5, alpha=0.8)
        plt.plot([0.5, 0.5], [0, 3], 'k--', lw=0.5)
        plt.plot([2.0, 2.0], [0, 3], 'k--', lw=0.5)
        plt.plot([0.0, 6], [1.0, 1.0], 'k--', lw=0.5)
        plt.plot([0.0, 6], [2.0, 2.0], 'k--', lw=0.5)
        plt.text(-0.95, 0.95, "$q=1.0$", size=12)
        plt.text(-0.95, 1.95, "$q=2.0$", size=12)
        plt.text(0.1, -0.2, "$a_c=0.5$", size=12)
        plt.text(1.6, -0.2, "$a_c=2.0$", size=12)
        plt.plot([0, 0], [-3, 3], 'k-', lw=0.5, alpha=0.8)
        plt.title("Values for $g(a)$ with weights from ${}$ to ${}$".format(min, max))
        ax.grid(color="gray")
        plt.ylim(-3, 3)
        plt.legend(loc="upper left")
        plt.savefig(config.plot_dir + "functions/" + self.graph_name + "_gx.png")
        plt.close("all")


    def get_fx_weights(self, min, max, lam):
        x = []
        y = []
        for weight in np.arange(min, max+0.1, 0.1):
            y.append(self.fx(weight, lam))
            x.append(weight)
        return x, y

    # plot f(x)
    def plot_fx(self, min, max, k=1):
        plt.figure()
        x,y = self.get_fx_weights(min, max, 1.0)
        plt.plot(x, y, alpha=1, label="$\lambda$=$1.0$")
        x,y = self.get_fx_weights(min, max, 0.5)
        plt.plot(x, y, alpha=1, label="$\lambda$=$0.5$")
        x,y = self.get_fx_weights(min, max, 0.1)
        plt.plot(x, y, alpha=1, label="$\lambda$=$0.1$")
        x,y = self.get_fx_weights(min, max, 1.5)
        plt.plot(x, y, alpha=1, label="$\lambda$=$1.5$")


        ax = plt.axes()
        plt.xlabel("Node Activity (a)")
        plt.ylabel("Values for f(a)")
        plt.title("Values for f(a) with weights from ${}$ to ${}$".format(min, max))
        ax.grid(color="gray")
        plt.plot([-1, 1], [0, 0], 'k-', lw=0.5, alpha=0.8)
        plt.plot([0, 0], [-1.5, 1.5], 'k-', lw=0.5, alpha=0.8)
        plt.legend()
        plt.savefig(config.plot_dir + "functions/" + self.graph_name + "_fx.png")
        plt.close("all")


    # plot f(x)
    def plot_fx_weight(self, min, max, k=0.5):
        x = []
        prev_val = 10
        y = [prev_val]
        for i in xrange(10):
            prev_val *= k
            y.append(prev_val)
            x.append(i)
        x.append(10)
        plt.figure()
        plt.plot(x, y, alpha=1)
        ax = plt.axes()
        plt.xlabel("Time $t$")
        plt.ylabel("Values for f(a)")
        plt.title("Values for f(a) with weight=${}$ and $\lambda$=${}$".format(10, 0.5))
        ax.grid(color="gray")
        plt.savefig(config.plot_dir + "functions/" + self.graph_name + "_fx_weight.png")
        plt.close("all")


    # getter for laplacian matrix (not needed)
    def get_laplacian(self):
        return laplacian(self.graph)


    # calculate eigenvector centrality
    def calc_ev_centrality(self, max_iter, selector):
        try:
            return self.graph.vertex_properties[selector]
        except:
            ev, ev_centrality = eigenvector(self.graph, weight=None, max_iter = max_iter)
            return ev_centrality

    def calculate_ratios(self):
        for i in xrange(len(self.dx)):
            activity_current = self.apm[i]
            activity_next = activity_current-self.dx[i]
            self.ratio = self.k1 - math.log(activity_next/activity_current) / self.deltapsi
            #self.ratio -= 0.03 * activity_current / (self.a_c * self.num_vertices)
            self.ratios.append(self.ratio)
        self.debug_msg("ratios ({}): {}".format(len(self.ratios), self.ratios), level=1)

    def set_ratio(self, index):
        self.ratio_index = index
        self.ratio = self.ratios[index]

    def reset_tau_iter(self, iter=0):
        self.tau_iter = iter

    def activity_dynamics(self, store_weights=False, store_taus=False, empirical=False):
        # Collect required input
        activity_weight = np.asarray(self.get_node_weights("activity"))
        # Calculate deltax
        ratio_ones = (self.ratio * np.asarray(self.ones_ratio))
        intrinsic_decay = self.activity_decay(activity_weight, ratio_ones)
        extrinsic_influence = self.peer_influence(activity_weight)
        activity_delta = (intrinsic_decay + extrinsic_influence)*self.deltatau

        t = 1.0
        # Check if already converged/diverged
        if self.cur_iteration % self.store_iterations == 0:
            t = np.dot(activity_delta, activity_delta)
        # Debug output & convergence/divergence criteria check
        if t < self.converge_at and not empirical:
            self.debug_msg(" \x1b[32m>>>\x1b[00m Simulation for \x1b[32m'{}'\x1b[00m with \x1b[34mratio={}\x1b[00m and "
                            "\x1b[34mdtau={}\x1b[00m \x1b[34mconverged\x1b[00m at \x1b[34m{}\x1b[00m with "
                            "\x1b[34m{}\x1b[00m".format(self.graph_name, self.ratio, self.deltatau, self.cur_iteration+1,
                                                        t), level=1)
            self.converged = True
        if (t == float("Inf") or t == float("NaN")) and not empirical:
            self.debug_msg(" \x1b[31m>>>\x1b[00m Simulation for \x1b[32m'{}'\x1b[00m with \x1b[34mratio={}\x1b[00m and "
                            "\x1b[34mdtau={}\x1b[00m \x1b[31mdiverged\x1b[00m at \x1b[34m{}\x1b[00m with "
                            "\x1b[34m{}\x1b[00m".format(self.graph_name, self.ratio, self.deltatau, self.cur_iteration+1,
                                                        t), level=1)
            self.diverged = True
        # Set new weights
        self.update_node_weights("activity", activity_delta)

        # Store weights to file
        if ((store_weights and self.cur_iteration % self.store_iterations == 0) and not empirical) or ((self.converged or self.diverged)
                                                                                   and not empirical):
            self.weights_file.write(("\t").join(["%.8f" % x for x in self.get_node_weights("activity")]) + "\n")
        elif ((store_weights and self.cur_iteration % self.store_iterations == 0) and empirical) or ((self.converged or self.diverged)
                                                                                   and empirical):
            #self.debug_msg(" --> Sum of weights at \x1b[32m{}\x1b[30m is \x1b[32m{}\x1b[30m".format(self.cur_iteration, str(sum(activity_weight + activity_delta) * self.a_c * self.graph.num_vertices())), level=1)
            self.weights_file.write(str(sum(activity_weight + activity_delta) * self.a_c * self.graph.num_vertices()) + "\n")

        # Store taus to file
        if store_taus and empirical and self.cur_iteration % self.store_iterations == 0:
            tau = (float(self.tau_iter)+1)*self.deltatau/self.deltapsi
            tau += self.ratio_index
            self.taus_file.write(str(tau) + "\n")
        self.tau_iter += 1

        # Increment current iteration counter
        self.cur_iteration += 1


    def peer_influence(self, x):
        pi = ((1.0 * x)/(np.sqrt(1.0+x**2)))
        return pi * self.A


    def activity_decay(self, x, ratio):
        return -x*ratio

    def debug_msg(self, msg, level=0):
        if self.debug_level <= level:
            print "  \x1b[35m-NWK-\x1b[00m [\x1b[36m{}\x1b[00m][\x1b[32m{}\x1b[00m] \x1b[33m{}\x1b[00m".format(
                datetime.datetime.now().strftime("%H:%M:%S"), self.run, msg)

    def update_binary_graph(self, rand_iter, save_specific=True):
        # Method needed to avoid race condition!
        try:
            self.store_graph(rand_iter, save_specific=True)
        except Exception as e:
            self.debug_msg(e.message, level=0)
            self.debug_msg(" ### Sleeping for 100 seconds before trying to store again!", level=0)
            sleep(100)
            self.update_binary_graph(rand_iter, save_specific)

    def debug_gt(self):
        gps = self.graph.gp.keys()
        vps = self.graph.vp.keys()
        eps = self.graph.ep.keys()
        self.debug_msg(" >> Inspecting graph properties: {}".format((", ").join(gps)), level=1)
        for gp_k in gps:
            self.debug_msg("   \x1b[36m- {}:\x1b[00m {}".format(gp_k, self.graph.gp[gp_k]), level=1)
        self.debug_msg(" >> Inspecting vertex properties: {}".format((", ").join(vps)), level=1)
        for vp_k in vps:
            self.debug_msg("   \x1b[32m- {}:\x1b[00m {}".format(vp_k, self.graph.vp[vp_k]), level=1)
        self.debug_msg(" >> Inspecting edge properties: {}".format((", ").join(eps)), level=1)
        for ep_k in eps:
            self.debug_msg("   \x1b[37m- {}:\x1b[00m {}".format(ep_k, self.graph.ep[ep_k]), level=1)
        print "Sum Posts: ", sum(self.graph.gp["posts"])
        print "Sum Replies: ", sum(self.graph.gp["replies"])


    def prepare_eigenvalues(self):
        self.top_eigenvalues = self.get_eigenvalues()
        self.k1 = max(self.top_eigenvalues)


    def load_graph_save(self, fpath):
        try:
            self.load_graph(fpath)
        except Exception as e:
            self.debug_msg(e.message, level=0)
            self.debug_msg(" ### Sleeping for 100 seconds before trying to load again!", level=0)
            sleep(100)
            self.load_graph(fpath)


    def load_graph(self, fpath):
        self.debug_msg("Loading GT", level=0)
        self.graph = load_graph(fpath)
        remove_self_loops(self.graph)
        remove_parallel_edges(self.graph)
        self.debug_msg("  --> Creating ones vector", level=0)
        self.ones_ratio = [1.0] * self.graph.num_vertices()
        self.debug_msg("  --> Getting Adjacency Matrix", level=0)
        self.A = adjacency(self.graph, weight=None)
        self.num_vertices = self.graph.num_vertices()
        self.num_edges = self.graph.num_edges()
        self.debug_msg("  --> Counted {} vertices".format(self.num_vertices), level=0)
        self.debug_msg("  --> Counted {} edges".format(self.num_edges), level=0)

