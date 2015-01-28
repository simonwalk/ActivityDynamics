from __future__ import division
__author__ = 'Simon Walk'

from itertools import izip
import math
import random
import config.config as config
import wrapper as wrapper
from collections import defaultdict
from numpy.lib.type_check import real, imag
import numpy as np
import scipy
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


class Network:

    # Create Network Object with default values
    # @params:
    # iterations          =   Max number of diffusion iterations if not converged/diverged before.
    #   directed            =   Is graph directed?
    #   graph_name          =   Name of graph, mainly used for storing plots and figures
    #   intrinsic_activity  =   Internal activity influence of nodes
    #   extrinsic_activity  =   External activity influence of nodes
    #   fx                  =   Diffusion function to choose! Available: "gossip", "linear" and "quadratic"
    #   Deprecated (soon to be removed):
    #   plot_with_labels    =   Boolean switch for plotting functions
    #   debug               =   Boolean switch for outputting debug info
    def __init__(self, iterations, directed, graph_name, intrinsic_activity=0.25, extrinsic_activity=0.25, fx="linear",
                 run=1, converge_at=1e-13, plot_with_labels=False, debug=False):
        self.num_iterations = iterations
        self.cur_iteration = 0
        self.directed = directed
        self.debug = debug
        self.a = intrinsic_activity
        self.b = extrinsic_activity
        self.name_to_id = {}
        self.plot_with_labels = plot_with_labels
        self.graph = nx.Graph()
        self.converged = False
        self.delta_dot = 1.0
        self.fx_selector = fx
        self.graph_name = graph_name
        self.diverged = False
        self.converge_at = float(converge_at)
        self.run = run
        self.ones = [1.0] * len(self.graph)
        self.L = ""
        self.A = ""


    def open_weights_file(self):
        self.weights_file_path = config.graph_source_dir + "weights/" + self.fx_selector + "/" + self.graph_name + \
                                 "_" + str(self.num_iterations) + "_iterations_" + str(self.a).replace(".","") + "_" + \
                                 str(self.b).replace(".", "") + "_run_" + str(self.run) + ".txt"
        self.weights_file = open(self.weights_file_path, "wb")
        self.weights_file.write(("\t").join(["%12f" % x for x in self.get_node_weights().values()]) + "\n")


    def close_weights_file(self):
        self.weights_file.close()


    def reset_attributes(self, a, b, temp_weights):
        #wrapper.set_node_attribute("activity", temp_weights)
        nx.set_node_attributes(self.graph, "activity", temp_weights)
        self.a = a
        self.b = b
        self.converged = False
        self.diverged = False
        self.cur_iteration = 0



    def get_node_weights(self):
        return nx.get_node_attributes(self.graph, 'activity')


    def set_node_weights(self, name, weights):
        nx.set_node_attributes(self.graph, name, weights)


    # randomly create node weights
    def add_node_weights(self, min=0.0, max=0.1):
        self.debug_msg("Adding random weights between {} and {} to nodes.".format(min, max))
        num_nodes = self.graph.number_of_nodes()
        for i in xrange(0, num_nodes):
            self.graph.node[i]['activity'] = random.uniform(min, max)


    # randomly create edge weights
    def add_edge_weights(self, min=0.0, max=0.1):
        self.debug_msg("Adding random weights between {} and {} to edges.".format(min, max))
        for e in self.graph.edges_iter(data=True):
            e[2]['activity'] = random.uniform(min, max)


    # def add_node(self, node_name, node_weight=1):
    #     if node_name not in self.graph.nodes():
    #         self.graph.add_node(node_name, weight=node_weight)
    #
    #
    # def add_edge(self, node_name1, node_name2, weight):
    #     self.graph.add_edge(node_name1, node_name2)
    #
    #
    # def import_graph(self, fname):
    #     f = open(config.graph_source_dir + fname, "rb")
    #     for l_index, line in enumerate(f):
    #         if l_index < 1:
    #             continue
    #         sline = line.strip().split("\t")
    #         self.add_node(sline[0], sline[3])
    #         self.add_node(sline[1], sline[3])
    #         self.add_edge(sline[0], sline[1], sline[2])
    #     f.close()

    #
    # def import_octave_adjacency_matrix(self, fname):
    #     f = open(config.graph_source_dir + fname, "rb")
    #     edge_list = []
    #     row_counter = 0
    #     for l_index, line in enumerate(f):
    #         if line.startswith("#"):
    #             continue
    #         sline = [int(x) for x in line.strip().split(" ")]
    #         for idx, x in enumerate(sline):
    #             if x == 1:
    #                 edge_list.append((row_counter, idx))
    #         row_counter += 1
    #     f.close()
    #     self.graph.add_edges_from(edge_list)
    #
    #
    # def add_octave_node_weights(self, fname):
    #     f = open(config.graph_source_dir + fname, "rb")
    #     id_counter = 0
    #     for l_index, line in enumerate(f):
    #         if line.startswith("#"):
    #             continue
    #         sline = line.strip()
    #         self.graph.node[id_counter]["weight"] = float(sline)
    #         id_counter += 1
    #     f.close()


    def store_graph(self):
        nx.write_gml(self.graph, config.graph_binary_dir +
                     self.fx_selector + "/" + self.graph_name + "_" +
                     str(self.cur_iteration) +
                     "_iterations_" + str(self.a).replace(".", "") + "_" +
                     str(self.b).replace(".", "") +
                     "_run_"+str(self.run)+"_weights.gml")


    def fx(self, x, a, k):
        if self.fx_selector == "linear":
            return -a * x
        if self.fx_selector == "quadratic":
            y = -a * x * (1 - x)
            if y > 0:
                if x < 0:
                    y = (-1 * x) * k
                else:
                    y = (-1 * x + 1) * k
            return y
        if self.fx_selector == "gossip":
            return a * (1 - x)


    def gx(self, weight, b=1.0):
        return (b / (1 + math.exp(-weight))) - b / 2


    def plot_gx(self, min, max):
        x = []
        y = []
        for weight in np.arange(min, max, 0.01):
            y.append(self.gx(weight, 1.0))
            x.append(weight)
        plt.figure()
        plt.plot(x, y, alpha=1)
        ax = plt.axes()
        plt.xlabel("Node Activity")
        plt.ylabel("Values for g(x)")
        plt.title("Values for g(x) with weights from {} to {}".format(min, max))
        ax.grid(color="gray")
        plt.savefig("plots/functions/" + self.graph_name + "_gx.png")
        plt.close("all")


    def plot_fx(self, min, max, k=1.9):
        x = []
        y = []
        for weight in np.arange(min, max, 0.01):
            y.append(self.fx(weight, 1, k))
            x.append(weight)
        plt.figure()
        plt.plot(x, y, alpha=1)
        ax = plt.axes()
        plt.xlabel("Node Activity")
        plt.ylabel("Values for f(x)")
        plt.title("Values for f(x) with weights from {} to {}".format(min, max))
        ax.grid(color="gray")
        plt.plot([-1, 2], [0, 0], 'k-', lw=1)
        plt.plot([0, 0], [-2, 2], 'k-', lw=1)
        plt.savefig("plots/functions/" + self.graph_name + "_fx.png")
        plt.close("all")


    def get_laplacian(self):
        return nx.laplacian_matrix(self.graph, weight="None")


    def propagate_diffusion_matrix(self, store_weights=False, weights_debug=False):

        x = np.array(self.get_node_weights().values())

        exp = math.exp
        g_x = (self.b / (1 + np.array([exp(-w) for w in x]))) - self.b / 2.0

        if self.fx_selector == "linear":
            deltax = (-self.a * x) + (self.A * g_x)
        elif self.fx_selector == "quadratic":
            deltax = (-self.a * x * (self.ones - x)) + (-self.b * self.L * g_x)
        elif self.fx_selector == "gossip":
            deltax = self.a * (self.ones - x) + (-self.b * self.L * g_x)

        if self.cur_iteration % 1000 == 0:
            t = np.dot(deltax, deltax)
        else:
            t = self.converge_at + 1

        self.delta_dot = t

        if t < self.converge_at:
            self.debug_msg("Converged at {} with {}".format(self.cur_iteration, t))
            self.converged = True
        if t == float("Inf") or t == float("NaN"):
            self.debug_msg("Diverged at {} with {}".format(self.cur_iteration, t))
            self.diverged = True

        self.set_node_weights("activity", dict(enumerate(x + deltax)))
        self.cur_iteration += 1

        if store_weights:
            self.weights_file.write(("\t").join(["%12f" % x for x in self.get_node_weights().values()]) + "\n")


    # Debug methods
    def print_weights(self):
        self.debug_msg("  -- {}".format(list(self.get_node_weights().values())))


    def get_b_vals(self, aval, num_vals=10):
        tmp = sorted(self.ew, reverse=True)[:num_vals]
        b_vals = []
        b_vals.append(float((4.0*aval)/(tmp[0])*0.9))
        for a,b in izip(tmp, tmp[1:]):
            ratio = (a+b)/2.0
            b_vals.append(float((4.0*aval)/(ratio)))
        return b_vals


    def debug_msg(self, msg):
        print "  -NWK (NX)- [{}] {}".format(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), msg)


    def calc_eigenvalues(self):
        self.debug_msg("Starting to calculate Eigenvalues")
        self.ew, self.ev = scipy.linalg.eig(nx.adjacency_matrix(self.graph, weight="None").todense())
        self.debug_msg("Finished calculating Eigenvalues")


    def plot_eigenvalues(self):
        plt.figure(figsize=(8, 2))
        plt.scatter(real(self.ew), imag(self.ew), c=abs(self.ew))
        plt.xlabel(r"$\operatorname{Re}(\kappa)$")
        plt.ylabel(r"$\operatorname{Im}(\kappa)$")
        plt.tight_layout()
        plt.savefig("plots/eigenvalues/" + self.graph_name + "_adjacency_spectrum.pdf")
        plt.close("all")


    def load_graph(self, fpath):
        self.debug_msg("Loading GML")
        self.graph = nx.read_gml(fpath)
        self.ones = [1.0] * len(self.graph)
        self.debug_msg("Getting L and A")
        self.L = nx.laplacian_matrix(self.graph, weight="None")
        self.A = nx.adjacency_matrix(self.graph, weight="None")