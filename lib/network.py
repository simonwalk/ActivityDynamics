from __future__ import division
from time import sleep

__author__ = 'Simon Walk, Florian Geigl, Denis Helic, Philipp Koncar'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

from lib.util import *
from config import config

import math
import random
import numpy as np
from numpy.lib.type_check import real, imag
import datetime

from graph_tool.all import *

import matplotlib
matplotlib.use('Agg')
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
    #   ratio               =   The ratio for the spreading of our activities.
    #   deltatau            =   The time that each iteration "represents".
    #   debug_level         =   The level of debug messages to be displayed.
    #   store_iterations    =   The interval for storing iterations (1 = all, 2 = every other, etc.)
    # Deprecated (soon to be removed):
    #   plot_with_labels    =   Boolean switch for plotting functions
    #   debug               =   Boolean switch for outputting debug info
    #   iterations          =   Max number of activity spreading iterations if not converged/diverged before.
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


    def calc_acs(self, ac_per_taus=None, min_ac=None):
        if ac_per_taus is None:
            self.a_cs = [max((np.mean(self.replies) + np.mean(self.posts)) / self.num_vertices, min_ac)] * (len(self.replies)-1)
        else:
            for i in xrange(len(self.replies)-ac_per_taus):
                j = i + ac_per_taus
                curr_ac = (np.mean(self.replies[i:j]) + np.mean(self.posts[i:j])) / self.num_vertices
                for k in xrange(i+ac_per_taus):
                    self.a_cs.append(curr_ac)
        self.set_ac(0)

    def set_ac(self, index):
        self.a_c = self.a_cs[index]


    def calc_ac(self, start_tau=0, end_tau=None, min_ac=40):
        replies = self.replies[start_tau:end_tau]
        posts = self.posts[start_tau:end_tau]
        return max((np.mean(replies) + np.mean(posts)) / self.num_vertices, min_ac)


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
        self.posts_per_user_per_day = []
        self.a_cs = []

        f = open(path, "rb")
        for ldx, line in enumerate(f):
            if ldx < 1:
                continue
            el = line.strip().split("\t")
            self.dx.append(float(el[0]))
            self.apm.append(float(el[1]))
            self.posts.append(float(el[2]))
            self.replies.append(float(el[3]))
            self.num_users.append(float(el[4]))
            self.posts_per_user_per_day.append(float(el[2])/float(el[4])/30.0)
        f.close()

        self.calc_acs(ac_per_taus)
        self.max_posts_per_day = self.calc_max_posts_per_day(start_tau, end_tau)
        self.g_per_month = self.calc_g_per_month()
        self.max_q = self.calc_max_q()
        self.mu = self.max_q / self.a_c
        self.deltapsi = self.mu
        self.debug_msg("max_q: {}".format(self.max_q), level=1)
        self.debug_msg("deltapsi: {}".format(self.deltapsi), level=1)
        self.debug_msg("max_posts_per_day: {}".format(self.max_posts_per_day), level=1)
        self.debug_msg("a_c: {}".format(self.a_c), level=1)
        self.debug_msg("kappa_1: {}".format(self.k1), level=1)


    # Creating all necessary folders for storing results, plots and figures
    def create_folders(self):
        folders = [config.graph_source_dir+"weights/"+self.graph_name+"/",
                   config.plot_dir + "weights_over_time/" + self.graph_name + "/",
                   #config.plot_dir + "scatterplots/" + self.graph_name + "/",
                   #config.plot_dir + "active_inactive/" + self.graph_name + "/",
                   #config.plot_dir + "percentage_comp/" + self.graph_name + "/",
                   config.plot_dir + "average_weights_over_tau/" + self.graph_name + "/",
                   config.plot_dir + "ratios_over_time/" + self.graph_name + "/",]
                   #config.plot_dir + "errors_over_time/" + self.graph_name + "/"]
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
        iname = self.graph_name + "_" + str(self.store_iterations) +"_"+\
                str(float(self.deltatau)).replace(".", "") + "_" + str(self.ratio).replace(".", "") + "_run_" + \
                str(self.run) + "_intrinsic.txt"
        ename = self.graph_name + "_" + str(self.store_iterations) +"_"+\
                str(float(self.deltatau)).replace(".", "") + "_" + str(self.ratio).replace(".", "") + "_run_" + \
                str(self.run) + "_extrinsic.txt"

        self.weights_file_path = folder+wname
        self.intrinsic_file_path = folder+iname
        self.extrinsic_file_path = folder+ename

        self.weights_file = open(self.weights_file_path, "wb")
        self.intrinsic_file = open(self.intrinsic_file_path, "wb")
        self.extrinsic_file = open(self.extrinsic_file_path, "wb")


    def write_weights_to_file(self):
        self.weights_file.write(("\t").join(["%.8f" % float(x) for x in self.get_node_weights("activity")]) + "\n")


    def write_summed_weights_to_file(self):
        self.weights_file.write(str(sum(self.get_node_weights("activity"))) + "\n")
        self.intrinsic_file.write("0"+"\n")
        self.extrinsic_file.write("0"+"\n")

    def close_weights_files(self):
        self.weights_file.close()
        self.intrinsic_file.close()
        self.extrinsic_file.close()

    #TODO Replace with Filter + Purge!
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

        except:
            self.debug_msg("  -> INFO: Could not store empirical activities!", level=1)


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


    # init empirical weight as average over all nodes
    def init_empirical_activity(self, ac_multiplicator=1):
        initial_empirical_activity = self.apm[0]/self.num_vertices/(self.a_c*ac_multiplicator)
        #print initial_empirical_activity
        random_init_nodes = self.num_users[0]

        # reset activity!
        # Todo: Avoid reset loop!
        for v in self.graph.vertices():
            self.graph.vp["activity"][v] = 0.0
        # randomly initiate minimal activity
        for v_idx in random.sample(xrange(self.num_vertices), int(random_init_nodes)):
            v = self.graph.vertex(v_idx)
            self.graph.vp["activity"][v] = initial_empirical_activity
        

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
        num_nodes = float(self.graph.num_vertices())
        weights = self.graph.new_vertex_property("double")
        num_lurker = int(math.ceil(num_nodes*distribution[0]))
        weights_list = [random.uniform(min, max) for x in xrange(num_lurker)]
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


    # store graph to gt
    def store_graph(self, run, save_specific=False, postfix=""):
        self.debug_msg("Storing Graph")

        path_random = config.graph_binary_dir + "/GT/{}/".format(self.graph_name)
        path_spec = config.graph_binary_dir + "/GT/{}/".format(self.graph_name.replace("RAND", "SPEC"))

        try:
            if not os.path.exists(path_random):
                self.debug_msg("Created folder: {}".format(path_random))
                os.makedirs(path_random)
        except Exception as e:
            self.debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))

        self.graph.save(path_random + "{}_run_{}{}.gt".format(self.graph_name, run, postfix))
        if save_specific:
            try:

                if not os.path.exists(path_spec):
                    self.debug_msg("Created folder: {}".format(path_spec))
                    os.makedirs(path_spec)
            except Exception as e:
                self.debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))
            self.graph.save(path_spec + "{}_run_{}{}.gt".format(self.graph_name.replace("RAND", "SPEC"), run, postfix))


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


    # # select random nodes equal to percentage for increasing/decreasing ratio
    # def get_percentage_of_nodes(self, percentage = 0.01, mult_factor = 0.9, specific=False, selector="evc",
    #                             max_iter=100000, spec_func=None):
    #
    #     num_nodes = self.graph.num_vertices()
    #     selection_limiter = int(math.ceil(num_nodes * percentage))
    #     self.debug_msg("   *** Collecting a total of {} ({}%) random nodes!".format(self.run, selection_limiter,
    #                                                                                 round(percentage*100)),level=0)
    #     if percentage == 0.0:
    #         return
    #     if specific:
    #         if spec_func == None:
    #             spec_func = self.calc_ev_centrality
    #         self.graph.vertex_properties[selector] = spec_func(max_iter, selector)
    #     if not specific:
    #         counter = 0
    #         rints = []
    #         while counter < selection_limiter:
    #             rint = random.randint(0, num_nodes-1)
    #             if rint in rints:
    #                 continue
    #             else:
    #                 self.ones_ratio[rint] *= float(mult_factor)
    #                 rints.append(rint)
    #                 counter += 1
    #     else:
    #         rints = zip(*heapq.nlargest(selection_limiter, enumerate(self.graph.vertex_properties[selector].a),
    #                                     key=operator.itemgetter(1)))[0]
    #         for i in rints:
    #             self.ones_ratio[i] *= float(mult_factor)


    def calculate_ratios(self):
        for i in xrange(len(self.apm)-1):
            activity_current = self.apm[i]
            activity_next = activity_current-self.dx[i]
            self.ratio = self.k1 - math.log(activity_next/activity_current) / self.deltapsi
            self.ratio -= 0.01 * activity_current / (self.a_c * self.num_vertices)
            self.ratios.append(self.ratio)
        self.debug_msg("ratios ({}): ".format(len(self.ratios), self.ratios), level=1)


    def set_ratio(self, index):
        self.ratio_index = index
        self.ratio = self.ratios[index]


    def activity_dynamics(self, store_weights=False, empirical=False):
        # Collect required input
        activity_weight = np.asarray(self.get_node_weights("activity"))
        # Calculate deltax
        ratio_ones = (self.ratio * np.asarray(self.ones_ratio))
        intrinsic_decay = self.activity_decay(activity_weight, ratio_ones)
        extrinsic_influence = self.peer_influence(activity_weight)
        activity_delta = (intrinsic_decay + extrinsic_influence) * self.deltatau
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
            self.intrinsic_file.write(("\t").join(["%.8f" % x for x in intrinsic_decay + activity_weight]) + "\n")
            self.extrinsic_file.write(("\t").join(["%.8f" % x for x in extrinsic_influence + activity_weight]) + "\n")
        elif ((store_weights and self.cur_iteration % self.store_iterations == 0) and empirical) or ((self.converged or self.diverged)
                                                                                   and empirical):
            #print "emp"
            self.weights_file.write(str(sum(activity_weight + activity_delta)) + "\n")
            self.intrinsic_file.write(str(abs(sum(intrinsic_decay))*self.deltatau) + "\n")
            self.extrinsic_file.write(str(abs(sum(extrinsic_influence))*self.deltatau) + "\n")
        # Increment current iteration counter
        self.cur_iteration += 1


    def peer_influence(self, x):
        pi = ((1.0 * x)/(np.sqrt(1.0+x**2)))
        return pi * self.A


    def activity_decay(self, x, ratio):
        return -x*ratio


    # def activity_dynamics_empirical(self, store_weights=False):
    #     # Collect required input
    #     x = np.asarray(self.get_node_weights("activity"))
    #     # Calculate deltax
    #     ratio_ones = (self.ratio * np.asarray(self.ones_ratio))
    #     fx = self.activity_decay(x, ratio_ones)
    #     gx = self.peer_influence(x)
    #     #print "fx ", fx
    #     #print "gx ", gx
    #     deltax = (fx + gx) * self.deltatau
    #     # Set new weights
    #     total_weights = sum(x + deltax)
    #     self.debug_msg(" --> Sum of weights: {}".format(total_weights), level=2)
    #     self.update_node_weights("activity", deltax)
    #     if (store_weights and self.cur_iteration % self.store_iterations == 0) or (self.converged or self.diverged):
    #         self.weights_file.write(str(total_weights) + "\n")
    #     # Increment current iteration counter
    #     self.cur_iteration += 1


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

        print "sum posts: ", sum(self.graph.gp["posts"])
        print "sum replies: ", sum(self.graph.gp["replies"])

    def prepare_eigenvalues(self):
        self.top_eigenvalues = self.get_eigenvalues()
        self.k1 = max(self.top_eigenvalues)


    def plot_eigenvalues(self):
        plt.figure(figsize=(8, 2))
        plt.scatter(real(self.top_eigenvalues), imag(self.top_eigenvalues), c=abs(self.top_eigenvalues))
        plt.xlabel(r"$Re(\kappa)$")
        plt.ylabel(r"$Im(\kappa)$")
        plt.tight_layout()
        plt.savefig("../DynamicNetworksResults/plots/eigenvalues/" + self.graph_name + "_adjacency_spectrum.pdf")
        plt.close("all")

    def load_graph_save(self, fpath):
        try:
            self.load_graph(fpath)
        except Exception as e:
            self.debug_msg(e.message, level=0)
            self.debug_msg(" ### Sleeping for 100 seconds before trying to store again!", level=0)
            sleep(100)
            self.load_graph(fpath)


    def load_graph(self, fpath):
        self.debug_msg("Loading GT", level=0)
        self.graph = load_graph(fpath)
        #self.reduce_to_largest_component()
        self.debug_msg("  --> Creating ones vector", level=0)
        self.ones_ratio = [1.0] * self.graph.num_vertices()
        self.debug_msg("  --> Getting Laplacian Matrix", level=0)
        self.L = laplacian(self.graph, weight=None)
        self.debug_msg("  --> Getting Adjacency Matrix", level=0)
        self.A = adjacency(self.graph, weight=None)
        #self.debug_msg("  --> Getting Degree Vector", level=0)
        #self.degree_vector = self.graph.vertex_properties["degree"].a
        self.num_vertices = self.graph.num_vertices()
        self.num_edges = self.graph.num_edges()
        self.debug_msg("  --> Counted {} Vertices".format(self.num_vertices), level=0)
        self.debug_msg("  --> Counted {} Edges".format(self.num_edges), level=0)


    # Stuff from Philipp Koncar...
    def calc_eigenvalues(self, num_ev=100):
        if num_ev > 100:
            num_ev = 100
        self.debug_msg("Starting calculation of {} Eigenvalues".format(num_ev))
        evals_large_sparse, evecs_large_sparse = largest_eigsh(self.A, num_ev * 2, which='LM')
        self.debug_msg("Finished calculating Eigenvalues")
        weights = sorted([float(x) for x in evals_large_sparse], reverse=True)[:num_ev]
        temp_str = ""
        for e in weights:
            temp_str += str(e) + ", "
        temp_str = temp_str[:-2]
        self.graph.graph_properties["ew"] = self.graph.new_graph_property("string", temp_str)
        self.prepare_eigenvalues()



    def update_vertex_properties(self, max_iter_ev=1000, max_iter_hits=1000):
        self.debug_msg("Starting to update vertex properties... ")

        self.debug_msg("\x1b[33m  ++ Updating PageRank\x1b[00m")
        pr = pagerank(self.graph)
        self.graph.vertex_properties["pagerank"] = pr

        self.debug_msg("\x1b[33m  ++ Updating Clustering Coefficient\x1b[00m")
        clustering = local_clustering(self.graph)
        self.graph.vertex_properties["clustercoeff"] = clustering

        self.debug_msg("\x1b[33m  ++ Updating Eigenvector Centrality\x1b[00m")
        ev, ev_centrality = eigenvector(self.graph, weight=None, max_iter=max_iter_ev)
        self.graph.vertex_properties["evcentrality"] = ev_centrality

        #self.debug_msg("\x1b[33m  ++ Updating HITS\x1b[00m")
        #eig, authorities, hubs = hits(self.graph, weight=None, max_iter=max_iter_hits)
        #self.graph.vertex_properties["authorities"] = authorities
        #self.graph.vertex_properties["hubs"] = hubs

        self.debug_msg("\x1b[33m  ++ Updating Degree Property Map\x1b[00m")
        degree = self.graph.degree_property_map("total")
        self.graph.vertex_properties["degree"] = degree

        self.debug_msg("Done.")

    # Calculates the current percentage of connectivity
    def current_percent_of_connectivity(self):
        num_of_cur_edges = self.graph.num_edges()
        num_of_com_edges = ((self.graph.num_vertices() - 1) * self.graph.num_vertices()) / 2
        percentage = (num_of_cur_edges / num_of_com_edges) * 100
        percentage = round(percentage, 2)
        self.debug_msg("Finished calculating percentage of the current connectivity of the graph: " + str(percentage) + "%")
        return percentage

    # Calculates and returns the numbers of edges needed to reach target percentage
    def edge_variance_to_percent(self, target_percentage):
        num_of_com_edges = ((self.graph.num_vertices() - 1) * self.graph.num_vertices()) / 2
        target_num_of_edges = (target_percentage /100) * num_of_com_edges
        num_to_add = target_num_of_edges - self.graph.num_edges()
        num_to_add = int(round(num_to_add))
        self.debug_msg("Finished calculating edge variance to " + str(target_percentage) + "%: " + str(num_to_add))
        return num_to_add

    # Returns the maximum possible number of edges of the graph.
    def maximum_edges_of_graph(self):
        return ((self.graph.num_vertices() - 1) * self.graph.num_vertices()) // 2

    # Calculates and returns the minimum number of edges for a spanning tree
    def minimum_edges_for_spanning_tree(self):
        return self.graph.num_vertices() - 1

    # Calculates and returns the minimum percentage for a spanning tree
    def minimum_percent_for_spanning_tree(self):
        min_edges = self.minimum_edges_for_spanning_tree()
        num_of_com_edges = ((self.graph.num_vertices() - 1) * self.graph.num_vertices()) / 2
        percentage = (min_edges / num_of_com_edges) * 100
        return round(percentage,2)

    # Updates the adjacency matrix contained in self.A
    def update_adjacency_matrix(self):
        self.debug_msg("Updating adjacency matrix...")
        self.A = adjacency(self.graph, weight=None)
        self.debug_msg("Done.")

    # Returns an array containing the amount of edges added/removed per iteration.
    # @params:
    # target_num = The target number of edges.
    # num_iter = The number of iterations.
    def get_num_edges_per_iterations_array(self, target_num, num_iter):
        if target_num == self.graph.num_edges():
            self.debug_msg("Target number and current number of edges are equal!")
            sys.exit("Program exit")
        elif target_num < self.graph.num_edges():
            num_edges = self.graph.num_edges() - target_num
        else:
            num_edges = target_num - self.graph.num_edges()
        per_iter = num_edges // num_iter
        rest = num_edges % num_iter
        num_array = [per_iter for i in range(0, num_iter)]
        for i in range(0, rest):
            num_array[i] += 1
        return num_array

    # Calculates an edge list of the graph. Depending on the given mode, either missing or current edges are
    # calculated.
    # @params:
    # mode = Specifies whether missing or current edges are calculated. Possible modes: current,
    #        current_excl_st (with this mode edges belonging to a random spanning tree are not contained in the list),
    #        missing
    # strategy = Specifies the way the list is sorted. Possible strategies: random, high to high, low to low, high to low
    # sorted_values = A list of sorted vertices is needed to sort the edge list.
    def calculate_edge_list(self, mode, strategy, sorted_values=None):
        self.debug_msg("Starting calculation of edge list (mode = " + mode + ", strategy = " + strategy + ")...")
        edge_tuples = []
        adj_matrix = self.A.todense()
        if "current" in mode:
            check_value = 1.0
            num_of_edges = self.graph.num_edges()
        else:
            check_value = 0.0
            num_of_edges = (((self.graph.num_vertices() - 1) * self.graph.num_vertices()) / 2) - self.graph.num_edges()
        for i in range(0, self.graph.num_vertices()):
            if i % int((self.graph.num_vertices() / 10)) == 0 and i != 0:
                self.debug_msg("  --> Covered " + str(int((float(i)/self.graph.num_vertices()) * 100)) + "% of vertices.")
            if strategy == "random":
                for j in range(0, self.graph.num_vertices()):
                    if adj_matrix[i, j] == check_value and i < j:
                        edge_tuples.append((i, j))
            elif strategy == "high_to_high" or mode == "low_to_low":
                for j in range(i + 1, self.graph.num_vertices()):
                    if adj_matrix[sorted_values[i][0], sorted_values[j][0]] == check_value:
                        edge_tuples.append((sorted_values[i][0], sorted_values[j][0]))
                        adj_matrix[sorted_values[i][0], sorted_values[j][0]] = 1.0
                        adj_matrix[sorted_values[j][0], sorted_values[i][0]] = 1.0
            else:
                for j in range(self.graph.num_vertices() - 1, i, -1):
                    if adj_matrix[sorted_values[i][0], sorted_values[j][0]] == check_value:
                        edge_tuples.append((sorted_values[i][0], sorted_values[j][0]))
                        adj_matrix[sorted_values[i][0], sorted_values[j][0]] = 1.0
                        adj_matrix[sorted_values[j][0], sorted_values[i][0]] = 1.0
        self.debug_msg("  --> Covered 100% of vertices.")

        if strategy == "random":
            shuffle(edge_tuples)

        if mode is "current_excl_st":
            self.debug_msg("  --> Removing spanning tree edges...")
            st_edges = random_spanning_tree(self.graph)
            self.graph.set_edge_filter(st_edges)
            num_of_edges -= self.graph.num_edges()
            st_adj_matrix = adjacency(self.graph, weight=None).todense()
            for i in range(0, self.graph.num_vertices()):
                for j in range(i+1, self.graph.num_vertices()):
                    if st_adj_matrix[i, j] == 1.0:
                        if (i, j) in edge_tuples:
                            edge_tuples.remove((i, j))
                        else:
                            edge_tuples.remove((j, i))
            self.debug_msg("  --> Removed " + str(self.graph.num_edges()) + " spanning tree edges from the graph.")
            self.graph.clear_filters()

        if len(edge_tuples) != num_of_edges:
            self.debug_msg("Error: The number of calculated edges did not meet the actual number of " + mode + " edges!")
            sys.exit("Program exit")
        else:
            self.debug_msg("Done. Number of calculated " + mode + " edges: " + str(len(edge_tuples)))
        return edge_tuples

    # Adds new edges to the graph.
    # @params:
    # num_edges = The number of edges to add.
    # applied_strategy = The used strategy. For possible strategies please see strategies dictionary below!
    def add_edge_to_graph(self, num_edges, applied_strategy):
        if num_edges > self.edge_variance_to_percent(100):
            self.debug_msg("The given amount of edges cannot be added because it does not need that many to make the graph complete! Max possible: " + str(int(self.edge_variance_to_percent(100))))
            sys.exit("Program exit")

        # Strategy implementations
        # !!! Make sure that function names match the one in the strategies dictionary below !!!

        def get_random_edges():
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "random")
            return self.edge_list[0:num_edges]

        def get_high_to_high_ev_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "high_to_high", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_high_to_low_ev_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "high_to_low", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_low_to_low_ev_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "low_to_low", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_high_to_high_degree_edges():
            sorted_degrees = get_sorted_vertices("degree", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "high_to_high", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_high_to_low_degree_edges():
            sorted_degrees = get_sorted_vertices("degree", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "high_to_low", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_low_to_low_degree_edges():
            sorted_degrees = get_sorted_vertices("degree", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "low_to_low", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_high_to_high_pr_edges():
            sorted_prvals = get_sorted_vertices("pagerank", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "high_to_high", sorted_prvals)
            return self.edge_list[0:num_edges]

        def get_high_to_low_pr_edges():
            sorted_prvals = get_sorted_vertices("pagerank", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "high_to_low", sorted_prvals)
            return self.edge_list[0:num_edges]

        def get_low_to_low_pr_edges():
            sorted_prvals = get_sorted_vertices("pagerank", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("missing", "low_to_low", sorted_prvals)
            return self.edge_list[0:num_edges]

        # helper functions

        def get_sorted_vertices(property, reverse):
            return sorted(enumerate(self.graph.vertex_properties[property].a), key=lambda x: x[1], reverse=reverse)

        # strategies dictionary

        strategies = {"random": get_random_edges,
                      "high_to_high_ev": get_high_to_high_ev_edges,
                      "high_to_low_ev": get_high_to_low_ev_edges,
                      "low_to_low_ev" : get_low_to_low_ev_edges,
                      "high_to_high_degree": get_high_to_high_degree_edges,
                      "high_to_low_degree": get_high_to_low_degree_edges,
                      "low_to_low_degree": get_low_to_low_degree_edges,
                      "high_to_high_pr": get_high_to_high_pr_edges,
                      "high_to_low_pr": get_high_to_low_pr_edges,
                      "low_to_low_pr": get_low_to_low_pr_edges}

        try:
            edges_tuples = strategies[applied_strategy]()
            #print edges_tuples
        except KeyError:
            self.debug_msg("The given strategy is not available! Please choose one of the following:")
            for key in strategies.iterkeys():
                self.debug_msg(key)
            sys.exit("Program exit")

        self.debug_msg("Adding " + str(len(edges_tuples)) + " edges to the graph...")
        self.graph.add_edge_list(edges_tuples)
        self.debug_msg("Done.")
        self.edge_list[0:num_edges] = []
        if len(self.edge_list) == 0:
            self.edge_list = None

    # Removes edges from the graph.
    # @params:
    # num_edges = The number of edges to remove.
    # applied_strategy = The used strategy. For possible strategies please see strategies dictionary below!
    def remove_edge_from_graph(self, num_edges, applied_strategy):
        if num_edges > self.graph.num_edges():
            self.debug_msg("It is not possible to remove more edges than the graph currently has! Max possible: " + str(self.graph.num_edges()))
            sys.exit("Program exit")

        # Strategy implementations
        # !!! Make sure that function names match the one in the strategies dictionary below !!!

        def get_random_edges():
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "random")
            return self.edge_list[0:num_edges]

        def get_random_st_edges():
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "random")
            return self.edge_list[0:num_edges]

        def get_high_to_high_ev_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "high_to_high", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_high_to_high_ev_st_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "high_to_high", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_high_to_low_ev_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "high_to_low", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_high_to_low_ev_st_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "high_to_low", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_low_to_low_ev_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "low_to_low", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_low_to_low_ev_st_edges():
            sorted_evecvals = get_sorted_vertices("evcentrality", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "low_to_low", sorted_evecvals)
            return self.edge_list[0:num_edges]

        def get_high_to_high_degree_edges():
            sorted_degrees = get_sorted_vertices("degree", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "high_to_high", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_high_to_high_degree_st_edges():
            sorted_degrees = get_sorted_vertices("degree", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "high_to_high", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_high_to_low_degree_edges():
            sorted_degrees = get_sorted_vertices("degree", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "high_to_low", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_high_to_low_degree_st_edges():
            sorted_degrees = get_sorted_vertices("degree", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "high_to_low", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_low_to_low_degree_edges():
            sorted_degrees = get_sorted_vertices("degree", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "low_to_low", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_low_to_low_degree_st_edges():
            sorted_degrees = get_sorted_vertices("degree", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "low_to_low", sorted_degrees)
            return self.edge_list[0:num_edges]

        def get_high_to_high_pr_edges():
            sorted_prvals = get_sorted_vertices("pagerank", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "high_to_high", sorted_prvals)
            return self.edge_list[0:num_edges]

        def get_high_to_high_pr_st_edges():
            sorted_prvals = get_sorted_vertices("pagerank", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "high_to_high", sorted_prvals)
            return self.edge_list[0:num_edges]

        def get_high_to_low_pr_edges():
            sorted_prvals = get_sorted_vertices("pagerank", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "high_to_low", sorted_prvals)
            return self.edge_list[0:num_edges]

        def get_high_to_low_pr_st_edges():
            sorted_prvals = get_sorted_vertices("pagerank", True)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "high_to_low", sorted_prvals)
            return self.edge_list[0:num_edges]

        def get_low_to_low_pr_edges():
            sorted_prvals = get_sorted_vertices("pagerank", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current", "low_to_low", sorted_prvals)
            return self.edge_list[0:num_edges]

        def get_low_to_low_pr_st_edges():
            sorted_prvals = get_sorted_vertices("pagerank", False)
            if self.edge_list is None:
                self.edge_list = self.calculate_edge_list("current_excl_st", "low_to_low", sorted_prvals)
            return self.edge_list[0:num_edges]

        # helper functions

        def get_sorted_vertices(property, reverse):
            return sorted(enumerate(self.graph.vertex_properties[property].a), key=lambda x: x[1], reverse=reverse)

        # strategies dictionary

        strategies = {"random": get_random_edges,
                      "random_st": get_random_st_edges,
                      "high_to_high_ev": get_high_to_high_ev_edges,
                      "high_to_high_ev_st": get_high_to_high_ev_st_edges,
                      "high_to_low_ev": get_high_to_low_ev_edges,
                      "high_to_low_ev_st": get_high_to_low_ev_st_edges,
                      "low_to_low_ev": get_low_to_low_ev_edges,
                      "low_to_low_ev_st": get_low_to_low_ev_st_edges,
                      "high_to_high_degree": get_high_to_high_degree_edges,
                      "high_to_high_degree_st": get_high_to_high_degree_st_edges,
                      "high_to_low_degree": get_high_to_low_degree_edges,
                      "high_to_low_degree_st": get_high_to_low_degree_st_edges,
                      "low_to_low_degree": get_low_to_low_degree_edges,
                      "low_to_low_degree_st": get_low_to_low_degree_st_edges,
                      "high_to_high_pr": get_high_to_high_pr_edges,
                      "high_to_high_pr_st": get_high_to_high_pr_st_edges,
                      "high_to_low_pr": get_high_to_low_pr_edges,
                      "high_to_low_pr_st": get_high_to_low_pr_st_edges,
                      "low_to_low_pr": get_low_to_low_pr_edges,
                      "low_to_low_pr_st": get_low_to_low_pr_st_edges}

        try:
            deleting_edges = strategies[applied_strategy]()
            #print deleting_edges
        except KeyError:
            self.debug_msg("The given strategy is not available! Please choose one of the following:")
            for key in strategies.iterkeys():
                self.debug_msg(key)
            sys.exit("Program exit")

        self.debug_msg("Removing " + str(len(deleting_edges)) + " edges from the graph...")
        for edge in deleting_edges:
            edge_to_remove = self.graph.edge(edge[0], edge[1])
            self.graph.remove_edge(edge_to_remove)
        self.debug_msg("Done.")
        self.edge_list[0:num_edges] = []
        if len(self.edge_list) == 0:
            self.edge_list = None

    # Adds new vertices to the graph.
    # @params:
    # num_vertices = The number of vertices to add.
    # degree = The degree every added vertex will have.
    # applied_strategy = The used strategy. For possible strategies please see strategies dictionary below!
    def add_vertex_to_graph(self, num_vertices, degree, applied_strategy):
        if degree >= self.graph.num_vertices():
            self.debug_msg("The given degree is not possible!")
            sys.exit("Program exit")

        # Strategy implementations
        # !!! Make sure that function names match the one in the strategies dictionary below !!!

        def get_random_vertices():
            return None

        def get_high_ev_vertices():
            return get_vertices("evcentrality", True)

        def get_low_ev_vertices():
            return get_vertices("evcentrality", False)

        def get_high_degree_vertices():
            return get_vertices("degree", True)

        def get_low_degree_vertices():
            return get_vertices("degree", False)

        def get_high_pr_vertices():
            return get_vertices("pagerank", True)

        def get_low_pr_vertices():
            return get_vertices("pagerank", False)

        # Helper functions

        def get_vertices(property, reverse):
            sorted_indices = sorted(enumerate(self.graph.vertex_properties[property].a), key=lambda x: x[1], reverse=reverse)
            matching_vertices = []
            for i in range(0, degree):
                matching_vertices.append(sorted_indices[i][0])
            return matching_vertices

        # strategies dictionary

        strategies = {"random": get_random_vertices,
                      "high_ev": get_high_ev_vertices,
                      "low_ev": get_low_ev_vertices,
                      "high_degree": get_high_degree_vertices,
                      "low_degree": get_low_degree_vertices,
                      "high_pr": get_high_pr_vertices,
                      "low_pr": get_low_pr_vertices}

        try:
            connecting_vertices = strategies[applied_strategy]()
            #print connecting_vertices
        except KeyError:
            self.debug_msg("The given strategy is not available! Please choose one of the following:")
            for key in strategies.iterkeys():
                self.debug_msg(key)
            sys.exit("Program exit")

        self.debug_msg("Adding " + str(num_vertices) + " vertices to the graph...")
        last_id = self.graph.num_vertices() - 1
        i = 1
        while i <= num_vertices:
            new_vertex = self.graph.add_vertex()
            #self.graph.vertex_properties["label"][new_vertex] = last_id + i
            if connecting_vertices is None:
                connecting_vertices = sample(range(0, self.graph.num_vertices()), degree)
                for src_vertex in connecting_vertices:
                    self.graph.add_edge(src_vertex, new_vertex)
                connecting_vertices = None
            else:
                for src_vertex in connecting_vertices:
                    self.graph.add_edge(src_vertex, new_vertex)
            i += 1
        self.debug_msg("Done.")
        # pin_map = self.graph.new_vertex_property("bool")
        # i = 0
        # for v in self.graph.vertices():
        #     if i < self.graph.num_vertices() - num_vertices:
        #         pin_map[v] = True
        #     else:
        #         pin_map[v] = False
        #     i = i + 1
        # newpos = self.graph.new_vertex_property("vector<float>")
        # self.graph.vertex_properties["pos"] = newpos
        # self.graph.vertex_properties["pos"] = sfdp_layout(self.graph, pin=pin_map, pos = self.graph.vertex_properties["pos"])
        # self.graph.vertex_properties["colors"] = community_structure(self.graph, 10000, 2)

    # Removes vertices from the graph.
    # @params:
    # num_vertices = The number of vertices to remove.
    # applied_strategy = The used strategy. For possible strategies please see strategies dictionary below!
    # reduce_to_largest_component = If set true, the graph will be reduced to the largest component.
    def remove_vertex_from_graph(self, num_vertices, applied_strategy, reduce_to_largest_component = False):
        if num_vertices > self.graph.num_vertices():
            self.debug_msg("The given number of vertices is not possible to remove because it is greater than the total number of vertices!")
            sys.exit("Program exit")

        # Strategy implementations
        # !!! Make sure that function names match the one in the strategies dictionary below !!!

        def get_random_vertices():
            return sample(range(0, self.graph.num_vertices()), num_vertices)

        def get_high_ev_vertices():
            return get_vertices("evcentrality", True)

        def get_low_ev_vertices():
            return get_vertices("evcentrality", False)

        def get_high_degree_vertices():
            return get_vertices("degree", True)

        def get_low_degree_vertices():
            return get_vertices("degree", False)

        def get_high_pr_vertices():
            return get_vertices("pagerank", True)

        def get_low_pr_vertices():
            return get_vertices("pagerank", False)

        # Helper functions

        def get_vertices(property, reverse):
            sorted_indices = sorted(enumerate(self.graph.vertex_properties[property].a), key=lambda x: x[1], reverse=reverse)
            matching_vertices = []
            for i in range(0, num_vertices):
                matching_vertices.append(sorted_indices[i][0])
            return matching_vertices

        # strategies dictionary

        strategies = {"random": get_random_vertices,
                      "high_ev": get_high_ev_vertices,
                      "low_ev": get_low_ev_vertices,
                      "high_degree": get_high_degree_vertices,
                      "low_degree": get_low_degree_vertices,
                      "high_pr": get_high_pr_vertices,
                      "low_pr": get_low_pr_vertices}

        try:
            deleting_vertices = strategies[applied_strategy]()
            #print deleting_vertices
        except KeyError:
            self.debug_msg("The given strategy is not available! Please choose one of the following:")
            for key in strategies.iterkeys():
                self.debug_msg(key)
            sys.exit("Program exit")

        deleting_vertices = sorted(deleting_vertices, reverse=True)

        self.debug_msg("Removing " + str(num_vertices) + " vertices from the graph...")
        for vertex in deleting_vertices:
            self.graph.remove_vertex(vertex)
        self.debug_msg("Done.")

        if reduce_to_largest_component is True:
            self.reduce_to_largest_component()


    # plot graph to file...this is deprecated!!!!
    def draw_graph(self, run=0, min_nsize=15, max_nsize=40, file_format="png", output_size=4000, appendix="",
                   label_color="orange"):
        self.debug_msg("Drawing {}".format(file_format))
        # pos = radial_tree_layout(self.graph, self.graph.vertex(0))
        # pos = fruchterman_reingold_layout(self.graph)
        #pos = self.graph.vertex_properties["pos"]

        try:
            graph_draw(self.graph, vertex_fill_color=self.graph.vertex_properties["colors"], edge_color="black",
                       output_size=(output_size, output_size), vertex_text_color=label_color,   #pos = pos,
                       vertex_size=prop_to_size(self.graph.vertex_properties["activity"], mi=min_nsize, ma=max_nsize),
                       vertex_text=self.graph.vertex_properties["label"], vertex_text_position=0,
                       output="graphs/{}_run_{}{}.{}".format(self.graph_name, run, "_" + appendix, file_format))
        except:
            ls = self.graph.new_vertex_property("int")
            for ndx, n in enumerate(self.graph.vertices()):
                ls[n] = str(ndx)
            self.graph.vertex_properties["label"] = ls
            graph_draw(self.graph, vertex_fill_color=self.graph.vertex_properties["colors"], edge_color="black",
                       output_size=(output_size, output_size), vertex_text_color=label_color,   #pos = pos,
                       vertex_size=prop_to_size(self.graph.vertex_properties["activity"], mi=min_nsize, ma=max_nsize),
                       vertex_text=self.graph.vertex_properties["label"], vertex_text_position=0,
                       output="graphs/{}_run_{}{}.{}".format(self.graph_name, run, "_" + appendix, file_format))
