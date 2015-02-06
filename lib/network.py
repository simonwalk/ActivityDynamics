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
        self.num_users = [] # self.graph.num_vertices()
        self.init_users = []
        self.posts_per_user_per_day = []
        self.a_cs = []

        f = open(path, "rb")
        for ldx, line in enumerate(f):
            if ldx < 1:
                continue
            el = line.strip().split("\t")
            print "el[1]: ", el[1]
            self.dx.append(float(el[1]))
            self.apm.append(float(el[2]))
            self.posts.append(float(el[3]))
            self.replies.append(float(el[4]))
            try:
                self.init_users.append(el[6].split(","))
            except:
                self.init_users.append(["dummy"])
            self.num_users.append(float(el[5]))
            self.posts_per_user_per_day.append(float(el[3])/float(el[5])/30.0)
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
        print self.apm[0]
        print self.num_users[0]
        print self.a_c
        initial_empirical_activity = self.apm[0]/(self.graph.num_edges()*2)/self.num_users[0]/self.a_c
        print "Init Activity: ", initial_empirical_activity

        init_nodes = self.init_users[0]
        # reset activity!
        for v in self.graph.vertices():
            self.graph.vp["activity"][v] = 0.0

        # randomly initiate minimal activity
        for v_id in init_nodes:
            n = self.graph.vertex(v_id)
            self.graph.vp["activity"][n] = initial_empirical_activity
        

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
            #self.ratio -= 0.01 * activity_current / (self.a_c * self.num_vertices)
            self.ratios.append(self.ratio)
        self.debug_msg("ratios ({}): {}".format(len(self.ratios), self.ratios), level=1)


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
        remove_self_loops(self.graph)
        remove_parallel_edges(self.graph)
        self.debug_msg("  --> Creating ones vector", level=0)
        self.ones_ratio = [1.0] * self.graph.num_vertices()
        #self.debug_msg("  --> Getting Laplacian Matrix", level=0)
        #self.L = laplacian(self.graph, weight=None)
        self.debug_msg("  --> Getting Adjacency Matrix", level=0)
        self.A = adjacency(self.graph, weight=None)
        self.num_vertices = self.graph.num_vertices()
        self.num_edges = self.graph.num_edges()
        self.debug_msg("  --> Counted {} Vertices".format(self.num_vertices), level=0)
        self.debug_msg("  --> Counted {} Edges".format(self.num_edges), level=0)
