__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

from lib.util import *
from lib.generator import *

APPROACHES = ["_RAND", "_SPEC"]
APPROACH = APPROACHES[0]

init_names = ["BEACHAPEDIA", "APBR", "CHARACTERDB", "SMWORG", "W15M", #0 - 4
              "AARDNOOT", "AUTOCOLLECTIVE", "CWW", "NOBBZ", "KARATE", #5 - 9
              "StackOverflow", "EnglishStackExchange","HistoryStackExchange", "MathStackExchange"] # 10 - 13

init_name = init_names[10]
deltatau = 0.001
deltataus = [deltatau]
store_itas = 10
tau_in_days = 30.0
graph_name = init_name
graph_source_name = graph_name + "_0PERC" + APPROACH

ratios = []

def create_network():
    global graph_name
    global graph_source_name
    graph_name += "_{}PERC".format(0) + APPROACH
    graph_source_name = graph_name
    nw = Network(False, graph_name, run=0)
    bg = Generator(graph_name)
    bg.debug_msg("Loading SMW network!")
    bg.load_graph(graph_name+"_run_"+str(0))
    #bg.reduce_to_largest_component()
    bg.clear_all_filters()
    #bg.calc_eigenvalues(max(int(bg.graph.num_vertices()/2)-1, 1))
    bg.calc_eigenvalues(4)
    bg.add_node_weights()
    bg.collect_colors()
    bg.draw_graph(0)
    bg.calc_vertex_properties()
    bg.store_graph(0, save_specific=True)
    nw.load_graph(config.graph_binary_dir+"GT/"+nw.graph_name+"/"+nw.graph_name+"_run_"+str(0)+".gt")
    nw.create_folders()
    nw.prepare_eigenvalues()
    nw.plot_eigenvalues()
    nw.plot_gx(-6, 6)
    nw.plot_fx(-1, 1)
    nw.plot_fx_weight(-1,1)


def calc_activity(rand_iter):

    nw = Network(False, graph_name, run=rand_iter, deltatau=deltatau, store_iterations=store_itas, tau_in_days=tau_in_days)
    fpath = config.graph_binary_dir+"GT/"+graph_source_name+"/"+graph_source_name+"_run_"+str(rand_iter)+".gt"
    nw.debug_msg("Loading {}".format(fpath), level=0)
    nw.load_graph_save(fpath)
    nw.clear_all_filters()
    nw.plot_fx(-1, 1)
    nw.plot_gx(-6, 6)
    nw.prepare_eigenvalues()
    nw.create_folders()
    nw.get_empirical_input(config.graph_binary_dir + "empirical_input/" + init_name + "_empirical_input.txt")
    nw.init_empirical_activity()
    nw.calculate_ratios()
    nw.set_ratio(0)
    nw.open_weights_files()
    nw.write_summed_weights_to_file()
    for i in xrange(len(nw.ratios)):
        nw.debug_msg(" --> Sum of weights: {}".format(sum(nw.get_node_weights("activity"))), level=1)
        nw.set_ac(i)
        nw.set_ratio(i)
        nw.debug_msg("Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                         "with \x1b[32m ratio={}\x1b[00m and "
                         "\x1b[32mdtau={}\x1b[00m and \x1b[32mdpsi={}\x1b[00m "
                         "for \x1b[32m{} iterations\x1b[00m".format(graph_name, nw.ratio, nw.deltatau, nw.deltapsi,
                                                                    int(nw.deltapsi/nw.deltatau)),
                             level=1)
        for j in xrange(int(nw.deltapsi/nw.deltatau)):
            nw.activity_dynamics(store_weights=True, empirical=True)
    nw.close_weights_files()
    nw.add_graph_properties()
    nw.store_graph(0, save_specific=True)


if __name__ == '__main__':
    #if "RAND" in APPROACH:
    #    create_network()
    #calc_activity(0)
    graph_name += "_{}PERC".format(0) + APPROACH
    graph_source_name = graph_name
    empirical_result_plot(graph_name)
    #avrg_activity_over_tau_empirical(graph_name)
    #plot_empirical_ratios(graph_name)
    #plot_empirical_error(graph_name)