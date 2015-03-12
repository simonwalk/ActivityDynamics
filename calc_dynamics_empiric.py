__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

from lib.util import *
from lib.generator import *
from multiprocessing import Pool


def create_network(graph_name):
    bg = Generator(graph_name)
    bg.debug_msg("Loading SMW network!")
    bg.load_graph(graph_name+"_run_"+str(0))
    bg.clear_all_filters()
    bg.calc_eigenvalues(2)
    bg.add_node_weights()
    bg.collect_colors()
    remove_self_loops(bg.graph)
    remove_parallel_edges(bg.graph)
    bg.draw_graph(0)
    bg.calc_vertex_properties()
    bg.store_graph(0)


# not the prettiest way to transfer params, but necessary for multiprocessing
def calc_activity(graph_name, store_itas, deltatau, rand_iter=0, tau_in_days=30):
    nw = Network(False, graph_name, run=rand_iter, deltatau=deltatau, store_iterations=store_itas,
                 tau_in_days=tau_in_days)

    fpath = nw.get_binary_filename(graph_name)
    nw.debug_msg("Loading {}".format(fpath), level=0)
    nw.load_graph_save(fpath)

    nw.prepare_eigenvalues()
    nw.create_folders()
    nw.get_empirical_input(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "_empirical.txt")
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
    nw.store_graph(0)


if __name__ == '__main__':

    empirical_ds = ["BeerStackExchange",    #0
                    "EnglishStackExchange", #1
                    "MathStackExchange",    #2
                    "StackOverflow",        #3
                    "HistoryStackExchange", #4
                    "CHARACTERDB",          #5
                    "BEACHAPEDIA",          #6
                    "NOBBZ",                #7
                    "W15M"]                 #8
    graph_name = empirical_ds[0]
    create_network(graph_name)
    deltatau = 0.001
    store_itas = 1
    calc_activity(graph_name, store_itas, deltatau)
    empirical_result_plot(graph_name)