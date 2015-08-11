__author__ = 'Simon Walk, Florian Geigl, Denis Helic'
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "simon.walk@tugraz.at"
__status__ = "Development"

import matplotlib
matplotlib.use('Agg')
from lib.util import *
from lib.generator import *
from multiprocessing import Pool

deltatau = 0.001
store_itas = 10
tid = 30
mode = "months"
plot_fmt = "pdf"
plot_only = 1

data_sets = ["BeerStackExchange",           # 0
             "BitcoinStackExchange",        # 1
             "ElectronicsStackExchange",    # 2
             "PhysicsStackExchange",        # 3
             "GamingStackExchange",         # 4
             "ComplexOperations",           # 5
             "BioInformatics",              # 6
             "Neurolex",                    # 7
             "DotaWiki",                    # 8
             "PracticalPlants"]             # 9

emp_data_set = data_sets[0]

def create_network(graph_name):
    bg = Generator(graph_name)
    bg.debug_msg("Loading SMW network!")
    bg.load_graph(graph_name+"_run_"+str(0))
    bg.clear_all_filters()
    bg.calc_eigenvalues(2)
    bg.add_node_weights(0, 0)
    #bg.collect_colors()
    remove_self_loops(bg.graph)
    remove_parallel_edges(bg.graph)
    #bg.draw_graph(0)
    bg.calc_vertex_properties()
    bg.store_graph(0)


# not the prettiest way to transfer params, but necessary for multiprocessing
def calc_activity(graph_name, store_itas, deltatau, rand_iter=0, tau_in_days=tid):
    nw = Network(False, graph_name, run=rand_iter, deltatau=deltatau, store_iterations=store_itas,
                 tau_in_days=tau_in_days)

    fpath = nw.get_binary_filename(graph_name)
    nw.debug_msg("Loading {}".format(fpath), level=0)
    nw.load_graph_save(fpath)
    nw.get_epochs_info()
    nw.prepare_eigenvalues()
    nw.create_folders()
    nw.get_empirical_input(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "/empirical.txt")
    #nw.open_weights_files()
    for i in range(0, len(nw.dx) + 1):
        debug_msg("Starting activity dynamics for epoch: " + str(i))
        nw.set_cur_epoch(i)
        nw.reduce_network_to_epoch()
        nw.update_vertex_properties()
        nw.update_model_parameters()
        nw.save_props_per_iter()
        nw.update_activity()
        nw.init_empirical_activity()
        nw.store_to_csapm()
        if i == len(nw.dx):
            break
        if i == 0:
            nw.store_to_sapm()
        nw.debug_msg(" --> Current activity: {}".format(sum(nw.get_node_weights("activity")) * nw.a_c * nw.graph.num_vertices()), level=1)
        nw.calculate_ratios()
        nw.set_ratio(-1)
        nw.reset_tau_iter()
        nw.debug_msg("Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                     "with \x1b[32m ratio={}\x1b[00m and "
                     "\x1b[32mdtau={}\x1b[00m and \x1b[32mdpsi={}\x1b[00m "
                     "for \x1b[32m{} iterations\x1b[00m".format(graph_name, nw.ratio, nw.deltatau, nw.mu,
                                                                int(nw.mu/nw.deltatau)),
                     level=1)
        for j in xrange(int(nw.mu/nw.deltatau)):
            nw.activity_dynamics(store_weights=False, store_taus=False, empirical=True)
        nw.debug_msg(" --> Current activity: {}".format(sum(nw.get_node_weights("activity")) * nw.a_c * nw.graph.num_vertices()), level=1)
        nw.store_to_sapm()

    print nw.sapm
    print nw.csapm

    #nw.close_weights_files()
    #nw.close_taus_files()
    nw.add_graph_properties()
    nw.store_graph(0)


if __name__ == '__main__':
    graph_name = emp_data_set
    # if not plot_only:
    #     create_network(graph_name)
    #     calc_activity(graph_name, store_itas, deltatau)
    # empirical_result_plot(graph_name, mode, plot_fmt)
    for graph_name in data_sets:
        empirical_result_plot(graph_name, mode, plot_fmt)