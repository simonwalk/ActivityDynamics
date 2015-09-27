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
plot_only = 0

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
    bg.add_node_weights()
    #bg.collect_colors()
    remove_self_loops(bg.graph)
    remove_parallel_edges(bg.graph)
    #start_date = datetime.date(2014, 2, 1)
    #for i in range(0, 6):
    #    bg.reduce_network_to_epoch(i, start_date)
    #    bg.draw_graph(i, output_size=1000)
    bg.calc_vertex_properties()

    out_file = open(config.graph_source_dir + "empirical_results/" + graph_name + "_deg_dis.txt", "w")
    for v in bg.graph.vertices():
        out_file.write(str(bg.graph.vp["degree"][v]) + "\n")
    out_file.close()

    import subprocess

    r_script_path = os.path.abspath(config.r_dir + 'deg_dis_plots.R')
    wd = r_script_path.replace("R Scripts/deg_dis_plots.R", "") + config.plot_dir + "empirical_results/"
    subprocess.call([config.r_binary_path, r_script_path, wd, graph_name])


    #bg.store_graph(0)


# not the prettiest way to transfer params, but necessary for multiprocessing
def calc_activity(graph_name, store_itas, deltatau, rand_iter=0, tau_in_days=tid):
    nw = Network(False, graph_name, run=rand_iter, deltatau=deltatau, store_iterations=store_itas,
                 tau_in_days=tau_in_days)

    fpath = nw.get_binary_filename(graph_name)
    nw.debug_msg("Loading {}".format(fpath), level=0)
    nw.load_graph_save(fpath)

    nw.prepare_eigenvalues()
    nw.create_folders()
    nw.get_empirical_input(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "/empirical.txt")
    nw.init_empirical_activity()
    nw.calculate_ratios()
    nw.set_ratio(0)
    #nw.open_weights_files()
    #nw.open_taus_files()
    #nw.write_summed_weights_to_file()
    #nw.write_initial_tau_to_file()
    for i in xrange(len(nw.ratios)):
        debug_msg("Starting activity dynamics for ratio: " + str(i+1))
        nw.debug_msg(" --> Sum of weights: {}".format(sum(nw.get_node_weights("activity"))), level=1)
        nw.set_ac(i)
        nw.set_ratio(i)
        nw.reset_tau_iter()
        if i == 0:
            nw.sapm.append(sum(nw.graph.vp["activity"].a) * nw.a_c * nw.graph.num_vertices())
        nw.debug_msg("Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                         "with \x1b[32m ratio={}\x1b[00m and "
                         "\x1b[32mdtau={}\x1b[00m and \x1b[32mdpsi={}\x1b[00m "
                         "for \x1b[32m{} iterations\x1b[00m".format(graph_name, nw.ratio, nw.deltatau, nw.mu,
                                                                    int(nw.deltapsi/nw.deltatau)),
                             level=1)
        for j in xrange(int(nw.mu/nw.deltatau)):
            nw.activity_dynamics(store_weights=False, store_taus=False, empirical=True)
        nw.sapm.append(sum(nw.graph.vp["activity"].a) * nw.a_c * nw.graph.num_vertices())
    #nw.close_weights_files()
    nw.add_graph_properties()
    nw.store_graph(0)


if __name__ == '__main__':
    #graph_name = emp_data_set
    for graph_name in data_sets:
    #if not plot_only:
        create_network(graph_name)
        #calc_activity(graph_name, store_itas, deltatau)
    #empirical_result_plot(graph_name, mode, plot_fmt)
    # for graph_name in data_sets:
    #     empirical_result_plot(graph_name, mode, plot_fmt)