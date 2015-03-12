from lib.util import *
from lib.generator import *
import time
from multiprocessing import Pool

emp_data_set = "BeerStackExchange"

start_date = datetime.date(2014, 1, 21)
network_epochs = 8

deltatau = 0.001
store_itas = 1

def create_network():
    bg = Generator(emp_data_set)
    bg.load_graph(emp_data_set+"_run_"+str(0))
    bg.clear_all_filters()
    bg.calc_eigenvalues(2)
    bg.add_node_weights()
    bg.collect_colors()
    remove_self_loops(bg.graph)
    remove_parallel_edges(bg.graph)
    bg.draw_graph(0)
    bg.calc_vertex_properties()
    bg.store_graph(0)


def create_network_epochs():
    nw = Network(False, emp_data_set)
    fpath = nw.get_binary_filename(emp_data_set)
    nw.load_graph(fpath)

    debug_msg("Loaded network: " + str(nw.graph.num_vertices()) + " vertices, " + str(nw.graph.num_edges()) + " edges")

    for epoch in range(1, network_epochs + 1):
        nw.reduce_network_to_epoch(start_date, epoch)
        nw.graph_name = emp_data_set + "_epoch_" + str(epoch)
        #nw.draw_graph(0)
        nw.plot_epoch(epoch)



def calc_activity():
    nw = Network(False, emp_data_set, run=0, deltatau=deltatau, store_iterations=store_itas, tau_in_days=25)
    fpath = nw.get_binary_filename(emp_data_set)
    nw.debug_msg("Loading " + fpath)
    nw.load_graph(fpath)
    nw.debug_msg("Loaded network: " + str(nw.graph.num_vertices()) + " vertices, " + str(nw.graph.num_edges()) + " edges")

    for epoch in range(1, network_epochs + 1):
        nw.reduce_network_to_epoch(start_date, epoch)
        nw.update_num_vertices_edges()
        nw.calc_eigenvalues_for_epoch(2)

    nw.get_empirical_input(config.graph_binary_dir + "empirical_data/" + nw.graph_name + "_empirical.txt", epoch_mode=True)
    nw.init_empirical_activity(epoch_mode=True)
    nw.calc_ratios_for_epochs()
    nw.open_weights_files()
    nw.write_summed_weights_to_file()
    for i in range(0, network_epochs):
        debug_msg("Starting activity dynamics for epoch: " + str(i+1))
        nw.reduce_network_to_epoch(start_date, i+1)
        nw.update_ones_ratio()
        nw.update_adjacency()
        nw.debug_msg(" --> Sum of weights: {}".format(sum(nw.get_node_weights("activity"))), level=1)
        nw.set_ac(i)
        nw.set_ratio(i)
        nw.set_deltapsi(i)
        nw.debug_msg(" --> Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                         "with \x1b[32m ratio={}\x1b[00m and "
                         "\x1b[32mdtau={}\x1b[00m and \x1b[32mdpsi={}\x1b[00m "
                         "for \x1b[32m{} iterations\x1b[00m".format(emp_data_set, nw.ratio, nw.deltatau, nw.deltapsi,
                                                                    int(nw.deltapsi/nw.deltatau)), level=1)
        for j in xrange(int(nw.deltapsi/nw.deltatau)):
            nw.activity_dynamics(store_weights=True, empirical=True)

    nw.close_weights_files()
    nw.add_graph_properties(epoch_mode=True)
    nw.store_graph(0)

if __name__ == '__main__':
    create_network()
    #create_network_epochs()
    calc_activity()
    #empirical_result_plot(emp_data_set)
    sys.exit()
    # for v in nw.graph.vertices():
    #     print "NODE: " + str(nw.graph.vertex_properties["nodeID"][v])
    #     print "FIRST ACTIVITY: " + str(nw.graph.vertex_properties["firstActivity"][v])
    #     print "***"