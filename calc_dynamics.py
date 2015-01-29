__author__ = 'Simon Walk'

from lib.util import *
from lib.generator import *
from multiprocessing import Pool


def calc_activity(rand_iter, graph_name, deltatau, deltapsi, store_itas, ratios, num_random_inits):
    nw = Network(False, graph_name, run=rand_iter, deltatau=deltatau, deltapsi = deltapsi,
                 store_iterations=store_itas, ratios=ratios, ratio_index=0, runs=num_random_inits)
    for ridx, ratio in enumerate(ratios):
        nw.debug_msg("\x1b[34mProcessing random init %d of %d for %s\x1b[00m" % (nw.run+1, num_random_inits,
                                                                                 graph_name), level=1)
        fpath = nw.get_binary_filename(graph_name)
        nw.debug_msg("Loading {}".format(fpath), level=0)
        nw.load_graph_save(fpath)
        nw.prepare_eigenvalues()
        nw.create_folders()
        nw.add_node_weights(0.0, 0.1)
        nw.debug_msg("Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                     "with \x1b[32mratio={}\x1b[00m and "
                     "\x1b[32mdtau={}\x1b[00m".format(graph_name, ratio, deltatau), level=0)
        temp_weights = nw.get_node_weights("activity")
        nw.reset_attributes(ratio, temp_weights)
        nw.open_weights_files()
        while(True):
            nw.activity_dynamics(store_weights=True)
            if nw.converged or nw.diverged:
                break
        nw.close_weights_files()
        if rand_iter == 0:
            nw.add_graph_properties()
            nw.store_graph(rand_iter, save_specific=True)


if __name__ == '__main__':
    graph_name = "Karate"
    num_random_inits = 10
    ratios = [x for x in xrange(0, 121, 10) if x > 0]
    ratios.reverse()
    ratios.append(1)
    ratios = [1, 10, 30, 80, 100, 120, 160, 200, 260, 320, 500]

    deltatau = 0.001
    deltapsi = 1
    store_itas = 10

    generator = Generator(graph_name)
    generator.create_network(generator.create_karate_graph, draw_graph=True, draw_ev=True)

    #pool = Pool(processes=10)
    #pool.map(calc_activity, xrange(0, [graph_name, deltatau, deltapsi, store_itas, ratios, num_random_inits]))
    #pool.close()
    #pool.join()
    #plot_weights_over_time(graph_name)
    #avrg_activity_over_tau(graph_name)