__author__ = 'Simon Walk'

from lib.util import *
from lib.generator import *
from multiprocessing import Pool

global graph_name
global percentage
global perc_counter
global init_name


APPROACHES = ["_RAND", "_SPEC"]
APPROACH = APPROACHES[0]

init_name = "Karate"
graph_name = init_name
percentage = np.arange(0.0, 0.11, 0.1)
perc_counter = 0
graph_source_name = graph_name + "_{}PERC".format(int(round(percentage[0]*100))) + APPROACH

num_iterations = 10000
num_random_inits = 10
num_nodes = 34
ratios = [10]
deltataus = [0.001]
store_itas = 10

def create_network():
    global graph_name
    global graph_source_name
    graph_name += "_{}PERC".format(int(round(percentage[perc_counter]*100))) + APPROACH
    graph_source_name = graph_name
    nw = Network(num_iterations, False, graph_name, run=0, plot_with_labels=True, store_iterations=store_itas)
    bg = Generator(graph_name, num_nodes)
    bg.debug_msg("Creating Karate network!")
    bg.create_karate_graph()
    bg.add_edge_weights(0.0, 0.1)
    bg.add_node_weights(0.0, 0.1)
    bg.calc_eigenvalues(max(int(bg.graph.num_vertices()/2)-1, 1))
    bg.collect_colors()
    bg.calc_vertex_properties()
    bg.draw_graph(0)
    bg.store_gml(0, save_specific=True)
    nw.load_graph(config.graph_binary_dir+"GML/"+nw.graph_name+"/"+nw.graph_name+"_run_"+str(0)+".gml")
    nw.prepare_eigenvalues()
    nw.create_folders()
    nw.plot_eigenvalues()
    nw.plot_gx(-20, 20)
    nw.plot_fx(-1, 2)

def calc_activity(rand_iter):
    for ridx, ratio in enumerate(ratios):
        if ridx == len(ratios)-1:
            next_ratio = 1
        else:
            next_ratio = ratios[ridx+1]
        for deltatau in deltataus:
            nw = Network(num_iterations, False, graph_name, run=rand_iter, percentage=percentage[perc_counter],
                         plot_with_labels=True, ratio=ratio, deltatau=deltatau, store_iterations=store_itas)
            nw.debug_msg("\x1b[34mProcessing random init %d of %d for %s\x1b[00m" % (nw.run+1, num_random_inits,
                                                                                        graph_name), level=1)
            fpath = config.graph_binary_dir+"GML/"+graph_source_name+"/"+graph_source_name+"_run_"+str(0)+".gml"
            nw.debug_msg("Loading {}".format(fpath), level=0)
            nw.load_graph(fpath)
            nw.prepare_eigenvalues()
            nw.create_folders()
            nw.add_edge_weights(0.0, 0.1)
            nw.add_node_weights(0.0, 0.1)
            temp_weights = nw.get_node_weights("activity")
            nw.debug_msg("Running Dynamic Simulation for '\x1b[32m{}\x1b[00m' "
                         "with \x1b[32mratio={}\x1b[00m and "
                         "\x1b[32mdtau={}\x1b[00m".format(graph_name, ratio, deltatau), level=0)
            # percentage of nodes to add/remove
            # percentage of edges to add/remove
            if "RAND" in APPROACH:
                nw.get_percentage_of_nodes(percentage=percentage[perc_counter],
                                           mult_factor=float(next_ratio)/float(ratio))
            else:
                nw.get_percentage_of_nodes(spec_func=nw.calc_ev_centrality, percentage=percentage[perc_counter],
                                           mult_factor=float(next_ratio)/float(ratio), specific=True,
                                           selector="evc")

            nw.reset_attributes(ratio, deltatau, temp_weights)
            nw.open_weights_file()
            nw.open_ab_file()
            i = 0
            while(True):
                nw.activity_dynamics_matrix(store_weights=True, weights_debug=True)
                if i % num_iterations == 0:
                    nw.debug_msg("   >> processing iteration: {} of {}".format(i, nw.num_iterations), level=0)
                if nw.converged or nw.diverged:
                    break
                i += 1
            nw.close_weights_file()
            nw.close_ab_file()

            if rand_iter == 0:
                f = open(config.graph_binary_dir + "summary/" + nw.graph_name + "_temp_plot_values.txt", "wb")
                f.write(("\t").join([str(rat) for rat in ratios]) + "\n")
                f.write(("\t").join([str(dt) for dt in deltataus]) + "\n")
                f.write(str(nw.num_iterations)+"\n")
                f.write(str(nw.graph_name)+"\n")
                f.write(("\t").join([str(x) for x in sorted(nw.ew, reverse=True)])+"\n")
                f.write(str(nw.store_iterations)+"\n")
                f.close()

if __name__ == '__main__':
    #if "RAND" in APPROACH:
    #    create_network()
    for p in percentage:
        graph_name = graph_source_name.replace("_{}PERC".format(int(round((percentage[0]*100)))),
                                               "_{}PERC".format(int(round((percentage[perc_counter]*100)))))

        pool = Pool(processes=10)
        pool.map(calc_activity, xrange(0, num_random_inits))
        pool.close()
        pool.join()
        transpose_files(graph_name, num_random_inits)
        avg_over_files(graph_name, num_random_inits, suffix="")
        mean_per_iteration(graph_name, num_random_inits, postfix="")
        avg_over_files(graph_name, num_random_inits, suffix="_T")
        plot_weights_over_time(graph_name)
        #store_sem_errors_over_b(graph_name, num_random_inits)
        # create_scatterplots(graph_name, graph_source_name)
        avrg_activity_over_tau(graph_name)
        perc_counter += 1