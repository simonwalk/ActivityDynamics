# def store_sem_errors_over_b(graph_name, num_random_inits=100):
#     debug_msg("  >>> Calculating SEM errors for final iterations")
#     a_vals, b_vals, num_iterations, gname = get_network_details(graph_name)
#
#     for aval in a_vals:
#         for bval in b_vals:
#
#             debug_msg("    >> " + config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + "_" +
#                       str(num_iterations) + "_iterations_" + str(aval).replace(".", "") + "_" + str(bval).replace(".", "") +
#                       "_run_0_100.txt", level=1)
#
#             node_errors_file = open(config.graph_source_dir + "errors/" + graph_name + "_" + str(aval).replace(".", "") +
#                                     "_" + str(bval).replace(".", "") + "_node_errors.txt", "wb")
#
#             sem_errors = defaultdict(float)
#             node_errors_dict = defaultdict(list)
#
#             for run in xrange(0, num_random_inits, 1):
#                 filepath = config.graph_source_dir + "weights/" + graph_name + "/" + graph_name + "_" + \
#                            str(num_iterations) + "_iterations_" + str(aval).replace(".", "") + "_" + \
#                            str(bval).replace(".", "") + "_run_" + str(run) + "_T.txt"
#                 f = open(filepath, "rb")
#                 for lidx, line in enumerate(f):
#                     sl = line.strip().split("\t")
#                     node_errors_dict[lidx].append(float(sl[-1]))
#
#             for k, v in node_errors_dict.iteritems():
#                 sem_errors[k] = sem(v)
#
#             node_errors_file.write(("\t").join([str(x) for x in sem_errors.values()]))
#             node_errors_file.write("\n")
#             node_errors_file.close()


# def plot_weights_over_b(graph_name):
#     debug_msg("\x1b[34mPlotting average weights over different b!\x1b[00m")
#     a_vals, b_vals, num_iterations, gname = get_network_details(graph_name)
#     node_weights = defaultdict(list)
#
#     storage_folder = config.plot_dir + "average_over_time/"
#     plt.figure()
#     for a in a_vals:
#         for b in b_vals:
#             fpath = get_avg_weight_fn(a, b, graph_name, num_iterations)
#             debug_msg("  ++ Processing a = {}, b = {}".format(a, b), level=1)
#             f = open(fpath, "rb")
#             for ldx, l in enumerate(f):
#                 node_weights[ldx].append(np.mean(np.array(np.fromstring(l, sep='\t'))))
#             f.close()
#     for k, v in node_weights.iteritems():
#         plt.plot(b_vals, v)
#     labels = ["%.4f" % x for x in b_vals]
#     plt.xticks(b_vals, labels, rotation=30)
#     plt.xlim(min(b_vals), max(b_vals))
#     plt.subplots_adjust(bottom=0.15)
#     plt.grid(color="gray")
#     plt.xlabel("Value of b")
#     plt.ylabel("Average Node Activity")
#     plt.title("Average node activities at increasing b and a=%f" % a_vals[0])
#     plt.savefig(storage_folder + "{}.png".format(graph_name))
#     plt.close("all")


# def plot_weights_over_a(graph_name, xlab="Activity Decay Rate (a)"):
#     debug_msg("\x1b[34mPlotting average weights over different a!\x1b[00m")
#     a_vals, b_vals, num_iterations, gname = get_network_details(graph_name)
#     node_weights = defaultdict(list)
#     storage_folder = config.plot_dir + "average_over_time/"
#
#     plt.figure()
#     for b in b_vals:
#         for a in a_vals:
#             fpath = get_avg_weight_fn(a,b,graph_name,num_iterations)
#             debug_msg("  ++ Processing a = {}, b = {}".format(a, b), level=1)
#             f = open(fpath, "rb")
#             for ldx, l in enumerate(f):
#                 node_weights[ldx].append(np.mean(np.array(np.fromstring(l, sep='\t'))))
#             f.close()
#     for k, v in node_weights.iteritems():
#         plt.plot(a_vals, v)
#     labels = ["%.4f" % x for x in a_vals]
#     plt.xticks(a_vals, labels, rotation=30)
#     plt.xlim(min(a_vals), max(a_vals))
#     plt.subplots_adjust(bottom=0.15)
#     plt.grid(color="gray")
#     plt.xlabel(xlab)
#     plt.ylabel("Average Node Activity")
#     plt.title("Average node activities at decreasing\n%s and b=%.2f" % (xlab, b_vals[0]))
#     ax = plt.gca()
#     ax.invert_xaxis()
#     plt.savefig(storage_folder + "{}.png".format(graph_name))
#     plt.close("all")

def plot_active_vs_inactive(graph_name, iterations_stored=1):
    debug_msg("  >>> Drawing Active vs. Inactive plot")
    avals, bvals, num_iterations, gname = get_network_details(graph_name)
    storage_folder = config.plot_dir + "active_inactive/" + "/" + graph_name + "/"
    ticks = [x for x in xrange(1, int(num_iterations / iterations_stored) + 1, 5)]
    labels = [str(x) for x in xrange(0, num_iterations + 1, 500)]
    for a in avals:
        for b in bvals:
            debug_msg("  ++ Processing a = {}, b = {}".format(a, b), level=1)
            active_nodes = defaultdict(int)
            lurker_nodes = defaultdict(int)
            inactive_nodes = defaultdict(int)
            number_of_nodes = 0
            iterations = 0
            fpath = get_avg_weight_fn(a, b, graph_name, num_iterations)
            pf = open(fpath, "rb")

            plt.figure()
            plt.xlim(0, num_iterations / iterations_stored)
            ax = plt.axes()
            ax.grid(color="gray")
            label_list = ['Active Users (>0.01)', 'Lurker (>0; <0.01)', 'Inactive Users (=0)']
            plt.xticks(ticks, labels, rotation=30)
            for lidx, l in enumerate(pf.readlines()):
                number_of_nodes += 1
                sl = l.strip().split('\t')
                lsl = len(sl)
                if iterations < lsl:
                    iterations = lsl

                for eidx, el in enumerate(sl):
                    el_f = float(el)
                    if el_f < 0.0001:
                        inactive_nodes[eidx] += 1
                        lurker_nodes[eidx] += 0
                        active_nodes[eidx] += 0
                    elif el_f >= 0.0001 and el_f < 0.1:
                        lurker_nodes[eidx] += 1
                        inactive_nodes[eidx] += 0
                        active_nodes[eidx] += 0
                    else:
                        active_nodes[eidx] += 1
                        inactive_nodes[eidx] += 0
                        lurker_nodes[eidx] += 0

            stack_coll = plt.stackplot(range(0, iterations), active_nodes.values(), lurker_nodes.values(),
                                       inactive_nodes.values(), alpha=0.5)
            t = plt.title("Distribution of Activity per Iteration\n(a={},b={})".format(a, b))
            t.set_y(1.07)
            plt.subplots_adjust(top=0.86)
            plt.xlabel("Iteration")
            plt.ylabel("Number of Users")
            lx = plt.subplot(111)
            lx.grid(color="gray")
            plt.ylim(0.0, number_of_nodes)

            proxy_rects = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack_coll]
            box = lx.get_position()
            lx.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            lx.legend(proxy_rects, label_list, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3,
                      prop={'size': 8})
            plt.savefig(storage_folder  + graph_name + "_" + str(num_iterations) +
                        "_iterations_" + str(a).replace(".", "") + "_" +
                        str(b).replace(".", "") + "_active_inactive.pdf")
            plt.close("all")
            pf.close()

def plot_average_percentage(graph_names, percentage=range(0, 101, 1),
                            approaches=["0PERC_RAND", "0PERC_SPEC"], legend_names=None):
    debug_msg("\x1b[34mPlotting average weight over percentage!\x1b[00m")
    storage_folder = config.plot_dir + "average_weights_over_percentage/"
    from itertools import cycle
    lines = ["-","--","-.",":"]
    plt.figure()
    lines = []
    linestyles = ["^", "+", "s", "o", "v", "d", "*"]

    r = open(storage_folder+"source_weights_for_r.txt", "wb")
    r.write("DSNAME\t"+ ("\t").join([str(x) for x in xrange(0,101)])+"\n")

    for idx, graph_name in enumerate(graph_names):

        for jdx, approach in enumerate(approaches):
            graph_source_name = graph_name + "_" + approach
            p = "/Volumes/MyBook4TB/Experiments Paper 1/" + graph_source_name.replace("0PERC", "PERC") + "/"
            a_vals, b_vals, num_iterations, gname = get_network_details(graph_source_name)
            b = b_vals[0]
            a = a_vals[0]
            node_weights = defaultdict(float)
            if "RAND" in approach:
                type_appr = "Random"
            else:
                type_appr = "Informed"

            for perc in percentage:
                try:
                    gn = graph_source_name.replace("0PERC", "{}PERC".format(int(perc)))

                    #fpath = get_avg_weight_fn(a,b,gn,num_iterations)
                    fpath = get_avg_weight_fn(a,b,gn,num_iterations)
                    debug_msg("  ++ Processing a = {}, b = {}, perc = {}".format(a, b, perc))
                    p = int(perc)
                    f = open(fpath, "rb")
                    lcounter = 0.0
                    for ldx, l in enumerate(f):
                        node_weights[p] = node_weights[p] + np.array(np.fromstring(l, sep='\t'))[-1]
                        lcounter += 1.0
                    f.close()
                    node_weights[p] = node_weights[p]/lcounter
                    debug_msg("  ----> perc = {}, avrg activity = {}".format(p, node_weights[p]))
                except Exception as e:
                    debug_msg("\x1b[31mERROR: {}\x1b[00m".format(str(e)))
            lines.append(plt.plot(node_weights.keys(), node_weights.values(), label=legend_names[idx] + " " + type_appr))
            r.write(legend_names[idx] + " " + type_appr+"\t")
            r.write(("\t").join([str(x) for x in node_weights.values()])+"\n")
            print len([str(x) for x in node_weights.values()])
            print node_weights.keys()
    ax = plt.subplot(111)
    ax.set_yscale('log')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})
    plt.grid(color="gray")
    plt.ylabel("Average Activity per Node")
    plt.xlabel("Percentage of Nodes")
    plt.title("Increasing activity for percentage of nodes\n from "+ r"$a_0$" " to " + r"$a_1$")
    plt.savefig(storage_folder + "{}.png".format("average_activity_per_percentage_increase"))
    plt.close("all")
    r.close()


def compare_percentage(init_name, graph_name, percentages, multiplicators, replace):
    a, b, iterations, gname = get_network_details(graph_name)

    for a_val in a:
        for b_val in b:
            plt.figure()
            ax = plt.axes()
            ax.grid(color="gray")
            plt.title("Total Average Activity\n(a={},b={})".format(a_val, b_val))
            plt.xlabel("Iteration")
            plt.ylabel("Total Average Activity")
            ax = plt.subplot(111)
            it_counter = 0.0
            for p in percentages:
                debug_msg("  ** Plotting b={}, p={}".format(b_val, p))
                for m in multiplicators:
                    gn = graph_name.replace(replace, "_{}PERC_SPEC_MULT{}".format(int(p * 100), m))
                    debug_msg("    >> opening: {}".format(get_avg_weight_fn(a_val, b_val, gn, iterations)))
                    f = open(get_avg_weight_fn(a_val, b_val, gn, iterations), "rb")
                    average_weights_per_iteration = defaultdict(float)
                    c_counter = 0.0
                    for lidx, l in enumerate(f):
                        sl = l.strip().split("\t")
                        c_counter = len(sl)
                        if it_counter < c_counter:
                            it_counter = float(c_counter)
                        for iidx, nweight in enumerate(sl):
                            average_weights_per_iteration[iidx] += float(nweight)
                    t = [x / c_counter for x in average_weights_per_iteration.values()]
                    ax.plot(xrange(int(c_counter)), t, label='p=%d' % int(p * 100))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.xlim(1, it_counter - 1)
            ticks = [x for x in xrange(1, int(it_counter) + 1, 5)]
            labels = [str(x) for x in xrange(0, iterations + 1, 500)]
            labels[0] = str(1)
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 10})
            plt.xticks(ticks, labels, rotation=30)
            plt.savefig(config.plot_dir + "percentage_comp/" + graph_name + "/" + graph_name + "_" + str(iterations) + "_iterations_" +
                        str(a_val).replace(".", "") + "_" + str(b_val).replace(".", "") + "_perc_comp.png")
            plt.close("all")

def print_matrix(matrix):
    for i in matrix:
        for j in i:
            if j < 0.0001 and not j == 0:
                print '{:.2e}'.format(j)[:10].center(10),
            else:
                print str(j)[:10].center(10),
        print ''

# def transpose_txt(filename, del_orig=False, sep='\t'):
#     data = np.loadtxt(filename, dtype=str)
#     if len(filename) > 4 and filename[-4] == '.':
#         filename_s = filename.rsplit('.', 1)
#     else:
#         filename_s = [filename, '']
#     out_filename = filename_s[0] + '_T.' + filename_s[1]
#     f = open(out_filename, 'wb')
#     row_len = len(data[0])
#     for i in xrange(row_len):
#         if i != 0:
#             f.write('\n')
#         f.write(sep.join(data[:, i]))
#         f.flush()
#     f.close()
#     if del_orig:
#         os.remove(filename)
#
#
# def transpose_files(graph_name, runs=1, sep='\t'):
#     ratios, deltatau, deltastep, graph_name, store_iterations, cas, ew = get_network_details(graph_name)
#     debug_msg("  >>> Transposing files for {}".format(graph_name), level=0)
#     path = config.graph_source_dir + "weights/" + graph_name + "/"
#     filename = graph_name + '_' + str(store_iterations) + "_" + str(deltatau).replace('.', '')
#     for i in xrange(0, runs):
#         #try:
#             fname = path + filename + "_run_" + str(i) + ".txt"
#             debug_msg(" >> Transposing: {}".format(filename + "_run_" + str(i) + ".txt"), level=1)
#             transpose_txt(fname, del_orig=False)
#         #except Exception:
#         #    print debug_msg("\x1b[31mERROR: File not found: {}\x1b[00m".format(filename + "_run_" + str(i) + ".txt"))
#         #    continue


# def avg_over_files(graph_name, runs=10, sep='\t', suffix="_T"):
#     a, b, iterations, gname, store_itas, acs = get_network_details(graph_name)
#     debug_msg("  >>> Averaging files for {}".format(gname), level=0)
#
#     for a_val in a:
#         for b_val in b:
#             path = config.graph_source_dir + "weights/" + graph_name + "/"
#             basic_filename = graph_name + '_' + str(iterations) + '_iterations_' + str(a_val).replace('.', '') + '_' + str(b_val).replace('.', '')
#             out_f = open(path + basic_filename + '_avg' + suffix + '.txt', 'wb')
#             basic_filename += '_run_'
#             debug_msg('Processing Average for: ' + basic_filename, level=1)
#             files = [open(path + basic_filename + str(i) + suffix + '.txt', 'r') for i in xrange(runs)]
#             data = np.array([np.fromstring(i.readline(), sep=sep) for i in files])
#             first = True
#             node_num = 0
#             while True:
#                 lengths = [len(i) for i in data]
#                 max_len = max(lengths)
#                 if max_len <= 1:
#                     break
#                 if first:
#                     first = False
#                 else:
#                     out_f.write('\n')
#                 if any(i != max_len for i in lengths):
#                     data = np.array([np.r_[i, [i[-1]] * (max_len - len(i))] for i in data])
#                 means = []
#                 #for l in data:
#                 #    means.append(sum(l)/float(len(l)))
#                 means = np.mean(data, axis=0)
#                 out_f.write(sep.join(str(i) for i in means))
#                 data = np.array([np.fromstring(i.readline(), sep=sep) for i in files])
#             out_f.close()
#             for opened_file in files:
#                 opened_file.close()


# def mean_per_iteration(graph_name, runs=100, sep='\t', postfix=""):
#     ratios, dtaus, iterations, gname, store_itas, acs = get_network_details(graph_name)
#     debug_msg("  >>> Mean per Iteration for {}".format(gname), level=0)
#
#     for ratio in ratios:
#         for dtau in dtaus:
#             path = config.graph_source_dir + "weights/" + graph_name + "/"
#             basic_filename = graph_name + '_' + str(iterations) + '_iterations_' + str(ratio).replace('.', '') + '_' + str(dtau).replace('.', '')
#             out_f = open(path + basic_filename + '_mean_per_ita'+postfix+'.txt', 'wb')
#             basic_filename += '_run_'
#             debug_msg('Processing Mean for: ' + basic_filename, level=1)
#             files = [open(path + basic_filename + str(i) + postfix +'.txt', 'r') for i in xrange(runs)]
#             iteration_averages = defaultdict(lambda : defaultdict(int))
#
#             # parse files
#             max_itas = 0
#             for fdx, f in enumerate(files):
#                 for ldx, l in enumerate(f):
#                     vals = np.fromstring(l, sep=sep)
#                     iteration_averages[ldx][fdx] += sum(vals)/len(vals)
#                     if ldx > max_itas:
#                         max_itas = ldx
#
#             for i in xrange(max_itas):
#                 if len(iteration_averages[i]) < runs:
#                     missing_fdx = set(xrange(0,runs)).difference(set(iteration_averages[i].keys()))
#                     for fdx in missing_fdx:
#                         iteration_averages[i][fdx] = iteration_averages[i-1][fdx]
#
#                 #print iteration_averages[i].values()
#                 #print "-------------"
#                 #mean = str(np.mean(iteration_averages[i].values()))
#                 #print mean
#                 out_f.write(str(np.mean(iteration_averages[i].values()))+"\t"+str(float(dtau)*i*store_itas)+"\n")
#             out_f.close()
#             for opened_file in files:
#                 opened_file.close()