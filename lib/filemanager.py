import datetime
import numpy as np
import os as os
import shutil


percentage_experiments = ["_PERC_RAND", "_PERC_SPEC"]
percentage = np.arange(0.0, 1.01, 0.01)

is_notebook = False
is_externalHDD = not is_notebook

if is_notebook:
    g_path = "/Volumes/StorageDisk/"
elif is_externalHDD:
    g_path = "/Volumes/MyBook4TB/"
else:
    g_path = "/Volumes/DataCluster/Git-Repos/"

source = g_path + "DynamicNetworks/"

def remove_file(fname):
    try:
        os.remove(fname)
    except:
        pass


def create_layout(path):
    folders = [path,
               path + "/averages",
               path + "/errors",
               path + "/figures",
               path + "/GML",
               path + "/summary",
               path + "/weights",
               path + "/figures/active_inactive",
               path + "/figures/average_over_time",
               path + "/figures/eigenvalues",
               path + "/figures/functions",
               path + "/figures/graph",
               path + "/figures/scatterplots",
               path + "/figures/weights_over_time"]
    try:
        for folder in folders:
            if not os.path.exists(folder):
                debug_msg("Creating folder: {}".format(folder))
                os.makedirs(folder)
    except Exception, e:
        debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))

def move_files_to_paper_structure(percentage_experiment):
    # averages:
    path = g_path + graph_name + percentage_experiment
    averages_source = source + "graph_sources/weights/"

    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100)))) + "/"
            cf = averages_source + current_graph_name
            average_files = [ f for f in os.listdir(cf) if os.path.isfile(cf+f) and "_avg.txt" in f]
            for avgf in average_files:
                debug_msg("Moving: \x1b[36m"+avgf+"\x1b[00m")
                shutil.move(cf+avgf, path+"/averages")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))

    averages_source = source + "graph_sources/errors/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100))))
            cf = averages_source
            error_files = [f for f in os.listdir(cf) if os.path.isfile(cf+f) and current_graph_name in f]
            for ef in error_files:
                debug_msg("Moving: \x1b[36m"+ef+"\x1b[00m")
                shutil.move(cf+ef, path+"/errors")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))

    try:
        source_dir = source + "graph_binaries/GML/"
        current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_0PERC")
        cf = source_dir + current_graph_name + "/" + current_graph_name +"_run_0.gml"
        debug_msg("Copying: \x1b[36m"+cf+"\x1b[00m")
        #remove_file(path+"/GML/" + current_graph_name +"_run_0.gml")
        shutil.copy(cf, path+"/GML/")
    except Exception, e:
        debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))

    averages_source = source + "graph_binaries/summary/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100))))
            cf = averages_source
            error_files = [f for f in os.listdir(cf) if os.path.isfile(cf+f) and current_graph_name in f]
            for ef in error_files:
                debug_msg("Moving: \x1b[36m"+ef+"\x1b[00m")
                shutil.move(cf+ef, path+"/summary")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))

    averages_source = source + "graph_sources/weights/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100)))) + "/"
            cf = averages_source + current_graph_name
            debug_msg("Moving: \x1b[36m"+current_graph_name+"\x1b[00m")
            shutil.move(cf, path+"/weights")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))
            continue


    averages_source = source + "plots/active_inactive/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100)))) + "/"
            cf = averages_source + current_graph_name
            debug_msg("Moving: \x1b[36m"+current_graph_name+"\x1b[00m")
            shutil.move(cf, path+"/figures/active_inactive")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))
            continue

    averages_source = source + "plots/average_over_time/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100))))
            cf = averages_source
            error_files = [f for f in os.listdir(cf) if os.path.isfile(cf+f) and current_graph_name in f]
            for ef in error_files:
                debug_msg("Moving: \x1b[36m"+ef+"\x1b[00m")
                shutil.move(cf+ef, path+"/figures/average_over_time")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))


    averages_source = source + "plots/eigenvalues/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100))))
            cf = averages_source
            error_files = [f for f in os.listdir(cf) if os.path.isfile(cf+f) and current_graph_name in f]
            for ef in error_files:
                debug_msg("Moving: \x1b[36m"+ef+"\x1b[00m")
                shutil.move(cf+ef, path+"/figures/eigenvalues")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))

    averages_source = source + "plots/functions/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(p*100)))
            cf = averages_source
            error_files = [f for f in os.listdir(cf) if os.path.isfile(cf+f) and current_graph_name in f]
            for ef in error_files:
                debug_msg("Moving: \x1b[36m"+ef+"\x1b[00m")
                shutil.move(cf+ef, path+"/figures/functions")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))


    averages_source = source + "plots/scatterplots/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100)))) + "/"
            cf = averages_source + current_graph_name
            debug_msg("Moving: \x1b[36m"+current_graph_name+"\x1b[00m")
            shutil.move(cf, path+"/figures/scatterplots")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))
            continue

    averages_source = source + "plots/weights_over_time/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100)))) + "/"
            cf = averages_source + current_graph_name
            debug_msg("Moving: \x1b[36m"+current_graph_name+"\x1b[00m")
            shutil.move(cf, path+"/figures/weights_over_time")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))
            continue


    averages_source = source + "graphs/"
    for p in percentage:
        try:
            current_graph_name = graph_name + percentage_experiment.replace("_PERC", "_{}PERC".format(int(round(p*100))))
            cf = averages_source
            error_files = [f for f in os.listdir(cf) if os.path.isfile(cf+f) and current_graph_name in f]
            for ef in error_files:
                debug_msg("Moving: \x1b[36m"+ef+"\x1b[00m")
                shutil.move(cf+ef, path+"/figures/graph")
        except Exception, e:
            debug_msg("\x1b[41mERROR:: {}\x1b[00m".format(e))


def debug_msg(msg):
    print "  \x1b[33m-I/O-\x1b[00m [\x1b[36m{}\x1b[00m] \x1b[32m{}\x1b[00m".format(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), msg)

for exp in percentage_experiments:
    graph_name = "Random_MEDIUM"
    #graph_name = "SBM_ASC_SAME_CONN"
    #graph_name = "PrefAttach_MEDIUM"
    #graph_name = "Karate"
    create_layout(g_path + graph_name + exp)
    move_files_to_paper_structure(exp)