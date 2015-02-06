import pandas as pd
import numpy as np
import os
import shutil
from graph_tool.all import *

debug = False

#wiki_selector = 10
#wiki_selector = -2
wiki_selector = 9
is_wiki = False

instances = ["BEACHAPEDIA", "APBR", "CHARACTERDB", "SMWORG", "W15M", "AARDNOOT", "AUTOCOLLECTIVE", "CWW", "NOBBZ",
             "StackOverflow", "EnglishStackExchange", "HistoryStackExchange", "MathStackExchange", "BeerStackExchange",
             "NematodesWIKI", "AWAYCITY", "CDB", "CCC"]
folders = ["beachapedia_org_collab_network.txt.sorted_results",
           "apbrwiki_com_change_network.txt.sorted_results",
           "characterdb_cjklib_org_change_network.txt.sorted_results",
           "semantic-mediawiki_org_change_network.txt.sorted_results",
           "wiki_15m_cc_change_network.txt.sorted_results",
           "aardnoot_nl_change_network.txt.sorted_results",
           "autonomecollective_org_change_network.txt.sorted_results",
           "cumbriawindwatch_co_uk_change_network.txt.sorted_results",
           "nobbz_de_collab_network.txt.sorted_results",
           "StackOverflow", "EnglishStackExchange",
           "HistoryStackExchange", "MathStackExchange", "BeerStackExchange",
           "nematodes_org_collab_network.txt.sorted_results",
           "awaycity_com_collab_network.txt.sorted_results",
           "characterdb_cjklib_org_collab_network.txt.sorted_results",
           "events_ccc_de_collab_network.txt.sorted_results"]
instance = instances[wiki_selector]

root_path = "/Volumes/DataStorage/Programming/"
#root_path = "/Users/simon/Desktop/ActivityDynamics/results/graph_sources/collaboration_networks/"
#root_path = "/Users/simon/Desktop/"
#root_path = "/opt/datasets/stackexchange/"
#root_path = "/Users/simon/Desktop/"

root_path_ratios = root_path + "ActivityDynamics/results/graph_binaries/empirical_input/"
storage_path = root_path_ratios + instance + "_empirical_input.txt"
init_weights_path = root_path_ratios + instance + "_weights.txt"

if is_wiki:
    source_path = root_path + "ActivityDynamics/results/graph_sources/collaboration_networks/"+folders[wiki_selector]+"/"
else:
    source_path = root_path + "ActivityDynamics/datasets/" + folders[wiki_selector]+"/"

print "Processing: {}".format(source_path)
binaries_path = root_path + "ActivityDynamics/results/graph_binaries/GT/"
gnames = [instance + "_run_0.gt"]
fnames = [binaries_path + instance + "/"]
copy_file = source_path + "net.gt"

gname = gnames[0]
fname = fnames[0]
if not os.path.exists(fname):
    os.makedirs(fname)
shutil.copy(copy_file, fname + gname)

# loading pickled files
df_posts = pd.read_pickle(source_path + "user_df_posts.ser")
df_replies = pd.read_pickle(source_path + "user_df_replies.ser")

graph = load_graph(source_path + "net.gt")
graph.clear_filters()

id_pmap = graph.vp["nodeID"]
id_to_vertex_dict = {int(id_pmap[v]): v for v in graph.vertices()}


# opening ratios file for activity dynamics simulation
max_row = len(df_posts.index)
print "Sum of Posts: {}".format(np.nansum(df_posts))
print "Sum of Replies: {}".format(np.nansum(df_replies))
print "Number of months: {}".format(len(df_posts))
print "Number of Users: {}".format(graph.num_vertices())


keys = set(id_to_vertex_dict.keys())
df_posts = pd.read_pickle(source_path + "user_df_posts.ser")
df_replies = pd.read_pickle(source_path + "user_df_replies.ser")
df_posts.fillna(0, inplace=True)
df_replies.fillna(0, inplace=True)
#FLO START
#sum(axis=1) sum over row
df_result = pd.DataFrame(columns=['posts'], data=(df_posts.sum(axis=1) + 1))
df_result['replies'] = (df_replies.sum(axis=1) + 1)
df_result['agg_activity'] = df_result['posts'] + df_result['replies'] #attention now you have +2 (+1 from posts & +1 from replies)
df_result['num_users'] = ((df_replies > 0) | (df_posts > 0)).sum(axis=1)
# delta x: current agg activity - next agg activity. need to shift with -1, since rolling_apply is shifted by one
df_result['dx'] = pd.rolling_apply(df_result['agg_activity'], func=lambda x: x[0] - x[-1], window=2,
                                   min_periods=2).shift(-1)
#resolve user ids to vertex ids
columns_resolved = np.array(map(int, [id_to_vertex_dict[i] for i in df_posts.columns]))
df_result['active_user_ids'] = ((df_replies > 0) | (df_posts > 0)).apply(func=lambda x: ','.join(map(str, columns_resolved[np.array(x)])), axis=1)
#sort columns
df_result = df_result[['dx', 'agg_activity', 'posts', 'replies', 'num_users', 'active_user_ids']]
#FLO END
df_result.to_csv(storage_path, sep="\t", header=True)