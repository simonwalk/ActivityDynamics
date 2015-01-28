import pandas as pd
import numpy as np
import os
import shutil


debug = False

wiki_selector = 2

instances = ["BEACHAPEDIA", "APBR", "CHARACTERDB", "SMWORG", "W15M", "AARDNOOT", "AUTOCOLLECTIVE", "CWW", "NOBBZ",
             "StackOverflow", "EnglishStackExchange", "HistoryStackExchange", "MathStackExchange"]
folders = ["beachapedia_org_change_network.txt.sorted_results",
           "apbrwiki_com_change_network.txt.sorted_results",
           "characterdb_cjklib_org_change_network.txt.sorted_results",
           "semantic-mediawiki_org_change_network.txt.sorted_results",
           "wiki_15m_cc_change_network.txt.sorted_results",
           "aardnoot_nl_change_network.txt.sorted_results",
           "autonomecollective_org_change_network.txt.sorted_results",
           "cumbriawindwatch_co_uk_change_network.txt.sorted_results",
           "nobbz_de_change_network.txt.sorted_results",
           "StackOverflow", "EnglishStackExchange",
           "HistoryStackExchange", "MathStackExchange"]
instance = instances[wiki_selector]

#root_path = "/Volumes/DataStorage/Programming/"
root_path = "/Users/simon/Desktop/Projects/"
root_path_ratios = root_path + "DynamicNetworksResults/graph_binaries/empirical_input/"

#root_path_ratios = "DynamicNetworksResults/graph_binaries/ratios/"

storage_path = root_path_ratios + instance + "_empirical_input.txt"
init_weights_path = root_path_ratios + instance + "_weights.txt"
source_path = root_path + "DynamicNetworks/data_preparation/sorted_wikis/"+folders[wiki_selector]+"/"

print "Processing: {}".format(source_path)

binaries_path = root_path + "DynamicNetworksResults/graph_binaries/GT/"
gnames = [instance + "_0PERC_RAND_run_0.gt", instance + "_0PERC_SPEC_run_0.gt"]
fnames = [binaries_path + instance + "_0PERC_RAND/", binaries_path + instance + "_0PERC_SPEC/"]

copy_file = source_path + "weighted_net.gt"



for i in range(2):
    gname = gnames[i]
    fname = fnames[i]
    if not os.path.exists(fname):
        os.makedirs(fname)
    shutil.copy(copy_file, fname + gname)

# loading pickled files
df_posts = pd.read_pickle(source_path + "user_df_posts.ser")
df_replies = pd.read_pickle(source_path + "user_df_replies.ser")

# opening ratios file for activity dynamics simulation
f = open(storage_path, "wb")
g = open(init_weights_path, "wb")

max_row = len(df_posts.index)
f.write("dx\tagg_activity\tposts\treplies\tnum_users\n")

print "Sum of Posts: {}".format(np.nansum(df_posts))
print "Sum of Replies: {}".format(np.nansum(df_replies))
print "Number of months: {}".format(len(df_posts))

for i in xrange(0, max_row):
    posts_current = np.nansum(df_posts.ix[i,:])+1


    posts_next = np.nansum(df_posts.ix[i+1,:])+1
    replies_current = np.nansum(df_replies.ix[i,:])+1
    replies_next = np.nansum(df_replies.ix[i+1,:])+1
    num_users = df_posts.ix[i+1,:].count()+1
    dx = (replies_current + posts_current) - (replies_next + posts_next)
    current_activity = (replies_current + posts_current)

    # write to file
    f.write(str(dx) + "\t" + str(current_activity) + "\t" + str(posts_current) + "\t" +
            str(replies_current) + "\t" + str(num_users) + "\n")

    if debug:
        print "Month {}".format(i)
        print "------------------------"
        print "  Posts: {} - {}".format(posts_current, posts_next)
        print "  Replies: {} - {}".format(replies_current, replies_next)
        print "  Users: {}".format(num_users)

    if i == max_row-2:
        break

f.close()
g.close()