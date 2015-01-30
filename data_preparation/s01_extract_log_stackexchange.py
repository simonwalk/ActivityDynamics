from __future__ import division

from util import *
import xml.etree.cElementTree as ElementTree
from collections import defaultdict


def extract_log_stackexchange(folder, posts_file='Posts.xml', comments_file='Comments.xml'):
    print_f('extract log from stackexchange')
    if folder.endswith('.7z'):
        status = 0
        print_f('got zipped file. try to unzip: ', folder)
        zip_file_path, zip_file_name = folder.rsplit('/', 1)
        zip_file_path += '/'
        folder = folder.rsplit('.', 1)[0] + '/'
        print_f('create folder:', folder)
        os.system('rm -r ' + folder)
        status = os.system('mkdir -p ' + folder)
        if status > 0:
            print_f('error creating folder:', folder, '\nerror code:', status)
        print_f('unzip: ' + zip_file_path + zip_file_name)
        cmd = '7za e ' + zip_file_path + zip_file_name + ' -y -o' + folder
        print '\tcmd:', cmd
        status = os.system(cmd)
        if status > 0:
            print_f('error unzipping. error-code:', status)
            exit()
        print_f('done')
    posts_file = folder + posts_file
    comments_file = folder + comments_file
    LINK_A_Q = True
    LINK_C_A = True
    LINK_C_Q = True
    LINK_A_A = False
    LINK_C_C = False

    POST_TYPE_QUESTION = 1
    POST_TYPE_ANSWER = 2
    POST_TYPE_COMMENT = 3
    flags = {name: val for name, val in sorted(locals().iteritems(), key=lambda x: x[0]) if name.startswith('LINK')}

    merged_filename = folder + "merged_file"
    print_f('convert:', posts_file)
    with open(merged_filename, 'w') as f:
        # f.write('#timestamp,post-id,user-id,type,parent-post-id\n')
        for counter, (event, elem) in enumerate(ElementTree.iterparse(posts_file)):
            if counter % 1000000 == 0:
                print_f("processed", counter / 1000000, "m")
            try:
                post_type_id = int(elem.get('PostTypeId'))
                parent_post_id = ''
                if post_type_id == POST_TYPE_ANSWER:
                    try:
                        parent_post_id = int(elem.get('ParentId'))
                        answer = True
                        post_type = POST_TYPE_ANSWER
                    except:
                        print_f('Answer but unparseable ParendId: ->', elem.get('ParentId'), '<-')
                        answer = False
                elif post_type_id == 1:
                    post = True
                    post_type = POST_TYPE_QUESTION
                else:
                    raise
            except:
                elem.clear()
                continue

            if post or answer:
                try:
                    user_id = int(elem.get('OwnerUserId'))
                    if user_id <= 0:
                        raise
                except:
                    elem.clear()
                    continue
                f.write(elem.get('CreationDate') + ',' + elem.get('Id') + ',' + str(user_id) + ',' + str(post_type) + ',' + str(parent_post_id) + '\n')
            elem.clear()
        print_f(posts_file, 'done')
        print_f('convert:', comments_file)
        for counter, (event, elem) in enumerate(ElementTree.iterparse(comments_file)):
            if counter % 1000000 == 0:
                print_f("processed", counter / 1000000, "m")
            try:
                parent_post_id = int(elem.get('PostId'))
                user_id = int(elem.get('UserId'))
                if user_id <= 0:
                    raise
            except:
                elem.clear()
                continue
            f.write(elem.get('CreationDate') + ',' + elem.get('Id') + ',' + str(user_id) + ',' + str(POST_TYPE_COMMENT) + ',' + str(parent_post_id) + '\n')
        print_f(comments_file, 'done')

    print_f('sort merged file')
    cmd = "sort -t ',' -k 1,1 -k 4,4 " + merged_filename + " > " + merged_filename + ".sorted"
    print cmd
    os.system(cmd)
    print_f('done')
    logfile_filename = folder + "extracted_logfile"

    print_f('construct logfile:')
    question_id_to_user_id = dict()
    answer_id_to_user_id = dict()
    comment_id_to_user_id = dict()
    all_answerer_ids_to_question = defaultdict(set)
    all_commenter_ids_to_post = defaultdict(set)
    comment_to_other_comment_user_ids = defaultdict(set)
    answers_without_question = 0
    comment_without_parent = 0
    questions = 0
    answers = 0
    comments = 0
    events_dist = defaultdict(int)
    with open(logfile_filename, 'w') as outfile:
        with open(merged_filename + '.sorted', 'r') as infile:
            for counter, line in enumerate(infile):
                if counter % 1000000 == 0:
                    print_f("processed", counter / 1000000, "m")
                line = line.strip('\n')
                timestamp, post_id, user_id, post_type, parent_post_id = line.split(',')
                post_id, user_id, post_type = int(post_id), int(user_id), int(post_type)
                line_prefix = timestamp + '\t'
                if post_type == POST_TYPE_QUESTION:
                    question_id_to_user_id[post_id] = user_id
                    questions += 1
                    outfile.write(line_prefix + str(user_id) + '\n')
                    events_dist['questions'] += 1
                else:
                    parent_post_id = int(parent_post_id)
                    if post_type == POST_TYPE_ANSWER:
                        answers += 1
                        answer_id_to_user_id[post_id] = user_id
                        if LINK_A_Q:
                            try:
                                parent_user = question_id_to_user_id[parent_post_id]
                                outfile.write(line_prefix + str(user_id) + '\t' + str(parent_user) + '\n')
                                events_dist['answers'] += 1
                            except KeyError:
                                answers_without_question += 1
                        if LINK_A_A:
                            for other_answerer in all_answerer_ids_to_question[parent_post_id]:
                                outfile.write(line_prefix + str(user_id) + '\t' + str(other_answerer) + '\n')
                                events_dist['answers amongst themselves'] += 1
                            all_answerer_ids_to_question[parent_post_id].add(user_id)
                    elif post_type == POST_TYPE_COMMENT:
                        comments += 1
                        comment_id_to_user_id[post_id] = user_id
                        try:
                            if LINK_C_A:
                                parent_user = answer_id_to_user_id[parent_post_id]
                                outfile.write(line_prefix + str(user_id) + '\t' + str(parent_user) + '\n')
                                events_dist['comments to answers'] += 1
                            else:
                                raise KeyError
                        except KeyError:
                            try:
                                if LINK_C_Q:
                                    parent_user = question_id_to_user_id[parent_post_id]
                                    outfile.write(line_prefix + str(user_id) + '\t' + str(parent_user) + '\n')
                                    events_dist['comments to questions'] += 1
                            except KeyError:
                                comment_without_parent += 1
                        if LINK_C_C:
                            for other_commenter in all_commenter_ids_to_post[parent_post_id]:
                                outfile.write(line_prefix + str(user_id) + '\t' + str(other_answerer) + '\n')
                                events_dist['comments amongst themselves'] += 1
                            all_commenter_ids_to_post[parent_post_id].add(user_id)
                    else:
                        print_f('unknown post-type:', post_type)
    num_events = np.sum(events_dist.values())
    print_f('#events:', num_events)
    print_f('num questions:', questions)
    print_f('num answers:', answers)
    print_f('num comments:', comments)
    print_f('answers without question:', str(answers_without_question / answers * 100)[:5], '%')
    print_f('comments without parent:', str(comment_without_parent / comments * 100)[:5], '%')
    print_f('logfile stored')
    print_f('plot events stats')
    events_dist = {key: val / num_events * 100 for key, val in events_dist.iteritems()}
    X = np.arange(len(events_dist))
    # plt.bar(X, events_dist.values(), align='center', width=0.5)
    # plt.xticks(X, events_dist.keys(), rotation=-10)
    # ymax = max(events_dist.values()) + 1
    # plt.ylabel('percentage')
    # plt.xlabel('type')
    # plt.ylim(0, ymax)
    # plt.grid()
    # plt.savefig(folder + 'extracted_logfile_event_stat.png', dpi=150)
    print_f('extract log from stackexchange done')
    return logfile_filename


if __name__ == '__main__':
    start = datetime.datetime.now()
    extract_log_stackexchange('/opt/datasets/stackoverflow/' if len(sys.argv) <= 1 else sys.argv[1])
    print '============================='
    print 'Overall Time:', str(datetime.datetime.now() - start)
    print 'ALL DONE -> EXIT'