import os
import matplotlib.pyplot as plt

class_name_list = ('bart', 'granpa', 'homer', 'lisa', 'marge')


def get_train_txt():
    list_ = os.listdir('/home/nomul/darknet/data/img')
    dstpath = '/home/nomul/darknet/data/train.txt'

    for i in list_:
        if i[-3:] == 'jpg':
            with open(dstpath, 'a') as wf:
                wf.write('/home/nomul/darknet/data/img/t/' + i + '\n')


def write_path(src_path, dst_fd):
    for i in os.listdir(src_path):
        dst_fd.write(src_path + '/' + i + '\n')


def get_test_txt():
    with open('/home/nomul/Downloads/darknet-master/test.txt', 'a') as fd:
        for i in class_name_list:
            write_path('/home/nomul/simpson_test/' + i, fd)


class ResultDescriptor:
    def __init__(self, class_name):
        self.class_name = class_name
        self.confidence = dict()
        for i in class_name_list:
            self.confidence[i] = 0


def get_result_list():
    result_list = []
    dict_count = dict()
    for i in class_name_list:
        dict_count[i] = 0

    with open('/home/nomul/Downloads/darknet-master/result.txt', 'r') as fd:
        list_lines = fd.readlines()
        idx = 0
        max_len = len(list_lines)

        while idx < max_len:
            line = list_lines[idx]
            if line[:5] == 'Enter':
                idx_start = line.find('simpson_test') + 13
                idx_end = idx_start + line[idx_start:].find('/')
                class_simpson = line[idx_start:idx_end]
                dict_count[class_simpson] += 1

                result_desc = ResultDescriptor(class_simpson)

                while idx + 1 < max_len:
                    next_line = list_lines[idx + 1]
                    if next_line[:5] == 'Enter':
                        break
                    else:
                        predict = next_line[:next_line.find(':')]
                        result_desc.confidence[predict] = eval(next_line[next_line.find(' '):next_line.find('%')])
                        idx += 1
                        continue
                result_list.append(result_desc)
            idx += 1

    return result_list, dict_count


def get_metrics(result_list, threshold):
    dict_tp = dict()
    dict_fp = dict()
    dict_tn = dict()
    for i in class_name_list:
        dict_tp[i] = 0
        dict_fp[i] = 0
        dict_tn[i] = 0

    for i in result_list:
        class_simpson = i.class_name
        for name in i.confidence:
            if name == class_simpson:
                if i.confidence[name] >= threshold:
                    dict_tp[name] += 1
                else:
                    dict_tn[name] += 1
            else:
                if i.confidence[name] >= threshold:
                    dict_fp[name] += 1
                else:
                    pass

    dict_precision = dict()
    dict_recall = dict()
    for i in class_name_list:
        dict_precision[i] = dict_tp[i] / (dict_tp[i] + dict_fp[i])
        dict_recall[i] = dict_tp[i] / (dict_tp[i] + dict_tn[i])

    return dict_precision, dict_recall


# get_train_txt()

# get_test_txt()

res_list, res_count = get_result_list()
print('[count]')
for i in res_count:
    print('%7s : %7d' % (i, res_count[i]))

for thres in range(0, 120, 20):
    print('\n\n★★★threshold : %d★★★' % thres)

    precision, recall = get_metrics(res_list, thres)

    print('[precision]')
    for i in precision:
        print('%7s : %7.4f' % (i, precision[i]))

    print('\n[recall]')
    for i in recall:
        print('%7s : %7.4f' % (i, recall[i]))

    # get F1 score
    F1_score = dict()
    for i in recall:
        F1_score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i])

    print('\n[F1 score]')
    for i in F1_score:
        print('%7s : %7.4f' % (i, F1_score[i]))

    # plot histogram
    x = range(len(precision))
    x_label = [i for i in precision]
    y1 = [precision[i] for i in precision]
    y2 = [recall[i] for i in recall]
    y3 = [F1_score[i] for i in F1_score]

    plt.title('threshold : %d' % thres)
    plt.bar([i - 0.25 for i in x], y1, label='precision', width=0.25)
    plt.bar([i for i in x], y2, label='recall', width=0.25)
    plt.bar([i + 0.25 for i in x], y3, label='F1 score', width=0.25)

    plt.xticks(x, x_label)

    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('threshold_%d.png' % thres)
    plt.show()

