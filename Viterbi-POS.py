import csv
from sklearn.model_selection import KFold

dictionary = dict()
tags = dict()
tag_list = []
bigram = dict()
in_words = []
bag_words = []
bag_tags = []
test_order = []
test_words = []

def eval(keys, predictions):
    """ Simple minded eval system: file1 is gold standard, file2 is system outputs. Result is straight accuracy. """

    count = 0.0
    correct = 0.0

    for key, prediction in zip(keys, predictions):
        key = key.strip()
        prediction = prediction.strip()
        if key == '': continue
        count += 1
        if key == prediction: correct += 1

    print("Evaluated ", count, " tags.")
    print("Accuracy is: ", correct / count)
    return correct/count


def reading():
    f = open("/Users/ashwinsankar/Desktop/NLP /HW2/berp-POS-training.txt", 'rt')
    reader = csv.reader(f, delimiter='\t')
    in_words.append(0)
    bag_words.append("<s>")
    bag_tags.append("<s>")
    for row in reader:
        if row:
            in_words.append(row[0])
            bag_words.append(row[1])
            bag_tags.append(row[2])
        else:
            t_c = int(in_words[-1]) + 1
            in_words.append(str(t_c))
            bag_words.append("</s>")
            bag_tags.append("</s>")
            in_words.append(0)
            bag_words.append("<s>")
            bag_tags.append("<s>")
    t_c = int(in_words[-1]) + 1
    in_words.append(str(t_c))
    bag_words.append("</s>")
    bag_tags.append("</s>")


def dictionary_filling(train_words, train_tags):
    for word, tag in zip(train_words, train_tags):
        found = 0
        if word in dictionary:
            for j in range(len(dictionary[word])):
                if tag in dictionary[word][j]:
                    dictionary[word][j][tag] += 1
                    found = 1
                    break
            if (found == 0):
                dictionary[word].append({tag: 1})
        else:
            dictionary[word] = [{tag: 1}]


def tag_dictionary():
    for tag in bag_tags:
        if tag not in tags:
            tags[tag] = 1
        else:
            tags[tag] += 1


def prob(count, total_count):
    return count / total_count


def init_transitional_probability():
    for x, y in zip(bag_tags, bag_tags[1:]):
        if x + y in bigram:
            bigram[x + y] += 1
        else:
            bigram[x + y] = 0
            # todo smoothing


def trans_prob(tag1, tag2):
    if tag1 + tag2 not in bigram:
        return 0
    else:
        return bigram[tag1 + tag2]


def smoothing():
    for tag1 in tags:
        for tag2 in tags:
            if tag1 + tag2 not in bigram:
                bigram[tag1 + tag2] = 2
            else:
                bigram[tag1 + tag2] += 2
    for tag in tags:
        tags[tag] += len(tags)*2


def observation_prob(word, tag):
    ob_count = 0
    if tag not in dictionary[word][0]:
        ob_count = 0
    else:
        ob_count = dictionary[word][0][tag]
    return prob(ob_count, tags[tag])


def viterbi(sequence):
    maximum = 0
    for i in range(len(sequence)):
        if sequence[i] not in dictionary:
            sequence[i] = 'UNK'
            # print('Encountered UNK')

    row, column = len(tags), len(sequence)
    V = [[0 for x in range(column)] for y in range(row)]
    back = [[0 for x in range(column)] for y in range(row)]
    for i in range(0, column):
        for j in range(0, row):
            if i == 0:
                V[3][i] = 1
            else:
                maximum = 0
                for k in range(0, row):
                    trans_count = trans_prob(tag_list[k], tag_list[j])
                    if maximum < V[k][i - 1] * prob(trans_count, tags[tag_list[k]]):
                        maximum = V[k][i - 1] * prob(trans_count, tags[tag_list[k]])
                        back[j][i] = k
                V[j][i] = maximum * observation_prob(sequence[i], tag_list[j])
    return row, column, V, back


def decode(row, column, V, back):
    output_list = []
    maxval = 0
    for i in range(2, column):
        maxval = 0
        argmax = -1
        for j in range(row):
            if maxval < V[j][i]:
                maxval = V[j][i]
                argmax = back[j][i]
        if (argmax != -1):
            output_list.append(tag_list[argmax])
    return output_list


def unknown_words(threshold=1):
    if 'UNK' not in dictionary:
        dictionary['UNK'] = [{'NN': 0}]
    fakedict = dictionary.copy()
    for word in fakedict:
        for tag in fakedict[word][0]:
            if word != 'UNK':
                if dictionary[word][0][tag] <= threshold:
                    if tag not in dictionary['UNK'][0]:
                        dictionary['UNK'][0][tag] = dictionary[word][0][tag]
                    else:
                        dictionary["UNK"][0][tag] += dictionary[word][0][tag]
                    del dictionary[word]


def read_test_words():
    f = open("/Users/ashwinsankar/Desktop/NLP /HW2/assgn2-test-set.txt", 'rt')
    reader = csv.reader(f, delimiter='\t')
    test_words.append("<s>")
    for row in reader:
        if row:
            test_order.append(row[0])
            test_words.append(row[1])
        else:
            test_words.append("</s>")
            test_words.append("<s>")
    test_words.append("</s>")

def removestarttag():
    for words in test_words:
        if words=='<s>' or words=='</s>':
            test_words.remove(words)

if __name__ == "__main__":
    reading()
    test_words,test_tags = [],[]
    train_words,train_tags = [] ,[]
    dictionary_filling(bag_words, bag_tags)
    unknown_words(3)
    tag_dictionary()
    tag_list = list(tags.keys())
    tag_list.sort()
    init_transitional_probability()
    smoothing()
    read_test_words()
    sequence = []
    test_sequence = []
    predict = []
    correct = 0
    accuracy = 0
    for word in test_words:
        if word == '</s>':
            sequence.append(word)
            row, column, V, back = viterbi(sequence)
            output_list = decode(row, column, V, back)
            # print(sequence, '\n', output_list)
            predict.extend(output_list)
            sequence = []
        else:
            sequence.append(word)
    removestarttag()
    removestarttag()
    f =open("/Users/ashwinsankar/Desktop/NLP /HW2/Sankaralingam-Ashwin-assgn2-test-output.txt",'w')
    for i in range(len(predict)):
        f.write(test_order[i])
        f.write('\t')
        f.write(test_words[i])
        f.write('\t')
        f.write(predict[i])
        f.write('\n')
        if test_words[i] == '.':
            f.write('\n')