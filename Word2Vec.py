import jieba
from jieba import posseg
from PyDBC import PyDBC
import re
from tqdm import tqdm
import tensorflow as tf
import math
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pyDBC = PyDBC()

window_size = 5 # 词窗口大小

single_window_size = (window_size-1)/2
single_window_size = int(single_window_size)

def pre_process(content):
    # 替换 <br> to '。'
    content = content.replace('<br>', '。')
    content = content.replace('<br/>', '。')
    content = content.replace('\t', '。')
    content = content.replace('\n', '。')
    content = content.replace('\xa0', '')

    # 去掉 （） 及其内文字
    while True:
        start = content.find('（')
        end = content.find('）', start)
        if start != -1 and end != -1:
            content = content[:start] + content[end+1:]
        else:
            break

    # 去掉 ()及其内文字
    while True:
        start = content.find('(')
        end = content.find(')', start)
        if start != -1 and end != -1:
            content = content[:start] + content[end + 1:]
        else:
            break

    # 去掉 《》及其内文字
    while True:
        start = content.find('《')
        end = content.find('》', start)
        if start != -1 and end != -1:
            content = content[:start] + content[end + 1:]
        else:
            break

    # 去除顿号
    content = content.replace('、', '')

    # 去除数字
    content = re.sub('\d', '', content)

    # 去除空格
    content = content.replace(' ', '')


    return content


def statistics_word():
    datalist = pyDBC.get_all(table='t_case', columns=['id', 'content'])
    wordmap = {}
    for data in tqdm(datalist):
        id_ = data[0]
        content = data[1]
        content = pre_process(content)

        for sen in re.split('[。，；：“”,]', content):
            sen = sen.strip()
            for word, attr in posseg.lcut(sen):
                if word in wordmap:
                    wordmap[word] += 1
                elif word not in wordmap and attr != 'nr'and attr != 'ns'and attr != 'nz':
                    wordmap[word] = 1
            # for word in jieba.lcut(sen):
            #     if word in wordmap:
            #         wordmap[word] += 1
            #     else:
            #         wordmap[word] = 1

    total = len(wordmap)
    for i, key in enumerate(sorted(wordmap, key=lambda k: wordmap[k], reverse=True)):
        print('当前{}, 总计{}'.format(i, total))
        if key.strip() == '':
            continue
        pyDBC.save(table='tool_word_embadding', rows={'number': str(i), 'keyWord': key, 'rateWord': str(wordmap[key])})


def map_word_IdNo():
    datalist = pyDBC.get_all(table='tool_word_embadding', columns=['id', 'keyWord'])
    word_map = {}
    for data in datalist:
        id_ = data[0]
        word = data[1]

        if word not in word_map:
            word_map[word] = id_
    # print(word_map)
    return word_map


def pre_word_2_vec_wordlist_window():
    datalist = pyDBC.get_all(table='t_case', columns=['id', 'content'])
    map_word_id = map_word_IdNo()
    word_x_y = []
    for data in tqdm(datalist):
        id_ = data[0]
        content = data[1]
        content = pre_process(content)

        for sen in re.split('[。，；：“”,]', content):
            sen = sen.strip()
            if sen == '':
                continue
            wordlcut = jieba.lcut(sen)
            for index, word in enumerate(wordlcut):
                for size in range(single_window_size):
                    # to left
                    if index >= size+1:
                        if word in map_word_id and wordlcut[index - size - 1] in map_word_id:
                            word_x_y.append([map_word_id[word], map_word_id[wordlcut[index - size - 1]]])
                    if index < len(wordlcut)-size-1:
                        if word in map_word_id and wordlcut[index + size + 1] in map_word_id:
                            word_x_y.append([map_word_id[word], map_word_id[wordlcut[index + size + 1]]])
    return word_x_y

batch_position = 0
#TODO:
word_x_y_list = pre_word_2_vec_wordlist_window()
def get_batch(batch_size):
    global batch_position
    lenght = len(word_x_y_list)
    inputs = []
    labels = []
    for _ in range(batch_size):
        x_y = word_x_y_list[random.randint(0, lenght-1)]
        inputs.append(x_y[0])
        labels.append([x_y[1]])
        batch_position += 1
        if batch_position >= lenght:
            batch_position = 0
    return inputs, labels


def generate_batch(loop_size, batch_size):
    global batch_position
    batches = []
    for _ in loop_size:
        batches.append(get_batch(batch_size))
    return batches


def word2vec():
    vocabulary_size = 41833 # word amount
    embedding_size = 300 # word dimension

    # 建立输入占位符
    train_inputs = tf.placeholder(tf.int32, shape=[None], name='train_inputs')
    train_labels = tf.placeholder(tf.int32, shape=[None, 1], name='train_labels')
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)), name='nce_weights')
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='nce_biases')
    embed = tf.nn.embedding_lookup(embeddings, train_inputs, name='embed')

    tf.add_to_collection('train_inputs', train_inputs)
    tf.add_to_collection('train_labels', train_labels)
    tf.add_to_collection('embeddings', embeddings)
    tf.add_to_collection('nce_weights', nce_weights)
    tf.add_to_collection('nce_biases', nce_biases)
    tf.add_to_collection('embed', embed)

    return train_inputs, train_labels, nce_weights, nce_biases, embed, vocabulary_size


# 训练
def train():
    batch_size = 500
    num_sampled = 64 # 负样本数量
    # 计算 NCE 损失函数, 每次使用负标签的样本.
    train_inputs, train_labels, nce_weights, nce_biases, embed, vocabulary_size = word2vec()
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled, vocabulary_size))
    # 使用 SGD 控制器.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
    saver = tf.train.Saver()
    tf.add_to_collection('embed', embed)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loop_no = -1
        while True:
            inputs, labels = get_batch(batch_size)
            feed_dict = {train_inputs: inputs, train_labels: labels}
            try:
                _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
            except BaseException:
                continue
            loop_no += 1
            if loop_no % 1000 == 0:
                save_path = saver.save(sess, "save/model.ckpt", global_step=loop_no)
                print('当前loop：{}, 精度：{}, 保存位置：{}'.format(loop_no, cur_loss, save_path))


def check_model():
    ckpt = tf.train.get_checkpoint_state("./save")  # 确定最新的参数文件

    with tf.Session() as sess:
        inputs, labels = get_batch(1)
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        embed = tf.get_collection('embed')[0]
        train_inputs = tf.get_collection('train_inputs')[0]
        result = sess.run(embed, feed_dict={train_inputs: inputs})
        print('result: {}, label:{}'.format(result, labels[0]))

def load_tensorboard():
    ckpt = tf.train.get_checkpoint_state("./save")  # 确定最新的参数文件

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        embed = tf.get_collection('embed')[0]

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs", sess.graph)
        writer.add_summary(embed)
        writer.close()



if __name__ == '__main__':
    # statistics_word()
    # map_word_IdNo()
    # pre_word_2_vec_wordlist_window()
    # print('/'.join(jieba.cut('每周三小时', cut_all=False)))
    # fun()
    # get_batch(500)
    train()
    # check_model()
    # load_tensorboard()

pyDBC.close()
