from fastNLP.core.predictor import Predictor

# 实例化model之后，load之前保存的权重文件
model_path = 'your_path/flat_lattice/weibo/best_MECTNER_f_2023-03-14-06-22-54'
states = torch.load(model_path).state_dict()
model.load_state_dict(states)

# 在fastNLP中实际上是定义了用于预测的类的，名为predictor
predictor = Predictor(model)  # 这里的model是加载权重之后的model

test_label_list = predictor.predict(datasets['test'][:1])['pred'][0]  # 预测结果
test_raw_char = datasets['test'][:1]['raw_chars'][0]  # 原始文字

# test_label_list就在test上预测出来的label，label对应的BIO可以通过以下代码查看
for d in vocabs['label']:
    print(d)


# 把label转换成实体（仅适用于MSRA数据集），
def recognize(label_list, raw_chars):
    """
    根据模型预测的label_list，找出其中的实体
    label_lsit: array
    raw_chars: list of raw_char
    return: entity_list: list of tuple(ent_text, ent_type)
    -------------
    ver: 20210303
    by: changhongyu
    """
    if len(label_list.shape) == 2:
        label_list = label_list[0]
    elif len(label_list) > 2:
        raise ValueError('please check the shape of input')

    assert len(label_list.shape) == 1
    assert len(label_list) == len(raw_chars)

    # 其实没有必要写这个
    # 但是为了将来可能适应bio的标注模式还是把它放在这里了
    starting_per = False
    starting_loc = False
    starting_org = False
    ent_type = None
    ent_text = ''
    entity_list = []

    for i, label in enumerate(label_list):
        if label in [0, 1, 2]:
            ent_text = ''
            ent_type = None
            continue
        # begin
        elif label == 10:
            ent_type = 'PER'
            starting_per = True
            ent_text += raw_chars[i]
        elif label == 4:
            ent_type = 'LOC'
            starting_loc = True
            ent_text += raw_chars[i]
        elif label == 6:
            ent_type = 'ORG'
            starting_org = True
            ent_text += raw_chars[i]
        # inside
        elif label == 9:
            if starting_per:
                ent_text += raw_chars[i]
        elif label == 8:
            if starting_loc:
                ent_text += raw_chars[i]
        elif label == 3:
            if starting_org:
                ent_text += raw_chars[i]
        # end
        elif label == 11:
            if starting_per:
                ent_text += raw_chars[i]
                starting_per = False
        elif label == 5:
            if starting_loc:
                ent_text += raw_chars[i]
                starting_loc = False
        elif label == 7:
            if starting_org:
                ent_text += raw_chars[i]
                starting_org = False
        elif label == 13:
            ent_type = 'PER'
            ent_text = raw_chars[i]
        elif label == 12:
            ent_type = 'LOC'
            ent_text = raw_chars[i]
        elif label == 14:
            ent_type = 'PER'
            ent_text = raw_chars[i]
        else:
            ent_text = ''
            ent_type = None
            continue

        if not (starting_per or starting_loc or starting_org) and len(ent_text):
            # 判断实体已经结束，并且提取到的实体有内容
            entity_list.append((ent_text, ent_type))

    return entity_list

# recognize(test_label_list, test_raw_char)
# Out：
# [('中共中央', 'ORG'),
#  ('中国致公党', 'ORG'),
#  ('中国致公党', 'ORG'),
#  ('中国共产党中央委员会', 'ORG'),
#  ('致公党', 'ORG')]
