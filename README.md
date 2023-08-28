# Social-media-data-NER
使用FLAT框架结合字向量和 ctb6.0词典，结合word2vec自建领域词典，基于CNN模型提取汉字的字形特点，采用了BMES进行数据 的标注，建立了抽取模型。

需要下载预训练的字符嵌入和单词嵌入，并将它们放在数据文件夹中。

Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): Google Drive or Baidu Pan

Bi-gram embeddings (gigaword_chn.all.a2b.bi.ite50.vec): Baidu Pan

Word(Lattice) embeddings (ctb.50d.vec): Baidu Pan

获得汉字结构组件(自由基)。论文中使用的偏旁部首来自新华在线词典。由于版权原因，这些数据无法发布。有一个方法可以取代漢語拆字字典,但不一致的字符分解方法不能保证可重复性。

运行：

python Utils/preprocess.py

python main.py --dataset weibo
