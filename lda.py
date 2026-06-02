# encoding=utf-8

import re
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ==========================================
# 1. 准备语料库 (Corpus)
# ==========================================
docs = [
    "这家餐厅味道很好，主打川菜和火锅，环境也不错",
    "昨天去了新开的咖啡馆，拿铁顺滑，甜点也很惊喜",
    "球队本赛季状态火热，主教练强调高位逼抢与传控",
    "人工智能正在改变医疗影像分析与药物发现的流程",
    "新出的手机拍照很强，夜景模式和人像模式都清晰",
    "我在健身房练器械和有氧，主要是减脂和增肌计划",
    "电影节连续放映多部文艺片，摄影与配乐都很出色",
    "电商平台的物流很快，售后客服也专业，购物体验好",
    "研究者提出新的预训练模型框架，用于文本生成任务",
    "这家面馆的牛肉面和小菜分量足，价格也很实惠",
]

# ==========================================
# 2. 超参数配置 (Hyperparameters)
# ==========================================
N_TOPICS = 8           # 假设我们要把这10句话聚成4个主题 (聚类簇数)
TOPN_WORDS = 10        # 每个主题提取权重排名前10的代表性词汇
NGRAM_RANGE = (1, 1)   # 词袋模型参数：只考虑单个词 (Unigram)，不考虑前后词的组合
MIN_HITS = 2           # 规则打标阈值：当某个主题的Top词与人工词典命中至少2个时，才赋予该名称

# 自定义停用词表：去除对分辨主题没有帮助的常见词、虚词、代词
STOPWORDS = """
的 了 在 和 与 也 都 很 比较 非常 可能 我们 他们 正在 新的 新出 新开 本赛季 这家 这个
不错 主要 计划 连续 多部 环境 模式 清晰 专业 平台 体验
""".split()
stop = set(STOPWORDS) # 转为集合，提升后续查找匹配的运行速度

# ==========================================
# 3. 自定义词典 (Domain Dictionary)
# 作用：防止 jieba 把专有名词切碎。例如把"咖啡馆"切成"咖啡"和"馆"
# ==========================================
for w in ["咖啡馆", "拿铁", "甜点"]:
    jieba.add_word(w)

# ==========================================
# 4. 文本预处理与分词函数
# ==========================================
def tok(s: str):
    # 正则清洗：只保留中文(\u4e00-\u9fa5)、英文字母(A-Za-z)和数字(0-9)，其余符号(标点、表情包等)全部替换为空格
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", " ", s)
    # 使用 jieba 进行精确模式分词
    seg = jieba.lcut(s)
    # 列表推导式过滤：
    # 1. len(w) > 1: 过滤掉单字（通常单字没有实际主题意义）
    # 2. not w.isdigit(): 过滤纯数字
    # 3. w not in stop: 过滤停用词
    return [w for w in seg if len(w) > 1 and (not w.isdigit()) and (w not in stop)]

# ==========================================
# 5. 文本向量化 (Text Vectorization)
# 作用：把文本转换为计算机能看懂的数学矩阵 (词频矩阵)
# ==========================================
vec = CountVectorizer(
    tokenizer=tok,         # 使用上面自定义的分词与清洗函数
    token_pattern=None,    # 屏蔽默认的正则分词，强制使用我们的 tokenizer
    ngram_range=NGRAM_RANGE,
    max_df=0.95,           # 核心参数：如果一个词在 95% 以上的句子中都出现了，说明它太普遍（比如“的”），直接忽略
    min_df=1               # 核心参数：词语出现的最小频率，由于我们数据少，设为1；在大数据集中通常设为5或10过滤生僻词
)
X = vec.fit_transform(docs)                # X 形状为 (文档数, 词表总词数)，值为词频
terms = np.array(vec.get_feature_names_out()) # terms 保存了矩阵每一列对应的真实词语名称

# ==========================================
# 6. LDA 主题模型训练 (Latent Dirichlet Allocation)
# LDA的核心思想：一篇文章可以由多个主题按比例混合而成，一个主题可以由多个词语按概率分布组成。
# ==========================================
lda = LatentDirichletAllocation(
    n_components=N_TOPICS, # 聚成几类
    random_state=42,       # 固定随机种子，保证每次运行结果一致
    max_iter=200,          # EM算法的最大迭代次数
    # 以下两个参数是 LDA 的核心贝叶斯先验参数：
    doc_topic_prior=0.1,   # 即 alpha。越小，代表一篇文章只会集中属于极少数几个主题（即主题分布越稀疏）
    topic_word_prior=0.01  # 即 beta。 越小，代表一个主题只会被极少数几个具体的词代表（即词分布越稀疏）
)
# doc_topic 保存的是：每一句话 属于 4个主题 的概率分布，形状 (10, 4)
doc_topic = lda.fit_transform(X)
# comp 保存的是：4个主题 中，每个主题下 词汇表里每个词的权重分布，形状 (4, 词表大小)
comp = lda.components_

# ==========================================
# 7. 提取每个主题的 Top N 核心词
# ==========================================
def top_terms(C, vocab, topn=10):
    out = []
    # 遍历每一个主题 (k)
    for k in range(C.shape[0]):
        # argsort 默认是从小到大排序，[::-1] 将其翻转为从大到小，再取前 topn 个索引
        idx = np.argsort(C[k])[::-1][:topn]
        # 根据索引，去词表(vocab)里把具体的汉字词语揪出来
        out.append(vocab[idx].tolist())
    return out

topic_words = top_terms(comp, terms, topn=TOPN_WORDS)

# ==========================================
# 8. 规则引擎：给主题赋予人类能看懂的命名
# 因为 LDA 跑出来的结果只有“主题0”、“主题1”，人类看不懂，
# 所以通过人工预设一些“种子词库”，看看模型算出来的 Top 词与哪个词库重合度最高。
# ==========================================
LABEL_SEEDS = {
    "餐饮美食": {"火锅","餐厅","面馆","牛肉面","川菜","味道","实惠"},
    "咖啡甜品": {"咖啡馆","拿铁","甜点"},
    "体育赛事": {"球队","主教练","传控","逼抢","赛季","状态"},
    "影视娱乐": {"电影节","文艺片","摄影","配乐","放映"},
    "电商服务": {"电商","物流","售后","客服","购物","体验","平台"},
    "数码影像": {"拍照","夜景","人像","手机","清晰"},
    "AI科技":   {"人工智能","预训练","模型","文本生成","医疗影像","药物发现","框架"},
    "健身健康": {"健身房","器械","有氧","减脂","增肌"}
}

def name_topic(words, min_hits=MIN_HITS):
    s = set(words) # 把当前主题算出来的 top 词转为集合
    best_label, best_hits = "未命名", 0
    
    # 遍历我们的人工词库字典
    for label, keyset in LABEL_SEEDS.items():
        hits = len(s & keyset) # 利用集合的交集操作 (&)，计算命中了几个词
        if hits > best_hits:
            best_label, best_hits = label, hits
            
    # 如果最高命中数超过了设置的阈值 (MIN_HITS=2)，就用这个人工标签，否则标记为"未命名"
    return best_label if best_hits >= min_hits else "未命名"

# 为模型跑出的 4 个主题依次打上中文标签
topic_names = [name_topic(ws) for ws in topic_words]

# ==========================================
# 9. 整理文档归属 (将概率转换为具体结论)
# ==========================================
# 找出每句话概率最大的那个主题的索引 (argmax)
top_idx  = doc_topic.argmax(1)
# 找出每句话归属于该主题的具体概率值 (max)
top_prob = doc_topic.max(1)

rows = []
for i, txt in enumerate(docs):
    k = top_idx[i]
    rows.append({
        "原句": txt,
        "主题编号": int(k),
        "主题名称": topic_names[k],
        "该主题概率": float(round(top_prob[i], 3)), # 保留3位小数
        "该主题Top词": ", ".join(topic_words[k][:8])
    })
df = pd.DataFrame(rows)

# ==========================================
# 10. 优化 Pandas 的输出格式 (防止终端显示折叠或截断)
# ==========================================
pd.set_option('display.max_rows', 200)       # 最大显示200行
pd.set_option('display.max_columns', 200)    # 最大显示200列
pd.set_option('display.width', 2000)         # 控制输出宽度，避免换行
pd.set_option('display.max_colwidth', None)  # 列的内容如果不设 None 会被...截断
pd.set_option('display.unicode.east_asian_width', True) # 修复中文字符对齐问题

# ==========================================
# 11. 打印最终的可视化报告
# ==========================================
topic_overview = pd.DataFrame({
    "主题编号": list(range(N_TOPICS)),
    "主题名称": topic_names,
    "Top词（按权重降序）": [", ".join(ws) for ws in topic_words]
})
print("\n================ 主题总览 ================\n")
print(topic_overview.to_string(index=False))

print("\n================ 文档归属（逐行） ================\n")
for i, row in df.iterrows():
    print(f"[Doc{i}] 原句：{row['原句']}")
    print(f"       主题：{row['主题编号']} | {row['主题名称']} | 概率：{row['该主题概率']}")
    print(f"       Top词：{row['该主题Top词']}\n")

print("\n================ 文档归属（整表） ================\n")
print(df.to_string(index=False))