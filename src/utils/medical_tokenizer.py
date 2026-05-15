#!/usr/bin/env python
"""
优化后的中文医学分词器
"""
import os
import jieba
import jieba.posseg as pseg

# 加载医学领域词典
MEDICAL_DICT = "./data/medical_dict.txt"

def init_medical_tokenizer():
    """初始化医学领域分词器"""
    # 添加医学词典
    if os.path.exists(MEDICAL_DICT):
        jieba.load_userdict(MEDICAL_DICT)
        print("已加载医学词典")
    
    # 添加常见医学术语
    medical_terms = [
        "高血压", "低血压", "冠心病", "糖尿病", "支气管炎",
        "布洛芬", "阿司匹林", "阿莫西林", "头孢类", "抗生素",
        "CT扫描", "MRI检查", "血常规", "心电图", "B超",
        "上呼吸道感染", "下呼吸道感染", "病毒性感冒", "细菌性肺炎"
    ]
    
    for term in medical_terms:
        jieba.add_word(term)
    
    print("医学分词器初始化完成")

def medical_tokenize(text: str) -> list:
    """医学文本分词（保留医学术语）"""
    words = []
    
    # 使用词性标注
    for word, flag in pseg.cut(text):
        # 过滤停用词但保留医学相关词
        if word.strip() and len(word) > 1:
            words.append(word)
    
    return words

# 创建医学词典文件
if __name__ == "__main__":
    medical_terms = """
高血压
低血压
冠心病
糖尿病
支气管炎
布洛芬
阿司匹林
阿莫西林
头孢类
抗生素
CT扫描
MRI检查
血常规
心电图
B超
上呼吸道感染
下呼吸道感染
病毒性感冒
细菌性肺炎
心肌梗死
脑血管疾病
消化系统疾病
呼吸系统疾病
内分泌系统疾病
免疫系统疾病
神经系统疾病
心血管系统疾病
泌尿系统疾病
骨骼系统疾病
皮肤疾病
眼科疾病
耳鼻喉疾病
口腔科疾病
妇产科疾病
儿科疾病
急诊科疾病
重症医学
康复医学
老年医学
中医学
西医学
中西医结合
药物治疗
手术治疗
物理治疗
康复治疗
放射治疗
化学治疗
免疫治疗
基因治疗
靶向治疗
临床试验
医学研究
医学教育
医学伦理
医疗管理
医疗服务
医疗保险
医疗改革
"""
    
    os.makedirs("./data", exist_ok=True)
    with open("./data/medical_dict.txt", "w", encoding="utf-8") as f:
        f.write(medical_terms.strip())
    print("医学词典文件已创建")