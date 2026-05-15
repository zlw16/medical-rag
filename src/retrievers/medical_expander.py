"""
medical_expander.py - 医学查询扩展器
"""
from typing import List, Dict
import re
from src.logger import logger

class MedicalQueryExpander:
    """医学查询扩展器（同义词扩展 + 缩写展开）"""
    
    def __init__(self):
        # 医学同义词词典
        self.synonyms = self._load_synonyms()
        
        # 医学缩写词典
        self.abbreviations = {
            "CT": "计算机断层扫描",
            "MRI": "核磁共振成像",
            "B超": "B型超声波检查",
            "ECG": "心电图",
            "EKG": "心电图",
            "血常规": "血液常规检查",
            "尿常规": "尿液常规检查",
            "生化": "生物化学检查",
            "X光": "X射线检查",
            "PET": "正电子发射计算机断层显像",
            "ICU": "重症监护室",
            "ER": "急诊科",
            "BP": "血压",
            "HR": "心率",
            "RR": "呼吸频率",
            "T": "体温",
            "SPO2": "血氧饱和度",
            "BMI": "身体质量指数",
            "IV": "静脉注射",
            "PO": "口服",
            "IM": "肌肉注射",
            "QD": "每日一次",
            "BID": "每日两次",
            "TID": "每日三次",
            "QID": "每日四次",
            "PRN": "必要时"
        }
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """加载医学同义词词典"""
        return {
            "感冒": ["伤风", "上感", "上呼吸道感染", "普通感冒", "cold"],
            "高血压": ["血压高", "hypertension", "high blood pressure"],
            "糖尿病": ["血糖高", "diabetes", "diabetes mellitus"],
            "饮食": ["膳食", "营养", "食物", "diet", "nutrition", "进食", "饮食控制"],
            "营养": ["饮食", "膳食", "nutrition"],
            "发烧": ["发热", "高烧", "体温升高", "fever"],
            "咳嗽": ["干咳", "湿咳", "cough"],
            "头痛": ["头疼", "headache"],
            "头晕": ["眩晕", "dizziness"],
            "呕吐": ["恶心", "vomiting"],
            "腹泻": ["拉肚子", "diarrhea"],
            "胃痛": ["腹痛", "stomachache"],
            "心脏病": ["心血管疾病", "heart disease"],
            "肺炎": ["肺部感染", "pneumonia"],
            "肝炎": ["肝脏炎症", "hepatitis"],
            "肾炎": ["肾脏炎症", "nephritis"],
            "关节炎": ["关节炎症", "arthritis"],
            "贫血": ["血虚", "anemia"],
            "过敏": ["超敏反应", "allergy"],
            "感染": ["传染", "infection"],
            "炎症": ["发炎", "inflammation"],
            "疼痛": ["痛", "pain"],
            "手术": ["开刀", "operation", "surgery"],
            "药物": ["药品", "medicine", "drug"],
            "治疗": ["医治", "treatment"],
            "预防": ["防护", "prevention"],
            "诊断": ["确诊", "diagnosis"],
            "检查": ["检验", "检测", "examination"],
            "疫苗": ["预防针", "vaccine"],
            "抗生素": ["抗菌素", "antibiotic"],
            "抗病毒": ["antiviral"],
            "布洛芬": ["ibuprofen", "芬必得", "美林"],
            "阿司匹林": ["aspirin", "乙酰水杨酸"],
            "阿莫西林": ["amoxicillin"],
            "头孢": ["cephalosporin"],
            "激素": ["荷尔蒙", "hormone"],
            "维生素": ["维他命", "vitamin"],
            "输液": ["打点滴", "静脉输液"],
            "输血": ["blood transfusion"],
            "化疗": ["化学治疗", "chemotherapy"],
            "放疗": ["放射治疗", "radiotherapy"],
            "理疗": ["物理治疗", "physical therapy"],
            "康复": ["恢复", "rehabilitation"],
            "住院": ["入院", "hospitalization"],
            "出院": ["离院", "discharge"],
            "门诊": ["诊所", "outpatient"],
            "急诊": ["emergency"],
            "挂号": ["预约", "registration"],
            "病历": ["病史", "medical record"],
            "医嘱": ["医生嘱咐", "medical advice"],
            "处方": ["药方", "prescription"],
            "剂量": ["用量", "dosage"],
            "疗程": ["治疗周期", "course of treatment"],
            "副作用": ["不良反应", "side effect"],
            "禁忌": ["禁止", "contraindication"],
            "过敏史": ["过敏记录", "allergy history"],
            "家族史": ["家族遗传史", "family history"],
            "吸烟史": ["吸烟记录", "smoking history"],
            "饮酒史": ["饮酒记录", "drinking history"]
        }
    
    def expand_query(self, query: str) -> str:
        """扩展查询（同义词 + 缩写展开）"""
        expanded_terms = set()
        expanded_terms.add(query)
        
        # 缩写展开
        for abbr, full in self.abbreviations.items():
            if abbr in query:
                expanded_terms.add(full)
                expanded_terms.add(abbr)
        
        # 同义词扩展
        for term, syns in self.synonyms.items():
            if term in query:
                for syn in syns:
                    expanded_terms.add(syn)
                expanded_terms.add(term)
        
        # 生成扩展查询（使用OR连接）
        if len(expanded_terms) > 1:
            expanded_query = " OR ".join(expanded_terms)
            logger.info(f"查询扩展: {query} -> {expanded_query}")
            return expanded_query
        
        return query
    
    def rewrite_query(self, query: str) -> List[str]:
        """生成多种查询改写形式"""
        rewrites = []
        
        # 原始查询
        rewrites.append(query)
        
        # 添加疑问词变体
        question_patterns = [
            "{}是什么",
            "{}怎么办",
            "{}的症状",
            "{}的治疗方法",
            "{}的注意事项",
            "{}的原因",
            "{}能治好吗",
            "{}严重吗"
        ]
        
        # 提取核心术语
        core_terms = self._extract_core_terms(query)
        
        for term in core_terms:
            for pattern in question_patterns:
                rewrites.append(pattern.format(term))
        
        return list(set(rewrites))
    
    def _extract_core_terms(self, query: str) -> List[str]:
        """提取查询中的核心医学术语"""
        terms = []
        for syn_group in self.synonyms.keys():
            if syn_group in query:
                terms.append(syn_group)
        return terms

# 测试
if __name__ == "__main__":
    expander = MedicalQueryExpander()
    test_queries = ["高血压需要注意什么", "发烧怎么办", "CT检查是什么"]
    
    for q in test_queries:
        expanded = expander.expand_query(q)
        rewrites = expander.rewrite_query(q)
        print(f"原始查询: {q}")
        print(f"扩展查询: {expanded}")
        print(f"改写形式: {rewrites[:3]}")
        print()
