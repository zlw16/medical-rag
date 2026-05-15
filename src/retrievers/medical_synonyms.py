"""
medical_synonyms.py - 医疗同义词扩展模块
为小数据集提供术语扩展能力
"""

from typing import List, Dict, Set
from src.logger import logger


class MedicalSynonymExpander:
    """医疗术语同义词扩展器"""

    def __init__(self):
        # 医疗术语同义词词典
        self.synonym_dict = self._load_synonyms()

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """加载医疗同义词词典"""
        return {
            # 常见疾病
            "感冒": ["伤风", "上感", "上呼吸道感染", "普通感冒", "cold"],
            "发烧": ["发热", "高烧", "体温升高", "fever"],
            "咳嗽": ["干咳", "湿咳", "cough"],
            "头痛": ["头疼", "headache"],
            "头晕": ["眩晕", "头昏", "dizziness"],
            "恶心": ["想吐", "反胃"],
            "呕吐": ["呕", "吐"],
            "腹泻": ["拉肚子", "泄泻", "diarrhea"],
            "便秘": ["大便不通", "constipation"],
            "腹痛": ["肚子痛", "abdominal pain"],
            "胃痛": ["胃绞痛", "gastralgia"],
            
            # 慢性病
            "高血压": ["血压高", "hypertension", "high blood pressure"],
            "糖尿病": ["消渴", "diabetes", "高血糖"],
            "血糖": ["blood glucose", "blood sugar", "血糖水平"],
            "高血脂": ["血脂高", "hyperlipidemia"],
            "冠心病": ["冠状动脉粥样硬化", "coronary heart disease"],
            "哮喘": ["气喘", "asthma"],
            "关节炎": ["关节痛", "arthritis"],

            # 饮食/营养
            "饮食": ["膳食", "营养", "食物", "diet", "nutrition", "eating", "进食", "饮食控制"],
            "营养": ["饮食", "膳食", "nutrition"],
            "碳水化合物": ["碳水", "carbohydrate", "糖类"],
            
            # 症状
            "疲劳": ["乏力", "疲倦", "fatigue"],
            "失眠": ["睡不着", "入睡困难", "insomnia"],
            "焦虑": ["紧张", "anxiety"],
            "抑郁": ["depression"],
            "心悸": ["心慌", "palpitation"],
            "胸闷": ["胸口发闷"],
            "呼吸困难": ["喘不过气", "dyspnea"],
            "鼻塞": ["鼻子堵", "nasal congestion"],
            "流涕": ["流鼻涕", "runny nose"],
            
            # 药物
            "布洛芬": ["ibuprofen", "芬必得", "美林"],
            "阿司匹林": ["aspirin", "乙酰水杨酸"],
            "阿莫西林": ["amoxicillin"],
            "头孢": ["cephalosporin"],
            "板蓝根": ["isatis root"],
            
            # 检查项目
            "血常规": ["血检", "blood test"],
            "尿常规": ["尿检", "urine test"],
            "心电图": ["ECG", "EKG", "electrocardiogram"],
            "CT": ["computed tomography"],
            "MRI": ["磁共振", "magnetic resonance imaging"],
            
            # 身体部位
            "心脏": ["heart"],
            "肝脏": ["liver"],
            "肾脏": ["kidney", "肾"],
            "肺": ["lungs"],
            "胃": ["stomach"],
            "肠": ["intestine"],
            "脑": ["brain"],
            
            # 治疗方式
            "手术": ["operation", "surgery"],
            "化疗": ["chemotherapy"],
            "放疗": ["radiotherapy", "radiation therapy"],
            "输液": ["打点滴", "intravenous infusion"],
            "打针": ["注射", "injection"],
            "理疗": ["物理治疗", "physical therapy"],
            
            # 中医术语
            "上火": ["热气", "内火"],
            "湿气": ["湿邪"],
            "气虚": ["元气不足"],
            "血虚": ["血不足"],
            "阴虚": ["阴液不足"],
            "阳虚": ["阳气不足"],
        }

    def expand_query(self, query: str) -> List[str]:
        """
        扩展查询词，添加同义词
        
        Args:
            query: 用户查询
        
        Returns:
            扩展后的查询列表（包含原始查询）
        """
        expanded = [query]
        added_terms: Set[str] = {query}

        # 分词处理
        import jieba
        words = list(jieba.cut(query))

        # 查找同义词
        for word in words:
            if word in self.synonym_dict:
                for synonym in self.synonym_dict[word]:
                    if synonym not in added_terms:
                        added_terms.add(synonym)
                        expanded.append(synonym)

        logger.info(f"查询扩展: {query} -> {expanded}")
        return expanded

    def expand_to_boolean_query(self, query: str) -> str:
        """
        扩展为布尔查询（用于BM25/TF-IDF）
        
        Args:
            query: 用户查询
        
        Returns:
            布尔查询字符串，如 "感冒 OR 伤风 OR 上感"
        """
        expanded = self.expand_query(query)
        return " OR ".join(expanded)

    def add_custom_synonym(self, term: str, synonyms: List[str]):
        """
        添加自定义同义词
        
        Args:
            term: 主术语
            synonyms: 同义词列表
        """
        if term not in self.synonym_dict:
            self.synonym_dict[term] = []
        
        for syn in synonyms:
            if syn not in self.synonym_dict[term]:
                self.synonym_dict[term].append(syn)
        
        logger.info(f"添加同义词: {term} -> {synonyms}")
