@echo off
echo === 医疗RAG系统安装脚本 ===

echo 1. 创建虚拟环境...
python -m venv venv

echo 2. 激活虚拟环境...
call venv\Scripts\activate.bat

echo 3. 安装依赖包...
pip install -r requirements.txt

echo 4. 创建知识库文件夹...
mkdir medical_knowledge

echo 5. 创建示例文档...
echo 【疾病】普通感冒 > medical_knowledge\示例文档.txt
echo. >> medical_knowledge\示例文档.txt
echo 【症状】鼻塞、流涕、咳嗽、咽痛 >> medical_knowledge\示例文档.txt
echo. >> medical_knowledge\示例文档.txt
echo 【治疗】多休息、多喝水、发热可用退烧药 >> medical_knowledge\示例文档.txt
echo. >> medical_knowledge\示例文档.txt
echo 【就医】症状持续超过3天需就医 >> medical_knowledge\示例文档.txt

echo.
echo ✅ 安装完成！
echo 运行命令: python main.py
pause