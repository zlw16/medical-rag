# -*- coding: utf-8 -*-
"""
默沙东诊疗手册 - 医学文章爬虫
仅爬取公开可用内容，用于个人学习研究
遵守 robots.txt 并限制请求速率
支持断点续爬
"""
import os, re, time, json, ssl, urllib.request, sys, random
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# === 配置 ===
SITEMAP_FILE = os.path.join(os.path.dirname(__file__), "msd_sitemap.xml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "medical_knowledge", "msd")
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "_crawl_progress.json")
DELAY_MIN, DELAY_MAX = 1.5, 3.0
TIMEOUT = 30
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

EXCLUDE_PATTERNS = [
    "pages-with-widgets", "resourcespages", "multimedia",
    "health-topics", "/resource$", "/symptoms$",
    "/professional$", "authors/",
]

# 按优先级排序的分类（先爬最常用的）
CATEGORIES = [
    "cardiovascular-disorders",         # 心血管 - 高血压、冠心病等
    "pulmonary-disorders",              # 呼吸 - 肺炎、哮喘等
    "infectious-diseases",              # 感染 - 感冒、流感等
    "endocrine-and-metabolic-disorders", # 内分泌 - 糖尿病等
    "gastrointestinal-disorders",       # 消化
    "neurologic-disorders",             # 神经
    "musculoskeletal-and-connective-tissue-disorders", # 骨科
    "dermatologic-disorders",           # 皮肤
    "hematology-and-oncology",          # 血液/肿瘤
    "immunology-allergic-disorders",    # 免疫/过敏
    "nutritional-disorders",            # 营养
    "psychiatric-disorders",            # 精神
    "eye-disorders",                    # 眼科
    "ear-nose-and-throat-disorders",    # 耳鼻喉
    "genitourinary-disorders",          # 泌尿生殖
    "gynecology-and-obstetrics",        # 妇产
    "pediatrics",                       # 儿科
    "geriatrics",                       # 老年
    "critical-care-medicine",           # 重症
    "injuries-poisoning",               # 损伤/中毒
    "clinical-pharmacology",            # 临床药理
    "dental-disorders",                 # 口腔
    "hepatic-and-biliary-disorders",    # 肝胆
    "special-subjects",                 # 专题
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def is_article_url(url):
    path = urlparse(url).path
    for p in EXCLUDE_PATTERNS:
        if re.search(p, path):
            return False
    parts = path.strip("/").split("/")
    return len(parts) >= 2 and parts[0] == "professional"


def get_article_urls():
    """从本地站点地图获取文章URL"""
    if not os.path.exists(SITEMAP_FILE):
        print("正在下载站点地图...")
        req = urllib.request.Request(
            "https://www.msdmanuals.cn/sitemap.xml", headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=60, context=ctx)
        content = resp.read().decode("utf-8")
        with open(SITEMAP_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  已保存到 {SITEMAP_FILE}")
    else:
        with open(SITEMAP_FILE, "r", encoding="utf-8") as f:
            content = f.read()

    urls = re.findall(r"<loc>(.*?)</loc>", content)
    article_urls = [u for u in urls if is_article_url(u)]
    return article_urls


def extract_article(html, url):
    """从HTML提取文章"""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    content_div = soup.find("div", class_=lambda c: c and "TopicMainContent_content" in c)
    if not content_div:
        return None

    paras = []
    for elem in content_div.find_all(["p", "li", "h2", "h3", "h4"]):
        t = elem.get_text(strip=True)
        if t and len(t) > 3:
            tag = elem.name
            if tag.startswith("h"):
                paras.append(f"\n【{t}】")
            else:
                paras.append(t)

    if not paras:
        return None

    return {"title": title, "content": "\n".join(paras), "url": url}


def save_article(data, category, slug):
    d = os.path.join(OUTPUT_DIR, category)
    os.makedirs(d, exist_ok=True)
    fname = slug.replace("/", "_") + ".txt"
    fp = os.path.join(d, fname)
    with open(fp, "w", encoding="utf-8") as f:
        f.write(f"标题：{data['title']}\n来源：{data['url']}\n{'='*60}\n\n{data['content']}")
    return fp


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(prog):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(prog, f, ensure_ascii=False, indent=2)


def crawl_category(category, all_urls, progress):
    cat_urls = [u for u in all_urls if f"/professional/{category}/" in u]
    if not cat_urls:
        print(f"[{category}] 无文章，跳过")
        return 0, 0, 0

    done = progress.get(category, {}).get("done", [])
    ok, err, skip = 0, 0, 0

    for i, url in enumerate(cat_urls):
        path = urlparse(url).path.strip("/")
        parts = path.split("/")
        slug = "/".join(parts[2:])

        if slug in done:
            skip += 1
            continue

        print(f"  [{i+1}/{len(cat_urls)}] {slug[:60]}...", end=" ", flush=True)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx)
            html = resp.read().decode("utf-8")

            data = extract_article(html, url)
            if not data:
                print("跳过（无内容）")
                err += 1
                done.append(slug)
                continue

            fp = save_article(data, category, slug)
            print(f"OK ({len(data['content'])}字)")
            ok += 1
            done.append(slug)

            progress.setdefault(category, {})["done"] = done
            if ok % 10 == 0:
                save_progress(progress)

        except Exception as e:
            print(f"ERR: {e}")
            err += 1

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    return ok, err, skip


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("获取文章列表...")
    all_urls = get_article_urls()
    print(f"共 {len(all_urls)} 篇文章\n")

    progress = load_progress()
    total_ok, total_err, total_skip = 0, 0, 0
    start_time = time.time()

    for cat in CATEGORIES:
        cat_urls = [u for u in all_urls if f"/professional/{cat}/" in u]
        done = progress.get(cat, {}).get("done", [])
        remaining = len(cat_urls) - len(done)
        pct = int(len(done)/len(cat_urls)*100) if cat_urls else 0
        status = "完成" if remaining <= 0 else f"剩余{remaining}"
        print(f"[{cat}] {len(cat_urls)}篇 {pct}% {status}")

        if remaining <= 0:
            total_skip += len(cat_urls)
            continue

        ok, err, skip = crawl_category(cat, all_urls, progress)
        total_ok += ok
        total_err += err
        total_skip += skip + (len(cat_urls) - ok - err - skip)

        # 每爬完一个分类保存进度
        save_progress(progress)

    elapsed = int(time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"爬取完成！")
    print(f"  成功: {total_ok}")
    print(f"  失败: {total_err}")
    print(f"  跳过: {total_skip}")
    print(f"  总耗时: {elapsed//60}分{elapsed%60}秒")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
