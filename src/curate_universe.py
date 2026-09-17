"""按类别策展 universe: 宽基 + 风格 + 一套行业, 砍掉地域与主题。

用户规则(2026-09-17): 地域和主题全砍, 宽基和风格全留, 行业只留一套。
原话是"只留申万一套", 但本库名称含「申万」的只有 3 个行业指数(证券/传媒/电子),
不是一套分类 —— 申万一级有 31 个行业, 库里没有。故按"一套就行"的意图,
默认取 **深证一级行业 10 个**(完整 10 分类, 2014-01-02 起)。
换族改 INDUSTRY_FAMILY 即可。

为什么要砍: 480 个指数的有效独立维数只有 3.1(参与率), 63 对相关 >0.995 是字面重复
(沪深双挂 / R 全收益版)。不砍的话"在 480 个指数上成立"这句话会把同一个市场因子的
480 个切片说成 480 次独立验证。

用法: python curate_universe.py [--family 深证|380|1000|深成]
输出: out/index_universe_curated.json + out/universe_curated_list.csv
"""
import os as _os; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # 仓库根
import argparse, json, re
import pandas as pd

OUT = f'{_R}/out'

# ---- 行业族(完整 10 分类的候选) ----
SECTORS = ['能源', '材料', '工业', '可选', '消费', '医药', '金融', '信息', '电信', '公用']
FAMILY_PAT = {
    '深证': lambda n: n.startswith('深证') and n[2:] in SECTORS,
    '深成': lambda n: n.startswith('深成') and '行业' in n,
    '380': lambda n: re.fullmatch(r'380(' + '|'.join(SECTORS) + ')', n) is not None,
    '1000': lambda n: re.fullmatch(r'1000(' + '|'.join(SECTORS) + ')', n) is not None,
}

# ---- 地域: 全砍 ----
GEO = ['长三角', '珠三角', '环渤海', '皖江', '苏州', '泰达', '中关村', '深报', '沪财',
       '小康', '一带一路', '新丝路', '南方', '央视', 'OCT', 'CBN', '海外', '沪股通',
       '地企', '国企', '沪企', '民企', '央企', '上国', '上央', '上民']

# ---- 行业关键词: 命中即判「行业」, 只有选定族内的才留, 其余全砍 ----
# ★必须先于风格/宽基判定 —— 否则 380医药 / 医药等权 / 巨潮地产 这类会漏进宽基或风格,
#   「行业只留一套」就没做到(第一版正是这么错的: 宽基里混进了 6 个族的行业指数)。
INDUSTRY = SECTORS + [
    '地产', '银行', '证券', '保险', '非银', '有色', '煤炭', '钢铁', '黑色', '石油',
    '天然气', '电力', '汽车', '传媒', '电子', '军工', '国防', '农业', '农林', '农牧',
    '食品', '白酒', '酒指数', '物流', '交通', '运输', '建筑', '基建', '化工', '生物',
    '医疗', '软件', '互联网', '通信', '计算机', 'TMT', 'IT指数', '装备', '制造',
    '商业', '服务', '资源', '大宗', '采矿', '水电', '批零', '餐饮', '商务', '科研',
    '公共', '综企', '环保', '新能', '节能', '安防', '文化', '金融地产', '主要消费',
    '可选消费', '原材料', '公用事业', '信息技术', '医药卫生', '电信业务', '细分医药',
    '上游', '中游', '下游', '投资品', '消费品', '周期', '高端', '原料', '商品',
]

# ---- 主题: 全砍(概念/事件/策略标签) ----
THEME = ['区块链', 'AI', '机器人', '物联网', '大数据', '智能家居', '信息安全', '移动互联',
         '生态', '高铁', '定向增发', '并购重组', '国有企业改革', '次新股', '转债', '交债',
         '创投', '金融科技', '腾安', '百度百发', '分析师', '战略新兴', '新兴', '持续产业',
         '国家安全', '一带一路',
         '优势', '主题', '领先', '率先', '龙头', '科技', '创新', '精选', '引擎',
         '时钟', 'GDP', '绩效', '治理', '责任', 'ETF', '基金', '乐富', '转债']

# ---- 风格: 全留 ----
STYLE = ['成长', '价值', '创业板G', '创业板V', '高贝', '低贝', '高波', '低波', '波动', '等权', 'EW', '基本',
         '稳定', '动态', '防御', '大盘', '中盘', '小盘', '巨潮', '分层', '红利', '超大盘']


def classify(code, name, ind_keep):
    n = str(name)
    if code in ind_keep:
        return '行业(保留族)'
    if any(k in n for k in INDUSTRY):        # ★先判行业
        return '行业(其他族)'
    if any(k in n for k in GEO):
        return '地域'
    if any(k in n for k in THEME):
        return '主题'
    if any(k in n for k in STYLE):
        return '风格'
    return '宽基'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='深证', choices=list(FAMILY_PAT))
    a = ap.parse_args()
    d = pd.read_csv(f'{OUT}/universe_480_list.csv')
    d['名称'] = d['名称'].astype(str)

    f = FAMILY_PAT[a.family]
    ind = d[d['名称'].map(f)]
    ind_keep = set(ind['代码'])
    print(f'行业族 = {a.family}, {len(ind_keep)} 个: ' + ', '.join(ind['名称']))

    d['类'] = [classify(c, n, ind_keep) for c, n in zip(d['代码'], d['名称'])]
    # 规则: 宽基 + 风格 + 选定行业族 全留; 地域/主题/其他行业族 全砍
    d['策展保留'] = d['类'].isin(['宽基', '风格', '行业(保留族)'])
    # 再叠加去重(相关>0.99 的簇只留一个) —— 用已有的 保留 列
    d['策展保留'] = d['策展保留'] & d['保留']

    print()
    print(d.groupby(['类', '策展保留']).size().unstack(fill_value=0).to_string())
    keep = d[d['策展保留']]['代码'].tolist()
    print(f'\n最终 {len(keep)} 个 (原 {len(d)})')

    d.to_csv(f'{OUT}/universe_curated_list.csv', index=False)
    json.dump({'rule': '地域+主题全砍; 宽基+风格全留; 行业只留一族(%s); 再去重 corr>0.99' % a.family,
               'family': a.family, 'codes': keep},
              open(f'{OUT}/index_universe_curated.json', 'w'), indent=1)
    print(f'写出 {OUT}/index_universe_curated.json 与 universe_curated_list.csv')

    for c in ['宽基', '风格', '行业(保留族)']:
        x = d[(d['类'] == c) & d['策展保留']]
        print(f'\n=== {c} ({len(x)}) ===')
        for i in range(0, len(x), 7):
            print('  ' + '  '.join(f'{r.名称[:9]:<10s}' for r in x.iloc[i:i + 7].itertuples()))


if __name__ == '__main__':
    main()
