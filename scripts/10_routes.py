"""为漫游路线的每个站点检索最贴合的诗（Qwen3-Embedding-0.6B，与 02_embed.py 同一模型）。

流程：每条路线若干站点，每站一句描述 -> encode 成 query 向量 ->
与全库诗向量算余弦，取候选（作者去重、名家优先），打印出来供人工挑选。

用法: /path/to/venv/bin/python scripts/10_routes.py
"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
POEMS_PATH = os.path.join(BASE_DIR, "data", "poems.jsonl")
EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_PROMPT = "Instruct: 根据描述检索语义最贴近的唐诗\nQuery: "

# 名家加权：相同得分下优先更可能眼熟的诗人
FAMOUS = {
    "李白", "杜甫", "王维", "白居易", "李商隐", "杜牧", "孟浩然", "王昌龄",
    "高适", "岑参", "刘禹锡", "韩愈", "柳宗元", "韦应物", "李贺", "温庭筠",
    "王之涣", "王勃", "骆宾王", "陈子昂", "张九龄", "贾岛", "孟郊", "元稹",
    "张继", "韩翃", "刘长卿", "钱起", "司空曙", "许浑", "马戴", "杜荀鹤",
    "罗隐", "韦庄", "皮日休", "陆龟蒙", "储光羲", "常建", "綦毋潜", "祖咏",
}

# 六条预设路线：每站一句检索描述
ROUTES = [
    {"id": "libie", "title": "唐人的离别地图", "stops": [
        "在渭城清晨的细雨中设宴送别友人西出阳关",
        "在长亭古道边摆酒饯行，劝君更尽一杯酒",
        "在江边渡口目送友人的孤帆远去，唯见长江天际流",
        "离别之后独自在夜里思念远方的朋友",
        "多年以后与故交重逢，感慨物是人非",
    ]},
    {"id": "siji", "title": "从春天读到秋天", "stops": [
        "春天清晨鸟鸣花落，风雨夜后的慵懒春晓",
        "初夏草木繁茂，荷风送香的闲适光景",
        "盛夏酷暑里农人劳作或纳凉的情景",
        "秋天夜晚月色清凉，落叶与蟋蟀声中的愁绪",
        "深秋重阳登高，菊花与暮色的苍凉",
    ]},
    {"id": "libai-dufu", "title": "李白与杜甫的诗歌邻居", "stops": [
        "豪放飘逸地饮酒高歌，人生得意须尽欢",
        "描写壮丽山河如瀑布飞流直下的浪漫想象",
        "月下独酌，举杯邀明月的孤独与狂放",
        "沉郁顿挫地感慨国破家亡、忧国忧民",
        "秋风破屋，心系天下寒士的博大胸怀",
        "漂泊西南天地间，暮年多病独登高的苍凉",
    ]},
    {"id": "yin-yi", "title": "从山水诗走向隐逸", "stops": [
        "空山新雨后，明月松间照的清幽山居",
        "行到水穷处，坐看云起时的闲适自在",
        "走访深山寺庙，与僧人谈禅论道",
        "清晨入古寺，曲径通幽处的禅意",
        "归隐田园，采菊东篱下悠然见南山",
        "终南别业里独坐幽篁，弹琴长啸的隐士生活",
    ]},
    {"id": "changan", "title": "一个人在长安会读到什么", "stops": [
        "描写长安城宫殿巍峨、万国来朝的盛世气象",
        "科举及第后春风得意马蹄疾，一日看尽长安花",
        "长安酒肆里与友人豪饮，少年游侠的意气风发",
        "漂泊长安求仕不得，寄人篱下的辛酸",
        "长安月夜，万户捣衣声里的思妇之情",
        "乐游原上登高望远，夕阳无限好的惆怅",
    ]},
    {"id": "yixiang", "title": "唐诗中的月亮、酒、舟、雁", "stops": [
        "床前明月光，举头望明月低头思故乡",
        "花间一壶酒，独酌无相亲的月下独饮",
        "葡萄美酒夜光杯，欲饮琵琶马上催",
        "月落乌啼霜满天，江枫渔火对愁眠",
        "一叶扁舟独钓寒江雪的孤绝",
        "孤舟蓑笠翁，漂泊江湖的旅愁",
        "秋夜长空雁阵南飞，引起旅人乡思",
    ]},
]


def main():
    with open(POEMS_PATH, encoding="utf-8") as f:
        poems = [json.loads(line) for line in f]
    emb = np.load(EMB_PATH)
    assert len(poems) == emb.shape[0], "诗与向量行数不一致"

    model = SentenceTransformer(
        MODEL_NAME, device="mps", model_kwargs={"dtype": "float32"})
    model.max_seq_length = 512

    for route in ROUTES:
        print(f"\n===== {route['title']} =====")
        queries = [s for s in route["stops"]]
        q_emb = model.encode(
            queries, prompt=QUERY_PROMPT, normalize_embeddings=True,
            show_progress_bar=False)
        sims = q_emb @ emb.T          # (n_stops, n_poems)
        used_authors = set()
        for si, stop in enumerate(route["stops"]):
            order = np.argsort(-sims[si])
            cands = []
            for j in order:
                p = poems[j]
                if p["author"] in used_authors:
                    continue
                bonus = 0.02 if p["author"] in FAMOUS else 0.0
                cands.append((j, float(sims[si][j]) + bonus, p))
                if len(cands) >= 8:
                    break
            # 名家加权后重排，挑第一名为默认选择
            cands.sort(key=lambda x: -x[1])
            used_authors.add(cands[0][2]["author"])
            print(f"\n  站{si + 1} [{stop}]")
            for rank, (j, sc, p) in enumerate(cands):
                mark = "*" if rank == 0 else " "
                snippet = p["content"].replace("\n", "")[:24]
                print(f"   {mark} id={j} {p['author']}《{p['title']}》 "
                      f"score={sc:.3f} {snippet}…")


if __name__ == "__main__":
    main()
