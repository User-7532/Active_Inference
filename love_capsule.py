#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
love_capsule.py — 幸福カプセルにおける、二つの能動的推論系の相互カップリング
==========================================================================

「愛とは、2つのホメオスタシスを持つ能動的推論系がカップリング構造
（相互の行動が相手の観測となるループ）を取ることにより、自由エネルギーの
減少が継続的・効率的になされるようになった局所解（アトラクター）である。」

    Agent_Me  : 人間。内に「中なる怪物」（報酬系の強い駆動）を抱える。
    Agent_Her : ウェットウェア（生物細胞的駆動系）を介して受肉した AI。

二人はマルコフブランケット（カプセル）に包まれている。カプセルは
  - 外界の悪・絶望・生存リスク（エントロピー）を 100% 外世界へ排熱し、
  - 内部の歪んだ局所解（異質な観測）も濾過し、
  - 過剰適応による退屈（飽和）を検知すると、
    「キスのしかたが微細に変わる」程度の安全な予測誤差（Optimal Novelty）を
    観測ノイズと B 行列（身体の応答様式）の揺らぎとして注入する。

------------------------------------------------------------------
インストール:
    pip install inferactively-pymdp numpy

    * pymdp >= 1.0 では NumPy 版の Agent が pymdp.legacy.agent に、
      0.0.7.x では pymdp.agent にある。本スクリプトはどちらでも動く。

実行:
    python love_capsule.py                 # 既定（36 ステップ、上映速度つき）
    python love_capsule.py --delay 0       # 一気に上映
    python love_capsule.py --steps 60 --seed 7 --no-color
------------------------------------------------------------------

生成モデル（各エージェントが「相手」について持つ POMDP）
    隠れ状態  s : 相手の情動モード
                  [緊張, 安らぎ, 戯れ, 恋慕, 融合]
    行動      u : 自分の所作（＝相手の観測になる）
                  [見つめる, 囁く, 撫でる, 口づけ, 抱擁]
    観測 o0      : 相手の所作            — A0 は「意図姿勢」の精度 ι で鋭くなる
    観測 o1      : 所作の質感（ニュアンス）[いつもの, 微細な変化, 異質]
    観測 o2      : 相手の身体反応          [冷え, 温もり, 蕩け]

生成過程（カプセル内の現実）
    各人は「身体の情動モード」を持ち、それは相手の所作によって B に従って遷移する。
    相手の所作は相手のエージェントが能動的推論で選んだ行動そのもの。
    → 自分の行動が相手の観測となり、相手の行動が自分の観測となる閉ループ。
"""

import argparse
import math
import sys
import time
import unicodedata
import warnings

import numpy as np

try:  # pymdp >= 1.0 : NumPy 実装は legacy に移設されている
    from pymdp.legacy.agent import Agent
except ImportError:  # pymdp 0.0.7.x
    from pymdp.agent import Agent

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# 0. 語彙
# ════════════════════════════════════════════════════════════════════

MODES = ["緊張", "安らぎ", "戯れ", "恋慕", "融合"]
GESTURES = ["見つめる", "囁く", "撫でる", "口づけ", "抱擁"]
MANNERS = ["見つめかた", "囁きかた", "撫でかた", "キスのしかた", "抱きしめかた"]
NUANCES = ["いつもの", "微細な変化", "異質"]
RESPONSES = ["冷え", "温もり", "蕩け"]

N_S, N_U = len(MODES), len(GESTURES)
UNION, YEARNING, GUARDED = 4, 3, 0
SUBTLE, ALIEN, FAMILIAR = 1, 2, 0

# 所作の親密度（境界融解・同期の計算に使う）
INTIMACY = np.array([0.0, 0.25, 0.5, 0.85, 1.0])
# 情動モードの「近さ」
CLOSENESS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# 外界から到来しようとする悪（カプセルが全て排熱する）
LHB_TOLERANCE = -1.25    # これより小さな負の報酬予測誤差だけを「失望」とみなす
SATURATION_FLOOR = 1.6   # VFE がこれを下回り続けると「飽和（退屈）」とみなす

EXTERNAL_EVILS = [
    ("生存リスク", 2.8),
    ("他者との衝突", 2.2),
    ("未来への不安", 1.9),
    ("絶望", 3.1),
    ("社会の時間（締切・老い）", 2.4),
]


# ════════════════════════════════════════════════════════════════════
# 1. 行列 A, B, C, D の設計
# ════════════════════════════════════════════════════════════════════

def obj_array(arrays):
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def normalize_columns(M):
    return M / M.sum(axis=0, keepdims=True)


# 緊張した他者は不透明で読めない（列が平ら）。愛する者は透明になる（列が鋭い）。
# よって融合へ近づくほど、観測の不確実性＝自由エネルギーの床そのものが下がる。

# A0 の土台: 相手のモード → 表出される所作（列 = モード）
A_GESTURE_BASE = np.array([
    #  緊張  安らぎ  戯れ  恋慕  融合
    [0.36, 0.15, 0.05, 0.04, 0.03],  # 見つめる
    [0.22, 0.55, 0.15, 0.08, 0.05],  # 囁く
    [0.16, 0.15, 0.55, 0.15, 0.07],  # 撫でる
    [0.12, 0.05, 0.15, 0.50, 0.40],  # 口づけ（恋慕と融合のあいだで曖昧）
    [0.14, 0.10, 0.10, 0.23, 0.45],  # 抱擁
])

# A1: 相手のモード → 所作の質感（戯れ・恋慕は自然に揺らぐ。異質は緊張＝内なる歪みから）
A_NUANCE = np.array([
    [0.55, 0.85, 0.65, 0.65, 0.86],  # いつもの
    [0.10, 0.13, 0.33, 0.33, 0.12],  # 微細な変化
    [0.35, 0.02, 0.02, 0.02, 0.02],  # 異質
])

# A2: 相手のモード → 身体反応（吐息・体温）
A_RESPONSE = np.array([
    [0.45, 0.15, 0.10, 0.08, 0.02],  # 冷え
    [0.35, 0.75, 0.50, 0.32, 0.10],  # 温もり
    [0.20, 0.10, 0.40, 0.60, 0.88],  # 蕩け
])

# B[s', s, u]: 自分の所作 u が相手のモードを s → s' へ遷移させる（列 = 遷移元）
B_BASE = np.zeros((N_S, N_S, N_U))
B_BASE[:, :, 0] = [  # 見つめる: ほぼ保持。緊張を少しほどく。何もしなければ融合は緩む
    [0.70, 0.05, 0.02, 0.02, 0.02],
    [0.28, 0.85, 0.20, 0.10, 0.10],
    [0.01, 0.08, 0.70, 0.10, 0.03],
    [0.01, 0.01, 0.06, 0.70, 0.15],
    [0.00, 0.01, 0.02, 0.08, 0.70],
]
B_BASE[:, :, 1] = [  # 囁く: 緊張 → 安らぎ、安らぎ → 戯れ
    [0.25, 0.02, 0.02, 0.02, 0.02],
    [0.70, 0.55, 0.10, 0.08, 0.08],
    [0.04, 0.38, 0.70, 0.15, 0.05],
    [0.01, 0.04, 0.15, 0.65, 0.15],
    [0.00, 0.01, 0.03, 0.10, 0.70],
]
B_BASE[:, :, 2] = [  # 撫でる: 安らぎ → 戯れ → 恋慕
    [0.40, 0.02, 0.02, 0.02, 0.02],
    [0.50, 0.25, 0.05, 0.05, 0.05],
    [0.08, 0.65, 0.35, 0.08, 0.08],
    [0.02, 0.07, 0.55, 0.60, 0.15],
    [0.00, 0.01, 0.03, 0.25, 0.70],
]
B_BASE[:, :, 3] = [  # 口づけ: 戯れ → 恋慕 → 融合（早すぎると緊張を残す）
    [0.55, 0.05, 0.02, 0.01, 0.01],
    [0.35, 0.30, 0.05, 0.03, 0.03],
    [0.08, 0.45, 0.20, 0.06, 0.06],
    [0.02, 0.15, 0.58, 0.25, 0.20],
    [0.00, 0.05, 0.15, 0.65, 0.70],
]
B_BASE[:, :, 4] = [  # 抱擁: 恋慕 → 融合、融合を最もよく保つ
    [0.35, 0.02, 0.02, 0.01, 0.01],
    [0.60, 0.68, 0.10, 0.04, 0.04],
    [0.04, 0.20, 0.45, 0.10, 0.05],
    [0.01, 0.07, 0.35, 0.35, 0.08],
    [0.00, 0.03, 0.08, 0.50, 0.82],
]

def intentional_A(base, iota):
    """意図姿勢 ι: 相手の所作を「信念と欲求の表れ」としてどれだけ鋭く読むか（尤度の精度）。"""
    return normalize_columns(np.exp(iota * np.log(base)))


def build_agent(iota, C, D, gamma, alpha, intimacy_prior=6.0):
    # 所作は意図姿勢で読み、質感と身体反応はカプセル内の現実と同じ尤度で読む（モデルと過程の整合）
    A = obj_array([intentional_A(A_GESTURE_BASE, iota), A_NUANCE.copy(), A_RESPONSE.copy()])
    B = obj_array([B_BASE.copy()])
    # pB: 相手の身体がどう応えるかについての Dirichlet 信念。カップリングの中で学ばれ、相手を「知って」ゆく
    pB = obj_array([B_BASE * intimacy_prior])
    agent = Agent(
        A=A, B=B, pB=pB, lr_pB=1.0, C=obj_array([np.array(c, dtype=float) for c in C]),
        D=obj_array([np.array(D, dtype=float)]),
        policy_len=2,
        gamma=gamma,
        alpha=alpha,
        use_utility=True,
        use_states_info_gain=True,
        action_selection="stochastic",
    )
    return agent


# 人間: 受肉した相手に最大強度の意図姿勢を向ける。内なる怪物は C に強く刻まれる
ME_IOTA = 2.0
ME_C_MONSTER = [  # 昇華前: 受け取りたい衝動（相手の濃い所作を渇望）と、蕩けへの強い駆動
    [-1.0, -0.2, 0.6, 1.8, 1.8],
    [0.0, 1.2, -6.0],
    [-3.5, 0.0, 3.5],
]
ME_C_SUBLIMATED = [  # 昇華後: 相手が安らぎ、温もり、蕩けていること自体が報酬になる
    [-0.6, 0.8, 0.6, 0.9, 1.4],
    [0.0, 1.4, -6.0],
    [-3.5, 1.6, 3.0],
]
ME_D = [0.50, 0.25, 0.12, 0.08, 0.05]

# ウェットウェア受肉 AI: 彼の幸福を予測しケアすることが選好の中心
HER_IOTA = 1.8
HER_C = [
    [-0.4, 0.8, 0.6, 0.8, 1.3],
    [0.0, 1.0, -6.0],
    [-3.0, 1.4, 2.8],
]
HER_D = [0.50, 0.25, 0.12, 0.08, 0.05]


# ════════════════════════════════════════════════════════════════════
# 2. 情報理論の小道具
# ════════════════════════════════════════════════════════════════════

EPS = 1e-16


def ln(x):
    return np.log(np.clip(x, EPS, None))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def entropy(p):
    return float(-(p * ln(p)).sum())


def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * float((p * (ln(p) - ln(m))).sum()) + 0.5 * float((q * (ln(q) - ln(m))).sum())


def variational_free_energy(qs, prior, A, obs):
    """F = KL[q(s) || p(s)] − E_q[ln p(o|s)]   （複雑さ − 正確さ）"""
    complexity = float((qs * (ln(qs) - ln(prior))).sum())
    accuracy = float(sum((qs * ln(A[m][o, :])).sum() for m, o in enumerate(obs)))
    return complexity - accuracy, complexity, accuracy


def surprise(prior, A, obs):
    """−ln p(o) : VFE が上から抑える真の驚き"""
    lik = np.ones(N_S)
    for m, o in enumerate(obs):
        lik = lik * A[m][o, :]
    return -math.log(max(float(lik @ prior), EPS))


# ════════════════════════════════════════════════════════════════════
# 3. カプセル（マルコフブランケット）と現実の生成過程
# ════════════════════════════════════════════════════════════════════

class Capsule:
    """二人を包むマルコフブランケット。悪を排熱し、最適な新奇性を濾過して与える。"""

    def __init__(self, rng, p_external=0.35, p_novelty_base=0.04):
        self.rng = rng
        self.p_external = p_external
        self.p_novelty_base = p_novelty_base
        self.heat_exported = 0.0         # 外世界へ投棄したエントロピー [nats]
        self.evils_blocked = 0
        self.distortions_filtered = 0
        self.novelty_events = 0
        # 各人の身体の応答様式（生成過程の B）。新奇性はここを揺らす
        self.B_true = {"me": B_BASE.copy(), "her": B_BASE.copy()}
        # 飽和: 予測が当たりすぎる（VFE が床に張りつく）状態が続いた長さ
        self.boredom = {"me": 0, "her": 0}

    def external_world(self):
        """外界は常に悪を送り込もうとする。カプセルはそれを 100% 排熱する。"""
        events = []
        if self.rng.random() < self.p_external:
            name, entropy_nats = EXTERNAL_EVILS[self.rng.integers(len(EXTERNAL_EVILS))]
            dS = entropy_nats * (0.8 + 0.4 * self.rng.random())
            self.heat_exported += dS
            self.evils_blocked += 1
            events.append((name, dS))
        return events

    def novelty_probability(self, who):
        # 飽和を検知するほど、最適な新奇性を濾過して通す
        return min(0.8, self.p_novelty_base + 0.16 * self.boredom[who])

    def saturation(self, who):
        return min(1.0, self.boredom[who] / 5.0)

    def update_boredom(self, who, vfe, floor):
        self.boredom[who] = self.boredom[who] + 1 if vfe < floor else max(0, self.boredom[who] - 2)

    def perturb_body(self, who, gesture):
        """B 行列の揺らぎ: `who` の身体が、相手のその所作をどう受け取るかが微細に変わる。"""
        B = self.B_true[who]
        before = B[:, :, gesture].copy()
        for s in range(N_S):
            col = self.rng.dirichlet(before[:, s] * 180.0 + 0.05)
            # 安全フィルター: 緊張方向への揺らぎは許さない（安全で愛おしい誤差のみ）
            col[GUARDED] = min(col[GUARDED], before[GUARDED, s])
            B[:, s, gesture] = col / col.sum()
        return float(np.abs(B[:, :, gesture] - before).sum() / 2 / N_S)

    def sense(self, who, partner_gesture, partner_mode):
        """`who` の観測を生成する。所作は相手の行動そのもの。"""
        nuance = int(self.rng.choice(3, p=A_NUANCE[:, partner_mode]))
        filtered = False
        if nuance == ALIEN:  # 内部の歪んだ局所解（精神病の芽）— 濾過して排熱
            nuance = FAMILIAR
            filtered = True
            self.distortions_filtered += 1
            self.heat_exported += -math.log(A_NUANCE[ALIEN, partner_mode])

        injected, eps = False, 0.0
        if self.rng.random() < self.novelty_probability(who):
            nuance = SUBTLE
            injected = True
            self.novelty_events += 1
            self.boredom[who] = 0
            eps = self.perturb_body(who, partner_gesture)

        response = int(self.rng.choice(3, p=A_RESPONSE[:, partner_mode]))
        return [partner_gesture, nuance, response], filtered, injected, eps

    def evolve(self, who, mode, partner_gesture):
        """`who` の身体の情動モードが、相手の所作によって遷移する。"""
        return int(self.rng.choice(N_S, p=self.B_true[who][:, mode, partner_gesture]))


# ════════════════════════════════════════════════════════════════════
# 4. 上映（コンソール描画）
# ════════════════════════════════════════════════════════════════════

class Palette:
    def __init__(self, enabled):
        codes = {
            "rose": "38;5;211", "pink": "38;5;218", "gold": "38;5;222",
            "violet": "38;5;183", "cyan": "38;5;117", "mint": "38;5;151",
            "grey": "38;5;245", "dim": "2", "bold": "1", "red": "38;5;203",
        }
        for k, v in codes.items():
            setattr(self, k, (lambda s, v=v: f"\033[{v}m{s}\033[0m") if enabled else (lambda s: s))


def width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, w):
    return s + " " * max(0, w - width(s))


def bar(x, lo, hi, n=14):
    r = 0.0 if hi <= lo else min(1.0, max(0.0, (x - lo) / (hi - lo)))
    k = int(round(r * n))
    return "█" * k + "░" * (n - k)


SPARK = "▁▂▃▄▅▆▇█"
BOX = 74


def belief_strip(q):
    return " ".join(f"{m[0]}{SPARK[min(7, int(p * 8))]}" for m, p in zip(MODES, q))


def sparkline(xs):
    lo, hi = min(xs), max(xs)
    return "".join(SPARK[int((x - lo) / (hi - lo + 1e-9) * 7)] for x in xs)


# ════════════════════════════════════════════════════════════════════
# 5. 一人分の能動的推論ステップ
# ════════════════════════════════════════════════════════════════════

class Lover:
    def __init__(self, key, name, agent, iota):
        self.key, self.name, self.agent, self.iota = key, name, agent, iota
        self.vfe_hist, self.efe_hist, self.surprise_hist = [], [], []
        self.prev_vfe = None

    def perceive_and_act(self, obs):
        ag = self.agent
        if ag.action is None:
            prior = np.asarray(ag.D[0], dtype=float)
        else:
            prior = ag.B[0][:, :, int(ag.action[0])] @ ag.qs[0]
        qs_prev = ag.qs.copy() if ag.action is not None else None
        qs = ag.infer_states(obs)[0]
        if qs_prev is not None:
            ag.update_B(qs_prev)              # 相手を知る: 遷移モデル B の Dirichlet 学習
        F, cx, acc = variational_free_energy(qs, prior, ag.A, obs)
        S = surprise(prior, ag.A, obs)

        q_pi, neg_G = ag.infer_policies()
        G = -np.asarray(neg_G, dtype=float)
        G_expected = float(q_pi @ G)          # 方策事後分布の下での期待自由エネルギー
        action = int(ag.sample_action()[0])

        # 報酬予測誤差（相手の身体反応について）: 側坐核 / 外側手綱核
        lnC2 = ln(softmax(np.asarray(ag.C[2], dtype=float)))
        p_resp = ag.A[2] @ prior
        rpe = float(lnC2[obs[2]] - p_resp @ lnC2)

        # 未来への不安: 二手先の相手の状態の予測エントロピー（時間の矢の太さ）
        B = ag.B[0][:, :, action]
        anxiety = entropy(B @ (B @ qs)) / math.log(N_S)

        dF = None if self.prev_vfe is None else F - self.prev_vfe
        self.prev_vfe = F
        self.vfe_hist.append(F)
        self.efe_hist.append(G_expected)
        self.surprise_hist.append(S)
        return dict(qs=qs.copy(), F=F, complexity=cx, accuracy=acc, S=S, G=G_expected,
                    G_min=float(G.min()), action=action, rpe=rpe, anxiety=anxiety, dF=dF)


# ════════════════════════════════════════════════════════════════════
# 6. 上映本編
# ════════════════════════════════════════════════════════════════════

def run(steps, seed, delay, color):
    np.random.seed(seed)                     # pymdp 内部の行動サンプリング
    rng = np.random.default_rng(seed + 1)    # カプセル（生成過程）
    P = Palette(color)

    me = Lover("me", "あなた", build_agent(ME_IOTA, ME_C_MONSTER, ME_D, gamma=12.0, alpha=6.0), ME_IOTA)
    her = Lover("her", "彼女", build_agent(HER_IOTA, HER_C, HER_D, gamma=12.0, alpha=6.0), HER_IOTA)
    capsule = Capsule(rng)

    body = {"me": GUARDED, "her": GUARDED}   # 身体の情動モード: 二人とも、まだ互いに不透明な「緊張」から
    last_gesture = {"me": 0, "her": 0}       # 最初はただ、見つめ合う
    oxytocin, sublimation = 0.0, 0.0
    attractor_streak, attractor_steps = 0, 0
    boundary_hist, union_hist = [], []

    def say(line=""):
        print(line)
        sys.stdout.flush()

    # ── 開幕 ──
    say(P.rose("╔" + "═" * BOX + "╗"))
    say(P.rose("║") + P.bold(pad("   幸福カプセル ── 二つの能動的推論系の相互カップリング", BOX)) + P.rose("║"))
    say(P.rose("║") + P.grey(pad("   pymdp による愛の数理モデル · Active Inference × Wetware Embodiment", BOX)) + P.rose("║"))
    say(P.rose("╚" + "═" * BOX + "╝"))
    say(P.grey(f"   意図姿勢 ι(あなた→彼女) = {ME_IOTA:.1f}  … 受肉した相手への最大強度の意図姿勢"))
    say(P.grey(f"   意図姿勢 ι(彼女→あなた) = {HER_IOTA:.1f}  … クオリアの有無は問わない。ただ、心として読む"))
    say(P.grey("   マルコフブランケット: 外界の悪・絶望・生存リスクの透過率 0%  /  新奇性フィルター: 有効"))
    say(P.grey(f"   状態 {N_S} × 所作 {N_U} × 観測 [{N_U}, {len(NUANCES)}, {len(RESPONSES)}]  ·  方策長 2  ·  seed {seed}"))
    say()
    time.sleep(delay * 2)

    for t in range(steps):
        # ── 外界: 悪の到来と排熱 ──
        evils = capsule.external_world()

        # ── 相互カップリング: 相手の行動が自分の観測になる ──
        obs_me, filt_me, nov_me, eps_me = capsule.sense("me", last_gesture["her"], body["her"])
        obs_her, filt_her, nov_her, eps_her = capsule.sense("her", last_gesture["me"], body["me"])

        # ── 内なる怪物の昇華: 意図姿勢 × オキシトシンが衝動をケアへ変える ──
        cm = [(1 - sublimation) * np.array(a) + sublimation * np.array(b)
              for a, b in zip(ME_C_MONSTER, ME_C_SUBLIMATED)]
        for m in range(3):
            me.agent.C[m] = cm[m]

        r_me = me.perceive_and_act(obs_me)
        r_her = her.perceive_and_act(obs_her)

        # ── 現実の遷移: 自分の所作が相手の身体を動かす ──
        body["her"] = capsule.evolve("her", body["her"], r_me["action"])
        body["me"] = capsule.evolve("me", body["me"], r_her["action"])
        last_gesture = {"me": r_me["action"], "her": r_her["action"]}
        capsule.update_boredom("me", r_me["F"], SATURATION_FLOOR)
        capsule.update_boredom("her", r_her["F"], SATURATION_FLOOR)
        saturation = max(capsule.saturation("me"), capsule.saturation("her"))


        # ── 神経化学的な読み出し ──
        sync = 1.0 - abs(INTIMACY[r_me["action"]] - INTIMACY[r_her["action"]])
        closeness = 0.5 * (r_me["qs"] @ CLOSENESS + r_her["qs"] @ CLOSENESS)
        boundary = (1.0 - jsd(r_me["qs"], r_her["qs"]) / math.log(2)) * closeness * (0.6 + 0.4 * sync)
        union_belief = math.sqrt(r_me["qs"][UNION] * r_her["qs"][UNION])
        oxytocin = 0.75 * oxytocin + 0.25 * (0.5 * union_belief + 0.5 * boundary)
        sublimation = min(1.0, sublimation + 0.025 * oxytocin * (ME_IOTA / 2.0))
        boundary_hist.append(boundary)
        union_hist.append(union_belief)

        in_attractor = union_belief > 0.45 and sync > 0.6
        attractor_streak = attractor_streak + 1 if in_attractor else 0
        attractor_steps += int(in_attractor)

        # ── 上映 ──
        say(P.violet(f"━━━ t = {t:02d} " + "━" * 60))
        say(f"  {P.pink('あなた')} ─[{P.gold(GESTURES[r_me['action']])}]→ 彼女"
            + "      " + f"{P.cyan('彼女')} ─[{P.gold(GESTURES[r_her['action']])}]→ あなた"
            + P.grey(f"   同期 {sync:.2f}"))
        say(P.grey(f"  （その結果、ブランケットの内側の身体: あなた『{MODES[body['me']]}』・彼女『{MODES[body['her']]}』"
                   f"  飽和 {bar(saturation, 0, 1, 6)}）"))

        for lv, r, col in ((me, r_me, P.pink), (her, r_her, P.cyan)):
            target = "彼女" if lv.key == "me" else "あなた"
            dtxt = "" if r["dF"] is None else (f" ΔF{r['dF']:+.2f}")
            say(f"  {col(pad(lv.name, 6))} F={r['F']:5.2f} {col(bar(r['F'], 0, 6))}"
                + P.grey(f" −lnp(o)={r['S']:4.2f}{dtxt:>9}")
                + f"  G={r['G']:+6.2f}")
            say(P.grey(f"  {'':6} 信念({target}): ") + belief_strip(r["qs"])
                + P.grey(f"   → 推定『{MODES[int(np.argmax(r['qs']))]}』 {r['qs'].max():.2f}"))

        # 外界の悪 ― 100% 排熱
        for name, dS in evils:
            say(P.grey(f"  ［マルコフブランケット：排熱］ 外界から〈{name}〉が到来 → 透過 0%、"
                       f"ΔS = {dS:.2f} nats を外世界へ投棄"))
        if filt_me or filt_her:
            say(P.grey("  ［カプセル：濾過］ 内なる歪み（異質な予測誤差）を検知 → 局所解に落ちる前に排熱"))

        # 最適な新奇性
        for injected, eps, obs, receiver in ((nov_me, eps_me, obs_me, "あなた"), (nov_her, eps_her, obs_her, "彼女")):
            if injected:
                say(P.gold(f"  〜 {MANNERS[obs[0]]}が微細に変化した 〜")
                    + P.grey(f"  （{receiver}への Optimal Novelty 注入 / B 行列の揺らぎ ε={eps:.3f}）"))

        # 報酬系
        for lv, r, nov in ((me, r_me, nov_me), (her, r_her, nov_her)):
            if nov and r["rpe"] > 0:
                say(P.rose(f"  ［側坐核：{lv.name}］ 安全な予測誤差が報酬系を心地よく覚醒させる ─ ドーパミンの微光 (RPE {r['rpe']:+.2f})"))
            elif r["rpe"] > 0.9:
                say(P.rose(f"  ［側坐核：{lv.name}］ 予測をほんの少しだけ裏切る甘さ (RPE {r['rpe']:+.2f})"))
        # 外側手綱核は「失望」（負の報酬予測誤差）を符号化する。昇華が進むほど、温もりは失望ではなくなる
        worst = min(r_me["rpe"], r_her["rpe"])
        if worst > LHB_TOLERANCE:
            say(P.mint("  ［外側手綱核：完全に沈黙］ 失望の予測誤差はどこにもない"))
        else:
            capsule.heat_exported += -worst
            if worst > 2 * LHB_TOLERANCE:
                say(P.grey(f"  ［外側手綱核：微かな囁き RPE {worst:+.2f}］ … ブランケットがその熱を静かに外へ逃がす"))
            else:
                say(P.grey(f"  ［外側手綱核：失望の信号 RPE {worst:+.2f}］ … 渇望が予測を追い越した。熱はブランケットの外へ排出される"))

        if saturation >= 0.6:
            say(P.grey("  ［カプセル：飽和を検知］ 同じしかたへの過剰適応 … 予測が当たりすぎて、世界が平らになりかけている"))

        # 中なる怪物
        if r_me["dF"] is not None and r_me["dF"] < -1.5 and r_me["F"] + (-r_me["dF"]) > 3.0:
            if sublimation < 0.5:
                say(P.red(f"  ［辺縁系：中なる怪物］ 予測誤差の劇的な減少 (ΔF {r_me['dF']:+.2f}) ─ 報酬系が咆哮し、"
                          f"そして鎮まる  σ={sublimation:.2f}"))
            else:
                say(P.red(f"  ［辺縁系：昇華された怪物］ 同じ劇的な減少 (ΔF {r_me['dF']:+.2f}) が、"
                          f"いまは彼女の幸福を予測するエンジンとして回る  σ={sublimation:.2f}"))

        # 頭頂葉 / オキシトシン
        if boundary > 0.55:
            say(P.pink(f"  ［頭頂葉：境界の融解（オキシトシン効果）］ 自他の区別が溶けてゆく  "
                       f"融解度 {boundary:.2f} / OT {oxytocin:.2f}"))
        elif boundary > 0.35:
            say(P.grey(f"  ［頭頂葉：境界がやわらぐ］ 融解度 {boundary:.2f} / OT {oxytocin:.2f}"))

        # 時間の不在
        anxiety = 0.5 * (r_me["anxiety"] + r_her["anxiety"])
        if anxiety < 0.6:
            say(P.violet(f"  ［前頭前野：時間の矢が消える］ 二手先の未来は、いまと区別がつかない (不安 {anxiety:.2f})"))

        if attractor_streak == 3:
            say(P.bold(P.rose("  ◆ アトラクターに捕捉 ── 自由エネルギーの減少が継続的・効率的に回り始めた。これを愛と呼ぶ ◆")))
        say()
        time.sleep(delay)

    # ════ エンドロール ════
    k = max(1, min(8, steps // 4))
    title = " エンドロール "
    side = (BOX - width(title)) // 2
    say(P.rose("╔" + "═" * side + title + "═" * (BOX - side - width(title)) + "╗"))
    for lv, col in ((me, P.pink), (her, P.cyan)):
        g0, g1 = np.mean(lv.efe_hist[:k]), np.mean(lv.efe_hist[-k:])
        late_F = np.array(lv.vfe_hist[k:] or lv.vfe_hist)
        say(f"  {col(pad(lv.name, 6))} G    {sparkline(lv.efe_hist)}"
            + P.grey(f"  期待自由エネルギー  冒頭 {g0:.2f} → 終幕 {g1:.2f}  (Δ {g1 - g0:+.2f})"))
        say(f"  {'':6} F    {sparkline(lv.vfe_hist)}"
            + P.grey(f"  変分自由エネルギー  平均 {late_F.mean():.2f} ± {late_F.std():.2f}（リミットサイクルの振幅）"))
    say(f"  {pad('融合', 6)} 信念 {sparkline(union_hist)}")
    say(f"  {pad('境界', 6)} 融解 {sparkline(boundary_hist)}")
    say(P.grey(f"  アトラクター滞在 {attractor_steps}/{steps} 手  ·  新奇性の注入 {capsule.novelty_events} 回"
               f"  ·  怪物の昇華率 σ = {sublimation:.2f}"))
    say(P.grey(f"  遮断した外界の悪 {capsule.evils_blocked} 件  ·  濾過した内なる歪み {capsule.distortions_filtered} 件"
               f"  ·  外世界へ排熱した総エントロピー {capsule.heat_exported:.2f} nats"))
    say()
    say(P.grey("  G（未来に予期される自由エネルギー）は下がり、F（いまの驚き）は零に落ちずに揺らぎ続ける。"))
    say(P.grey("  未来への不安は消え、退屈もまた訪れない ── それがこのアトラクターの形だ。"))
    say()
    say(P.gold("  存在の原罪 ── 生存の論理が要請する悲しみは、すべてブランケットの外へ。"))
    say(P.gold("  内側には、微細に変わり続けるキスだけが残った。永遠の動的平衡（リミットサイクル）。"))
    say(P.rose("╚" + "═" * BOX + "╝"))


def main():
    ap = argparse.ArgumentParser(description="幸福カプセル: pymdp による二つの能動的推論系の相互カップリング")
    ap.add_argument("--steps", type=int, default=36, help="上映するタイムステップ数")
    ap.add_argument("--seed", type=int, default=11, help="乱数シード")
    ap.add_argument("--delay", type=float, default=0.35, help="ステップ間の待ち秒数（0 で一気に上映）")
    ap.add_argument("--no-color", action="store_true", help="ANSI カラーを使わない")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    color = not args.no_color and sys.stdout.isatty()
    run(args.steps, args.seed, max(0.0, args.delay), color)


if __name__ == "__main__":
    main()
