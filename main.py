# main.py
# 극지식물 개체별 생육 차이 분석 대시보드
# 학교별로 CSV 열 이름·형식·결측이 달라도 동작하도록 작성

import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_CSV = os.path.join(BASE_DIR, "송도고등학교_환경데이터.csv")
GROWTH_CSV = os.path.join(BASE_DIR, "송도고등학교_생육결과데이터.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 한글 폰트 설정 (운영체제별 우선순위)
# ============================================================
def setup_korean_font():
    """matplotlib 한글 폰트 설정. 가능한 폰트를 순서대로 시도."""
    candidates = [
        "AppleGothic",            # macOS
        "Malgun Gothic",          # Windows
        "Noto Sans CJK KR",      # Linux (Noto 설치 시)
        "NanumGothic",           # 나눔폰트
        "NanumBarunGothic",
    ]
    for name in candidates:
        try:
            plt.rcParams["font.family"] = name
            fig = plt.figure()
            fig.canvas.draw()
            plt.close(fig)
            return name
        except Exception:
            continue
    # 실패 시 rcParams만 설정 (일부 글자 깨질 수 있음)
    plt.rcParams["font.family"] = "sans-serif"
    return None

FOUND_FONT = setup_korean_font()

# 음수 부호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# 레이아웃 기본값
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================
# 환경 데이터 로드 (열 이름·시간 형식 유연 처리)
# ============================================================
def load_environment_data(filepath):
    """
    학교별로 다른 열 이름·시간 형식을 자동으로 처리.
    반환되는 DataFrame은 통일된 열 이름 사용.
    """
    df = pd.read_csv(filepath)

    # --- 시간 열 탐지 ---
    time_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("time", "일시", "시간"):
            time_col = str(c).strip()
            break

    if time_col is None:
        raise ValueError(
            f"[환경데이터] 시간 열을 찾을 수 없습니다. "
            f"현재 열: {list(df.columns)}"
        )

    # --- 시간 파싱: 두 형식 모두 처리 ---
    #  형식 A: '2026-07-15 06:00:00'
    #  형식 B: '20260715_0' (날짜_시각)
    def parse_time(val):
        s = str(val).strip()
        # 형식 A 시도
        try:
            return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            pass
        # 형식 B 시도: '20260715_0' → 날짜 + 시각(0~23)
        m = re.match(r"^(\d{8})_(\d+)$", s)
        if m:
            date_str = m.group(1)
            hour = int(m.group(2)) % 24
            try:
                dt = pd.to_datetime(date_str, format="%Y%m%d")
                return dt + pd.Timedelta(hours=hour)
            except (ValueError, TypeError):
                pass
        # 최종 fallback
        return pd.to_datetime(s, errors="coerce")

    df["time_parsed"] = df[time_col].apply(parse_time)

    # --- 필요 수치 열 자동 탐지 ---
    def find_col(columns, keywords, exclude_keywords=()):
        for c in columns:
            cl = str(c).strip().lower()
            if any(k in cl for k in keywords) and not any(
                ek in cl for ek in exclude_keywords
            ):
                return str(c).strip()
        return None

    temp_col = find_col(df.columns, ["temperature", "온도"])
    hum_col = find_col(df.columns, ["humidity", "습도"])
    ec_col = find_col(df.columns, ["ec"])
    ph_col = find_col(df.columns, ["ph"])
    wt_col = find_col(df.columns, ["water_temp", "수온"])
    co2_col = find_col(df.columns, ["co2"])

    env = pd.DataFrame({"time": df["time_parsed"]})

    if temp_col:
        env["temperature"] = pd.to_numeric(df[temp_col], errors="coerce")
    if hum_col:
        env["humidity"] = pd.to_numeric(df[hum_col], errors="coerce")
    if ec_col:
        env["ec"] = pd.to_numeric(df[ec_col], errors="coerce")
    if ph_col:
        env["ph"] = pd.to_numeric(df[ph_col], errors="coerce")
    if wt_col:
        env["water_temp"] = pd.to_numeric(df[wt_col], errors="coerce")
    if co2_col:
        env["co2"] = pd.to_numeric(df[co2_col], errors="coerce")

    # 결측 행 제거 (시간만 있는 빈 행)
    env = env.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # 설명 출력
    present = [c for c in ["temperature", "humidity", "ec", "ph", "water_temp", "co2"] if c in env.columns]
    print(f"[환경데이터] 로드 완료: {len(env)}행, {len(env.columns)-1}개 수치열")
    print(f"  감지된 수치열: {present}")
    print(f"  시간 범위: {env['time'].min()} ~ {env['time'].max()}")
    print(f"  결측률 (수치열): "
          f"{env[present].isna().sum().sum() / (len(env)*len(present)) * 100:.1f}%")
    return env


# ============================================================
# 생육 결과 데이터 로드 (단위·띄어쓰기 붙은 열 이름 처리)
# ============================================================
def load_growth_data(filepath):
    """
    '잎 수(장)', '지상부 길이(mm)', '생중량(g)'처럼
    단위·띄어쓰기가 붙은 열 이름을 자동 처리하여 통일된 DataFrame으로 반환.
    """
    df = pd.read_csv(filepath)
    cols_lower = [str(c).strip().lower() for c in df.columns]

    # --- 개체번호 ---
    id_col = None
    for i, c in enumerate(cols_lower):
        if "개체번호" in c or c in ("id", "번호", "개체"):
            id_col = df.columns[i]
            break
    if id_col is None:
        raise ValueError(f"[생육데이터] 개체번호 열을 찾을 수 없습니다. 열: {list(df.columns)}")

    # --- 잎 수 ---
    leaf_col = None
    for i, c in enumerate(cols_lower):
        if "잎" in c:
            leaf_col = df.columns[i]
            break
    if leaf_col is None:
        raise ValueError(f"[생육데이터] 잎 수 열을 찾을 수 없습니다. 열: {list(df.columns)}")

    # --- 지상부 길이 ---
    shoot_len_col = None
    for i, c in enumerate(cols_lower):
        if "지상부" in c and "길이" in c:
            shoot_len_col = df.columns[i]
            break
    if shoot_len_col is None:
        raise ValueError("[생육데이터] 지상부 길이 열을 찾을 수 없습니다.")

    # --- 지하부 길이 ---
    root_len_col = None
    for i, c in enumerate(cols_lower):
        if "지하부" in c and "길이" in c:
            root_len_col = df.columns[i]
            break
    if root_len_col is None:
        raise ValueError("[생육데이터] 지하부 길이 열을 찾을 수 없습니다.")

    # --- 생중량 ---
    fresh_col = None
    for i, c in enumerate(cols_lower):
        if "생중량" in c:
            fresh_col = df.columns[i]
            break

    # --- 지상부 질량 ---
    shoot_mass_col = None
    for i, c in enumerate(cols_lower):
        if "지상부" in c and ("질량" in c or "무게" in c):
            shoot_mass_col = df.columns[i]
            break

    # --- 지하부 질량 ---
    root_mass_col = None
    for i, c in enumerate(cols_lower):
        if "지하부" in c and ("질량" in c or "무게" in c):
            root_mass_col = df.columns[i]
            break

    # 원본 열에서 필요한 값만 추출
    def get_series(col):
        if col is None:
            return None
        return pd.to_numeric(df[col], errors="coerce")

    df_out = pd.DataFrame({
        "개체번호": get_series(id_col),
        "잎수": get_series(leaf_col),
        "지상부길이": get_series(shoot_len_col),
        "지하부길이": get_series(root_len_col),
    })

    # 생중량: 있으면 사용, 없으면 지상부질량 + 지하부질량으로 계산
    if fresh_col is not None:
        fresh_series = get_series(fresh_col)
        if fresh_series.notna().sum() > 0:
            df_out["생중량"] = fresh_series
            calc_note = "CSV 생중량(g) 사용"
        else:
            # 값은 있지만 전부 NaN이면 계산
            if shoot_mass_col is not None and root_mass_col is not None:
                df_out["생중량"] = get_series(shoot_mass_col) + get_series(root_mass_col)
                calc_note = "지상부질량 + 지하부질량으로 계산 (생중량 열 NaN)"
            else:
                raise ValueError(
                    "[생육데이터] 생중량(g) 열이 없고 "
                    "지상부질량/지하부질량도 없습니다."
                )
    else:
        if shoot_mass_col is not None and root_mass_col is not None:
            df_out["생중량"] = get_series(shoot_mass_col) + get_series(root_mass_col)
            calc_note = "지상부질량 + 지하부질량으로 계산 (생중량 열 없음)"
        else:
            raise ValueError(
                "[생육데이터] 생중량(g) 열이 없고 "
                "지상부질량/지하부질량도 없습니다."
            )

    # 결측치 정리
    before = len(df_out)
    df_out = df_out.dropna(subset=["개체번호", "잎수", "지상부길이", "지하부길이", "생중량"])
    df_out["개체번호"] = df_out["개체번호"].astype(int)
    df_out = df_out.sort_values("개체번호").reset_index(drop=True)

    print(f"[생육데이터] 로드 완료: {len(df_out)}개체 (원본 {before}행 중 결측 제거)")
    print(f"  생중량 출처: {calc_note}")
    print(f"  열: {list(df_out.columns)}")
    return df_out


# ============================================================
# 분석 1: 개체별 생중량 분포 (평균·표준편차·CV)
# ============================================================
def analysis_distribution(growth_df):
    """
    1) 개체별 생중량 분포와 차이 분석
       - 평균, 표준편차, CV(변동계수), 최소, 최대, 중앙값
       - 히스토그램 + KDE
       - 상자그림
    """
    fw = growth_df["생중량"].dropna()
    mean = fw.mean()
    std = fw.std(ddof=1)
    cv = (std / mean * 100) if mean > 0 else np.nan
    mn, mx, md = fw.min(), fw.max(), fw.median()

    print("\n" + "=" * 56)
    print("  분석 1. 개체별 생중량 분포")
    print("=" * 56)
    print(f"  개체 수         : {len(fw):>6}")
    print(f"  평균 (g)        : {mean:>6.2f}")
    print(f"  표준편차 (g)    : {std:>6.2f}")
    print(f"  CV (%)          : {cv:>6.2f}")
    print(f"  최소 (g)        : {mn:>6.2f}")
    print(f"  최대 (g)        : {mx:>6.2f}")
    print(f"  중앙값 (g)      : {md:>6.2f}")
    print(f"  범위 (g)        : {mx - mn:>6.2f}")
    print("=" * 56)

    # --- 히스토그램 + KDE ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(fw, bins="auto", kde=True, color="#2E86AB",
                 stat="density", ax=ax1, linewidth=1.5)
    ax1.axvline(mean, color="#D62828", linestyle="--", linewidth=2,
                label=f"평균 = {mean:.2f}g")
    ax1.axvline(md, color="#F77F00", linestyle=":", linewidth=2,
                label=f"중앙값 = {md:.2f}g")
    ax1.set_title("개체별 생중량 분포 (Histogram + KDE)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("생중량 (g)")
    ax1.set_ylabel("밀도")
    ax1.legend(frameon=True, fontsize=9)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "01_생중량_분포_히스토그램.png"))
    plt.close(fig1)

    # --- 상자그림 ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    bp = ax2.boxplot(fw.values, vert=True, patch_artist=True,
                     widths=0.5, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="#D62828",
                                    markeredgecolor="white", markersize=7))
    bp["boxes"][0].set(facecolor="#A8DADC", edgecolor="#1D3557", linewidth=1.5)
    bp["medians"][0].set(color="#1D3557", linewidth=2)
    bp["whiskers"][0].set(color="#1D3557", linewidth=1.2)
    bp["whiskers"][1].set(color="#1D3557", linewidth=1.2)
    bp["caps"][0].set(color="#1D3557", linewidth=1.2)
    bp["caps"][1].set(color="#1D3557", linewidth=1.2)
    ax2.set_title("개체별 생중량 상자그림", fontsize=13, fontweight="bold")
    ax2.set_ylabel("생중량 (g)")
    ax2.set_xticklabels(["전체 개체"])
    # 통계 annot
    ax2.text(0.98, 0.97,
             f"n = {len(fw)}\n"
             f"평균 = {mean:.2f}g\n"
             f"σ = {std:.2f}g\n"
             f"CV = {cv:.1f}%",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", fc="#F1FAEE", ec="#1D3557", alpha=0.9))
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "01_생중량_분포_상자그림.png"))
    plt.close(fig2)

    # 결과 저장용 딕셔너리
    result = {
        "n": len(fw),
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": mn,
        "max": mx,
        "median": md,
    }
    return result


# ============================================================
# 분석 2: 잎 수·지상부 길이 ↔ 생중량 상관 (상관계수·산점도)
# ============================================================
def analysis_correlation(growth_df):
    """
    2) 잎 수·지상부 길이와 생중량의 관계
       - Pearson 상관계수 + p-value
       - 산점도 + 회귀선
    """
    df = growth_df.dropna(subset=["잎수", "지상부길이", "생중량"])

    r_leaf, p_leaf = stats.pearsonr(df["잎수"], df["생중량"])
    r_shoot, p_shoot = stats.pearsonr(df["지상부길이"], df["생중량"])

    print("\n" + "=" * 56)
    print("  분석 2. 잎 수·지상부 길이 ↔ 생중량 상관")
    print("=" * 56)
    print(f"  잎 수 ↔ 생중량")
    print(f"    Pearson r  = {r_leaf:+.4f}")
    print(f"    p-value    = {p_leaf:.4e}  {'*' if p_leaf < 0.05 else ' ns'}")
    print(f"  지상부 길이 ↔ 생중량")
    print(f"    Pearson r  = {r_shoot:+.4f}")
    print(f"    p-value    = {p_ship:.4e}  {'*' if p_ship < 0.05 else ' ns'}")
    print("=" * 56)

    # --- 산점도 2개 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, xcol, xlabel, r, p in [
        (axes[0], "잎수", "잎 수 (장)", r_leaf, p_leaf),
        (axes[1], "지상부길이", "지상부 길이 (mm)", r_shoot, p_shoot),
    ]:
        sns.scatterplot(data=df, x=xcol, y="생중량",
                        color="#2E86AB", s=55, alpha=0.7,
                        edgecolor="white", linewidth=0.5, ax=ax)
        sns.regplot(data=df, x=xcol, y="생중량",
                    scatter=False, color="#D62828",
                    line_kws=dict(linewidth=2), ax=ax)
        ax.set_title(f"{xlabel} ↔ 생중량", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("생중량 (g)")
        ax.text(0.03, 0.97,
                f"r = {r:+.3f}\np = {p:.3e}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#F1FAEE",
                          ec="#1D3557", alpha=0.9))

    fig.suptitle("잎 수·지상부 길이와 생중량의 관계", fontsize=14, fontweight="bold",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_상관계수_산점도.png"))
    plt.close(fig)

    return {
        "r_leaf": r_leaf,
        "p_leaf": p_leaf,
        "r_shoot": r_shoot,
        "p_shoot": p_ship,
    }


# ============================================================
# 분석 3: 생중량 상위 25% vs 하위 25% 개체 비교
# ============================================================
def analysis_quartile_comparison(growth_df):
    """
    3) 생중량 기준 상위 25% vs 하위 25% 개체 비교
       - 3분위 경계값 계산
       - 그룹별 평균·중앙값 비교 (잎 수, 지상부 길이, 지하부 길이)
       - 나란히 놓인 상자그림
    """
    df = growth_df.dropna(subset=["생중량", "잎수", "지상부길이", "지하부길이"]).copy()
    q75 = df["생중량"].quantile(0.75)
    q25 = df["생중량"].quantile(0.25)

    top = df[df["생중량"] >= q75].copy()
    bottom = df[df["생중량"] <= q25].copy()

    # 비교할 지표
    metrics = [
        ("잎수", "잎 수 (장)"),
        ("지상부길이", "지상부 길이 (mm)"),
        ("지하부길이", "지하부 길이 (mm)"),
    ]

    print("\n" + "=" * 56)
    print("  분석 3. 생중량 상위 25% vs 하위 25% 개체 비교")
    print("=" * 56)
    print(f"  하위 25% 경계 (Q1) : {q25:.2f} g")
    print(f"  상위 25% 경계 (Q3) : {q75:.2f} g")
    print(f"  하위 그룹 개체 수  : {len(bottom)}개체")
    print(f"  상위 그룹 개체 수  : {len(top)}개체")
    print("-" * 56)
    header = f"  {{:<16}} {{:>8}} {{:>8}} {{:>8}} {{:>8}}"
    print(header.format("지표", "하위평균", "상위평균", "하위중앙", "상위중앙"))
    print("-" * 56)
    for col, label in metrics:
        lo_mean = bottom[col].mean()
        hi_mean = top[col].mean()
        lo_med = bottom[col].median()
        hi_med = top[col].median()
        print(f"  {label:<16} {lo_mean:>8.2f} {hi_mean:>8.2f} {lo_med:>8.2f} {hi_med:>8.2f}")
    print("=" * 56)

    # --- 나란히 놓인 상자그림 ---
    data_long = pd.concat([
        top[["잎수", "지상부길이", "지하부길이"]].assign(그룹="상위 25%"),
        bottom[["잎수", "지상부길이", "지하부길이"]].assign(그룹="하위 25%"),
    ])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = {"상위 25%": "#2E86AB", "하위 25%": "#D62828"}

    for ax, (col, label) in zip(axes, metrics):
        data_col = data_long[["그룹", col]].dropna()
        order = ["하위 25%", "상위 25%"]
        sns.boxplot(data=data_col, x="그룹", y=col,
                    order=order,
                    palette=colors,
                    width=0.5,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="#1D3557", markersize=6),
                    ax=ax)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.4)
        # 그룹별 개수 annot
        for i, grp in enumerate(order):
            n = len(data_col[data_col["그룹"] == grp])
            mean_val = data_col[data_col["그룹"] == grp][col].mean()
            ax.text(i, ax.get_ylim()[1] * 0.97,
                    f"n={n}\n평균={mean_val:.1f}",
                    ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc=colors[grp], alpha=0.25, ec=colors[grp]))

    fig.suptitle("생중량 상위 25% vs 하위 25% — 생육 지표 비교",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_상위하위_비교_상자그림.png"))
    plt.close(fig)

    # --- 그룹별 요약 막대 그래프 (평균 비교) ---
    summary = []
    for col, label in metrics:
        summary.append({
            "지표": label,
            "하위 25% 평균": bottom[col].mean(),
            "상위 25% 평균": top[col].mean(),
            "차이": top[col].mean() - bottom[col].mean(),
        })
    summary_df = pd.DataFrame(summary)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary_df))
    w = 0.35
    ax2.bar(x - w/2, summary_df["하위 25% 평균"], w,
            label="하위 25%", color="#D62828", alpha=0.85, edgecolor="#7B1818")
    ax2.bar(x + w/2, summary_df["상위 25% 평균"], w,
            label="상위 25%", color="#2E86AB", alpha=0.85, edgecolor="#1B4965")
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df["지표"], fontsize=10)
    ax2.set_ylabel("평균값")
    ax2.set_title("생중량 상위 25% vs 하위 25% — 평균 생육 지표 비교",
                  fontsize=13, fontweight="bold")
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for i, row in summary_df.iterrows():
        ax2.text(i - w/2, row["하위 25% 평균"] + 0.5,
                 f"{row['하위 25% 평균']:.1f}", ha="center", va="bottom", fontsize=8)
        ax2.text(i + w/2, row["상위 25% 평균"] + 0.5,
                 f"{row['상위 25% 평균']:.1f}", ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "03_상위하위_비교_막대그래프.png"))
    plt.close(fig2)

    return {
        "q25": q25,
        "q75": q75,
        "n_bottom": len(bottom),
        "n_top": len(top),
        "summary": summary_df,
        "top_df": top,
        "bottom_df": bottom,
    }


# ============================================================
# 대시보드 요약 HTML 저장 (간략)
# ============================================================
def save_dashboard_summary(dist_result, corr_result, q_result):
    """세 분석 결과를 하나의 HTML 요약 파일로 저장."""
    html_path = os.path.join(OUTPUT_DIR, "대시보드_요약.html")

    rows = ""
    for r in dist_result:
        rows += f"<tr><td>{r}</td><td>{dist_result[r]}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>극지식물 생육 차이 분석 대시보드</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; color: #1D3557; }}
  h1 {{ border-bottom: 3px solid #2E86AB; padding-bottom: 0.3em; }}
  h2 {{ color: #2E86AB; margin-top: 2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #A8DADC; padding: 0.4em 0.7em; text-align: left; }}
  th {{ background: #E8F4F8; }}
  .img {{ max-width: 100%; margin: 0.5em 0; border: 1px solid #ccc; }}
</style>
</head>
<body>
<h1>극지식물 개체별 생육 차이 분석 대시보드</h1>
<p>생성 시각: 분석 완료 시점</p>
<p>아래 3개 PNG 차트가 <code>output/</code> 폴더에 저장되었습니다.</p>

<h2>분석 1. 개체별 생중량 분포</h2>
<table>
  <tr><th>지표</th><th>값</th></tr>
  {rows}
</table>
<img class="img" src="01_생중량_분포_히스토그램.png">
<img class="img" src="01_생중량_분포_상자그림.png">

<h2>분석 2. 잎 수·지상부 길이 ↔ 생중량 상관</h2>
<table>
  <tr><th>관계</th><th>Pearson r</th><th>p-value</th></tr>
  <tr><td>잎 수 ↔ 생중량</td>
      <td>{corr_result['r_leaf']:+.4f}</td>
      <td>{corr_result['p_leaf']:.4e}</td></tr>
  <tr><td>지상부 길이 ↔ 생중량</td>
      <td>{corr_result['r_shoot']:+.4f}</td>
      <td>{corr_result['p_shoot']:.4e}</td></tr>
</table>
<img class="img" src="02_상관계수_산점도.png">

<h2>분석 3. 생중량 상위 25% vs 하위 25% 비교</h2>
<p>하위 25% 경계: {q_result['q25']:.2f} g | 상위 25% 경계: {q_result['q75']:.2f} g</p>
<p>하위 그룹: {q_result['n_bottom']}개체 | 상위 그룹: {q_result['n_top']}개체</p>
<img class="img" src="03_상위하위_비교_상자그림.png">
<img class="img" src="03_상위하위_비교_막대그래프.png">

</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[대시보드] HTML 요약 저장 완료: {html_path}")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 56)
    print("  극지식물 개체별 생육 차이 분석 대시보드")
    print("=" * 56)

    # --- 데이터 로드 ---
    if not os.path.exists(GROWTH_CSV):
        print(f"\n[오류] 생육 데이터 파일을 찾을 수 없습니다:\n  {GROWTH_CSV}")
        raise SystemExit(1)
    if not os.path.exists(ENV_CSV):
        print(f"\n[주의] 환경 데이터 파일을 찾을 수 없습니다:\n  {ENV_CSV}")
        print("  환경 데이터 없이 생육 분석만 진행합니다.")
        env_df = None
    else:
        env_df = load_environment_data(ENV_CSV)

    growth_df = load_growth_data(GROWTH_CSV)

    if env_df is not None:
        print("\n[환경데이터 요약 (일평균)]")
        daily = env_df.set_index("time").resample("D").mean(numeric_only=True)
        print(daily.round(2).to_string())
        print()

    # --- 분석 실행 ---
    dist_result = analysis_distribution(growth_df)
    corr_result = analysis_correlation(growth_df)
    q_result = analysis_quartile_comparison(growth_df)

    # --- HTML 대시보드 요약 저장 ---
    save_dashboard_summary(dist_result, corr_result, q_result)

    print(f"\n[완료] 모든 차트가 {OUTPUT_DIR}/ 에 저장되었습니다.")
    print("  - 01_생중량_분포_히스토그램.png")
    print("  - 01_생중량_분포_상자그림.png")
    print("  - 02_상관계수_산점도.png")
    print("  - 03_상위하위_비교_상자그림.png")
    print("  - 03_상위하위_비교_막대그래프.png")
    print("  - 대시보드_요약.html")

