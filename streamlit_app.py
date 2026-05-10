import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import streamlit as st
from sympy import SympifyError


def find_zero_crossings(x_values, y_values):
    roots = []
    for i in range(len(y_values) - 1):
        y1, y2 = y_values[i], y_values[i + 1]
        if not np.isfinite(y1) or not np.isfinite(y2):
            continue
        if y1 == 0:
            roots.append(x_values[i])
        elif y1 * y2 < 0:
            root = x_values[i] - y1 * (x_values[i + 1] - x_values[i]) / (y2 - y1)
            roots.append(root)

    filtered = []
    for r in roots:
        if not any(abs(r - existing) < 1e-3 for existing in filtered):
            filtered.append(r)
    return [float(np.round(r, 4)) for r in filtered]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Nimbus Sans L"],
    "font.weight": "normal",
    "mathtext.fontset": "dejavusans",
    "mathtext.rm": "dejavusans",
    "mathtext.it": "dejavusans:italic",
    "mathtext.bf": "dejavusans:bold",
    "text.usetex": False
})

st.set_page_config(page_title="일변수 함수 그래프", layout="wide")

st.title("📈 일변수 함수 그래프 그리기")

num_points = 500

expr_input = st.text_input("함수식 f(x)", value="sin(x)")
show_special_points = st.checkbox("극점/변곡점 표시", value=False)

x_min = -10.0
x_max = 10.0

if expr_input:
    try:
        x = sp.symbols("x")
        expr = sp.sympify(expr_input, locals={"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                                              "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
                                              "pi": sp.pi, "E": sp.E})

        func = sp.lambdify(x, expr, modules=["numpy"])
        xs = np.linspace(x_min, x_max, num_points)
        ys = func(xs)

        mask = np.isfinite(ys)
        if np.count_nonzero(mask) == 0:
            st.warning("유효한 함수 값을 계산할 수 없습니다. 식과 범위를 확인하세요.")
        else:
            fig, ax = plt.subplots()
            ax.plot(xs[mask], ys[mask], color="#1f77b4", linewidth=2)
            ax.set_xlabel("x", fontsize=12, fontfamily="serif")
            ax.set_ylabel("f(x)", fontsize=12, fontfamily="serif")
            ax.set_title(f"f(x) = {expr_input}", fontdict={"fontsize": 18, "fontfamily": "serif", "fontweight": "light"})

            if show_special_points:
                try:
                    derivative = sp.diff(expr, x)
                    second_derivative = sp.diff(expr, x, 2)
                    dfunc = sp.lambdify(x, derivative, modules=["numpy"])
                    dd_func = sp.lambdify(x, second_derivative, modules=["numpy"])
                    d_vals = dfunc(xs)
                    dd_vals = dd_func(xs)

                    extremum_xs = find_zero_crossings(xs, d_vals)
                    inflection_xs = find_zero_crossings(xs, dd_vals)

                    if extremum_xs:
                        y_ext = [float(func(x0)) for x0 in extremum_xs]
                        ax.scatter(extremum_xs, y_ext, color="red", s=50, zorder=5, label="Extremum")
                        for x0, y0 in zip(extremum_xs, y_ext):
                            ax.annotate("E", xy=(x0, y0), xytext=(6, 6), textcoords="offset points",
                                        color="red", fontsize=10, fontfamily="sans-serif")
                    if inflection_xs:
                        y_inf = [float(func(x0)) for x0 in inflection_xs]
                        ax.scatter(inflection_xs, y_inf, color="green", s=50, zorder=5, label="Inflection")
                        for x0, y0 in zip(inflection_xs, y_inf):
                            ax.annotate("I", xy=(x0, y0), xytext=(6, -12), textcoords="offset points",
                                        color="green", fontsize=10, fontfamily="sans-serif")
                    if extremum_xs or inflection_xs:
                        ax.legend(loc="upper right", fontsize=10)
                except Exception:
                    st.warning("극점/변곡점 표시 중 오류가 발생했습니다.")

            ax.axhline(0, color="black", linewidth=1)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("black")
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["left"].set_linewidth(1)

            st.pyplot(fig)

            st.write("---")
            st.write("### 상세 정보")
            st.write(f"- 계산된 유효 점 개수: {np.count_nonzero(mask)} / {num_points}")
    except SympifyError:
        st.error("입력한 함수식이 잘못되었습니다. 올바른 일변수 함수식을 입력해 주세요.")
    except Exception as exc:
        st.error(f"그래프를 그리는 동안 오류가 발생했습니다: {exc}")
else:
    st.info("먼저 함수식을 입력해 주세요.")
