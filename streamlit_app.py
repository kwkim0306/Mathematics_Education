import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import sympy as sp
import streamlit as st
from pathlib import Path
from sympy import SympifyError
from sympy.calculus.util import continuous_domain, function_range
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

NUM_POINTS = 500


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


def simplify_value(value):
    if isinstance(value, sp.Expr):
        try:
            simplified = sp.simplify(value)
            return simplified
        except Exception:
            return value

    try:
        return sp.nsimplify(value, [sp.pi, sp.E,
                                    sp.sqrt(2), sp.sqrt(3), sp.sqrt(5), sp.sqrt(6), sp.sqrt(7), sp.sqrt(10),
                                    sp.sqrt(11), sp.sqrt(13), sp.sqrt(17), sp.sqrt(19)])
    except Exception:
        return value


def format_value(value):
    exact = simplify_value(value)
    if isinstance(exact, sp.Expr) and exact.is_real:
        try:
            return r"$%s$" % sp.latex(exact)
        except Exception:
            pass
    try:
        float_val = float(exact)
        if abs(float_val - round(float_val)) < 1e-8:
            return str(int(round(float_val)))
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def find_symbolic_roots(eq, x, lower, upper):
    try:
        sol = sp.solveset(eq, x, domain=sp.Interval(lower, upper))
        if sol.is_FiniteSet:
            roots = []
            for item in sol:
                if item.is_real:
                    roots.append(item)
            return roots
    except Exception:
        pass
    return []


def get_domain(expr, x):
    try:
        return continuous_domain(expr, x, sp.S.Reals)
    except Exception:
        return sp.S.Reals


def get_range(expr, x, dom):
    try:
        return function_range(expr, x, dom)
    except Exception:
        return sp.S.Reals


def format_interval(value):
    if isinstance(value, (sp.Interval, sp.Union, sp.FiniteSet, sp.Reals)):
        return r"$%s$" % sp.latex(value)
    return str(value)


def make_sign_chart(fn, critical_points, x_min, x_max):
    pts = [x_min] + sorted([float(p) for p in critical_points if x_min < float(p) < x_max]) + [x_max]
    intervals = []
    for a, b in zip(pts[:-1], pts[1:]):
        mid = (a + b) / 2
        try:
            val = fn(mid)
            if not np.isfinite(val):
                sign = "불명"
            elif val > 0:
                sign = "증가"
            elif val < 0:
                sign = "감소"
            else:
                sign = "정지"
        except Exception:
            sign = "불명"
        intervals.append((a, b, sign))
    return intervals


def classify_extremum(second_derivative, x, point):
    try:
        sec_val = float(second_derivative.subs(x, point))
        if sec_val > 0:
            return "극소"
        if sec_val < 0:
            return "극대"
    except Exception:
        pass
    return "판별 불가"


def get_symmetry(expr, x):
    try:
        if sp.simplify(expr.subs(x, -x) - expr) == 0:
            return "짝함수"
        if sp.simplify(expr.subs(x, -x) + expr) == 0:
            return "홀함수"
    except Exception:
        pass
    return "대칭 없음"


def get_period(expr, x):
    try:
        period = sp.periodicity(expr, x)
        if period is not None:
            return period
    except Exception:
        pass
    return None


def get_asymptotes(expr, x):
    horizontals = []
    verticals = []
    try:
        lim_plus = sp.limit(expr, x, sp.oo)
        lim_minus = sp.limit(expr, x, -sp.oo)
        if lim_plus.is_real:
            horizontals.append(lim_plus)
        if lim_minus.is_real and lim_minus != lim_plus:
            horizontals.append(lim_minus)
    except Exception:
        pass
    try:
        poles = sp.singularities(expr, x)
        for pole in poles:
            if pole.is_real:
                verticals.append(pole)
    except Exception:
        pass
    return horizontals, verticals


transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

sympy_locals = {
    "x": sp.symbols("x"),
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


@st.cache_data
def analyze_expression(expr_input, x_min, x_max):
    x = sympy_locals["x"]
    expr = parse_expr(expr_input, transformations=transformations, local_dict=sympy_locals)
    derivative = sp.diff(expr, x)
    second_derivative = sp.diff(expr, x, 2)

    sample_xs = np.linspace(x_min, x_max, NUM_POINTS)
    extremum_xs = find_symbolic_roots(derivative, x, x_min, x_max)
    if not extremum_xs:
        dfunc = sp.lambdify(x, derivative, modules=["numpy"])
        extremum_xs = find_zero_crossings(sample_xs, dfunc(sample_xs))
        extremum_xs = [sp.nsimplify(x0, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)]) for x0 in extremum_xs]

    inflection_xs = find_symbolic_roots(second_derivative, x, x_min, x_max)
    if not inflection_xs:
        dd_func = sp.lambdify(x, second_derivative, modules=["numpy"])
        inflection_xs = find_zero_crossings(sample_xs, dd_func(sample_xs))
        inflection_xs = [sp.nsimplify(x0, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)]) for x0 in inflection_xs]

    return expr, derivative, second_derivative, extremum_xs, inflection_xs


base_dir = Path(__file__).resolve().parent
font_path = base_dir / "fonts" / "NotoSansKR-Regular.ttf"
font_prop = fm.FontProperties()
font_name = "DejaVu Sans"
if font_path.exists():
    try:
        fm.fontManager.addfont(str(font_path))
        font_prop = fm.FontProperties(fname=str(font_path))
        font_name = font_prop.get_name()
    except Exception:
        font_prop = fm.FontProperties()
        font_name = "DejaVu Sans"
plt.rcParams.update({
    "font.family": font_name,
    "font.sans-serif": [font_name, "DejaVu Sans", "Arial", "Liberation Sans", "Nimbus Sans L"],
    "font.weight": "normal",
    "text.usetex": False
})

st.set_page_config(page_title="일변수 함수 그래프", layout="wide")

st.title("📈 일변수 함수 그래프 그리기")

num_points = 500

expr_input = st.text_input("함수식 f(x)", value="sin(x)")
show_special_points = st.checkbox("극점/변곡점 표시", value=False)
show_axis_x = st.checkbox("x축에 특수점 x좌표 표시", value=False)
show_axis_y = st.checkbox("y축에 특수점 y좌표 표시", value=False)

x_min = -10.0
x_max = 10.0

if expr_input:
    try:
        expr, derivative, second_derivative, extremum_xs, inflection_xs = analyze_expression(expr_input, x_min, x_max)
        x = sympy_locals["x"]
        func = sp.lambdify(x, expr, modules=["numpy"])
        xs = np.linspace(x_min, x_max, num_points)
        ys = func(xs)

        mask = np.isfinite(ys)
        if np.count_nonzero(mask) == 0:
            st.warning("유효한 함수 값을 계산할 수 없습니다. 식과 범위를 확인하세요.")
        else:
            fig, ax = plt.subplots()
            ax.plot(xs[mask], ys[mask], color="#1f77b4", linewidth=2)
            ax.set_xlabel(r"$x$", fontsize=12, fontproperties=font_prop)
            ax.set_ylabel(r"$f(x)$", fontsize=12, fontproperties=font_prop)
            ax.set_title(r"$f(x) = %s$" % sp.latex(expr), fontsize=18, fontproperties=font_prop)

            if show_special_points:
                try:
                    if extremum_xs:
                        y_ext = []
                        for x0 in extremum_xs:
                            try:
                                y_ext.append(float(expr.subs(x, x0)))
                            except Exception:
                                y_ext.append(float(func(float(x0))))
                        x_vals = [float(xx) for xx in extremum_xs]
                        ax.scatter(x_vals, y_ext, color="red", s=3, zorder=4, label="극점")
                        if show_axis_x:
                            ax.scatter(x_vals, np.zeros_like(x_vals), color="red", s=2, zorder=4, alpha=0.8)
                            for x0 in extremum_xs:
                                ax.text(float(x0), 0.05, format_value(x0), color="red", fontsize=6, ha="center", va="bottom", fontproperties=font_prop)
                        if show_axis_y:
                            ax.scatter(np.zeros_like(y_ext), y_ext, color="red", s=2, zorder=4, alpha=0.8)
                            for y0 in y_ext:
                                ax.text(0.05, y0, format_value(y0), color="red", fontsize=6, ha="left", va="center", fontproperties=font_prop)
                    if inflection_xs:
                        y_inf = []
                        for x0 in inflection_xs:
                            try:
                                y_inf.append(float(expr.subs(x, x0)))
                            except Exception:
                                y_inf.append(float(func(float(x0))))
                        x_vals = [float(xx) for xx in inflection_xs]
                        ax.scatter(x_vals, y_inf, color="green", s=3, zorder=4, label="변곡점")
                        if show_axis_x:
                            ax.scatter(x_vals, np.zeros_like(x_vals), color="green", s=2, zorder=4, alpha=0.8)
                            for x0 in inflection_xs:
                                ax.text(float(x0), 0.05, format_value(x0), color="green", fontsize=6, ha="center", va="bottom", fontproperties=font_prop)
                        if show_axis_y:
                            ax.scatter(np.zeros_like(y_inf), y_inf, color="green", s=2, zorder=4, alpha=0.8)
                            for y0 in y_inf:
                                ax.text(0.05, y0, format_value(y0), color="green", fontsize=6, ha="left", va="center", fontproperties=font_prop)
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

            if show_special_points:
                st.write("### 도함수 정보")
                st.latex(r"f'(x) = " + sp.latex(derivative))
                st.latex(r"f''(x) = " + sp.latex(second_derivative))

            st.write("---")
            st.write("### 상세 정보")
            st.write(f"- 계산된 유효 점 개수: {np.count_nonzero(mask)} / {num_points}")
    except SympifyError:
        st.error("입력한 함수식이 잘못되었습니다. 올바른 일변수 함수식을 입력해 주세요.")
    except Exception as exc:
        st.error(f"그래프를 그리는 동안 오류가 발생했습니다: {exc}")
else:
    st.info("먼저 함수식을 입력해 주세요.")
