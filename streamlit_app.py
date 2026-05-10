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
            return "y축에 대하여 대칭"
        if sp.simplify(expr.subs(x, -x) + expr) == 0:
            return "원점에 대하여 대칭"
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


def sign_symbol(value):
    try:
        if value == 0:
            return "0"
        if value > 0:
            return "+"
        if value < 0:
            return "-"
    except Exception:
        pass
    return "?"


def format_point_label(point, extremum_xs, inflection_xs, second_derivative, expr, x):
    if point in extremum_xs:
        sec_val = None
        try:
            sec_val = float(second_derivative.subs(x, point))
        except Exception:
            pass
        y_val = None
        try:
            y_val = expr.subs(x, point)
            y_val_simplified = sp.nsimplify(y_val, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)])
        except Exception:
            y_val_simplified = None
        if sec_val is not None:
            if sec_val > 0:
                label = "극소"
            elif sec_val < 0:
                label = "극대"
            else:
                label = "극값"
        else:
            label = "극값"
        if y_val_simplified is not None:
            return r"$%s$ (%s)" % (sp.latex(y_val_simplified), label)
        return label
    if point in inflection_xs:
        y_val = None
        try:
            y_val = expr.subs(x, point)
            y_val_simplified = sp.nsimplify(y_val, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)])
        except Exception:
            y_val_simplified = None
        if y_val_simplified is not None:
            return r"$%s$ (변곡점)" % sp.latex(y_val_simplified)
        return "변곡점"
    return ""


def build_variation_table(expr, x, derivative, second_derivative, derivative_fn, second_derivative_fn, extremum_xs, inflection_xs, x_min, x_max):
    critical_points = sorted(set([float(p) for p in extremum_xs + inflection_xs if x_min < float(p) < x_max]))
    values = [-np.inf] + critical_points + [np.inf]
    headers = ["..."]
    for point in critical_points:
        headers.append(r"$%s$" % sp.latex(sp.nsimplify(point, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)])))
        headers.append("...")

    y1_cells = []
    y2_cells = []
    y_cells = []
    y_values = []

    expr_fn = sp.lambdify(x, expr, modules=["numpy"])

    def sample_value(fn, low, high):
        if low == -np.inf and high == np.inf:
            mid = 0.0
        elif low == -np.inf:
            mid = high - 1.0
        elif high == np.inf:
            mid = low + 1.0
        else:
            mid = (low + high) / 2.0
        try:
            value = fn(mid)
            if not np.isfinite(value):
                return None
            return value
        except Exception:
            return None

    for i in range(len(values) - 1):
        low, high = values[i], values[i + 1]
        y1_mid = sample_value(derivative_fn, low, high)
        y2_mid = sample_value(second_derivative_fn, low, high)
        y_mid = sample_value(expr_fn, low, high)
        y1_cells.append(sign_symbol(y1_mid) if y1_mid is not None else "?")
        y2_cells.append(sign_symbol(y2_mid) if y2_mid is not None else "?")
        y_values.append(format_value(y_mid) if y_mid is not None else "?")
        if y1_mid is None:
            y_cells.append("?")
        elif y1_mid > 0:
            y_cells.append("↗")
        elif y1_mid < 0:
            y_cells.append("↘")
        else:
            y_cells.append("→")
        if i < len(critical_points):
            point = critical_points[i]
            y1_point = sign_symbol(derivative_fn(point))
            y2_point = sign_symbol(second_derivative_fn(point))
            y_point = expr.subs(x, point)
            y1_cells.append("0" if abs(derivative_fn(point)) < 1e-6 else y1_point)
            y2_cells.append("0" if abs(second_derivative_fn(point)) < 1e-6 else y2_point)
            y_values.append(format_value(y_point))
            y_cells.append(format_point_label(point, extremum_xs, inflection_xs, second_derivative, expr, x))

    table_lines = []
    table_lines.append("| x | " + " | ".join(headers) + " |")
    table_lines.append("| --- " + " | ---" * len(headers) + " |")
    table_lines.append("| y' | " + " | ".join(y1_cells) + " |")
    table_lines.append("| y'' | " + " | ".join(y2_cells) + " |")
    table_lines.append("| y | " + " | ".join(y_cells) + " |")
    table_lines.append("| f(x) | " + " | ".join(y_values) + " |")
    return "\n".join(table_lines)


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

            domain = get_domain(expr, x)
            value_range = get_range(expr, x, domain)
            symmetry = get_symmetry(expr, x)
            period = get_period(expr, x)
            try:
                x_intercepts = sp.solveset(sp.simplify(expr), x, domain=domain)
                if x_intercepts.is_FiniteSet:
                    x_intercepts = [sp.nsimplify(pt, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)]) for pt in x_intercepts]
                else:
                    x_intercepts = []
            except Exception:
                x_intercepts = []
            try:
                y_intercept = expr.subs(x, 0)
            except Exception:
                y_intercept = None
            derivative_fn = sp.lambdify(x, derivative, modules=["numpy"])
            second_derivative_fn = sp.lambdify(x, second_derivative, modules=["numpy"])
            derivative_chart = make_sign_chart(derivative_fn, extremum_xs, x_min, x_max)
            convexity_chart = make_sign_chart(second_derivative_fn, inflection_xs, x_min, x_max)
            horiz_asymptotes, vert_asymptotes = get_asymptotes(expr, x)

            st.write("---")
            st.write("### 상세 정보")
            st.markdown(f"**1. 정의역과 치역**  \n- 정의역: {format_interval(domain)}  \n- 치역: {format_interval(value_range)}")
            x_int_text = ", ".join([r"$%s$" % sp.latex(pt) for pt in x_intercepts]) if x_intercepts else "없음"
            y_int_text = r"$%s$" % sp.latex(y_intercept) if y_intercept is not None else "없음"
            st.markdown(f"**2. 곡선과 좌표축의 교점**  \n- x절편: {x_int_text}  \n- y절편: {y_int_text}")
            period_text = r"$%s$" % sp.latex(period) if period is not None else "없음"
            st.markdown(f"**3. 곡선의 대칭성과 주기**  \n- 대칭성: {symmetry}  \n- 주기: {period_text}")

            st.write("**4. 함수의 증가/감소, 극대와 극소, 곡선의 볼록성과 변곡점 (증감표)**")
            variation_table = build_variation_table(expr, x, derivative, second_derivative, derivative_fn, second_derivative_fn, extremum_xs, inflection_xs, x_min, x_max)
            st.markdown(variation_table)

            st.write("**5. 극한과 점근선**")
            st.markdown(f"- 우극한: $%s$  \n- 좌극한: $%s$" % (sp.latex(sp.limit(expr, x, sp.oo)), sp.latex(sp.limit(expr, x, -sp.oo))))
            if horiz_asymptotes:
                st.markdown(f"- 수평점근선: %s" % ", ".join([r"$y=%s$" % sp.latex(h) for h in horiz_asymptotes]))
            else:
                st.markdown("- 수평점근선: 없음")
            if vert_asymptotes:
                st.markdown(f"- 수직점근선: %s" % ", ".join([r"$x=%s$" % sp.latex(v) for v in vert_asymptotes]))
            else:
                st.markdown("- 수직점근선: 없음")
    except SympifyError:
        st.error("입력한 함수식이 잘못되었습니다. 올바른 일변수 함수식을 입력해 주세요.")
    except Exception as exc:
        st.error(f"그래프를 그리는 동안 오류가 발생했습니다: {exc}")
else:
    st.info("먼저 함수식을 입력해 주세요.")
