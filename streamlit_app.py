import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import streamlit as st
from sympy import SympifyError

st.set_page_config(page_title="일변수 함수 그래프", layout="wide")

st.title("📈 일변수 함수 그래프 그리기")
st.write("함수식을 입력하면 해당 함수의 그래프를 그려줍니다. 예: `x**2`, `sin(x)`, `exp(-x)`, `log(x+1)`")

with st.sidebar:
    st.header("설정")
    num_points = st.slider("표시할 점 개수", min_value=100, max_value=2000, value=500, step=100)
    show_expression = st.checkbox("입력식 표시", value=True)

expr_input = st.text_input("함수식 f(x)", value="sin(x)")

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

        if show_expression:
            st.markdown(f"**입력식:** `{expr_input}`")
            st.markdown(f"**변환된 식:** `{str(expr)}`")

        mask = np.isfinite(ys)
        if np.count_nonzero(mask) == 0:
            st.warning("유효한 함수 값을 계산할 수 없습니다. 식과 범위를 확인하세요.")
        else:
            fig, ax = plt.subplots()
            ax.plot(xs[mask], ys[mask], color="#1f77b4", linewidth=2)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title(f"f(x) = {expr_input}")
            ax.grid(True, alpha=0.3)
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
