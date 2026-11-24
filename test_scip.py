import streamlit as st
from pyscipopt import Model, quicksum

st.title("Test SCIP - CL02")

if st.button("Tester SCIP"):
    try:
        m = Model("Test_SCIP")
        m.hideOutput()
        x = [m.addVar(vtype="B", name=f"x_{i}") for i in range(2000)]

        for i in range(2000):
            m.addCons(x[i] <= 1)
            m.addCons(x[i] >= 0)

        m.setObjective(quicksum(x), "maximize")
        m.setParam("limits/time", 10)
        m.optimize()

        st.write("Variables :", m.getNVars(transformed=False))
        st.write("Contraintes :", m.getNConss(transformed=False))
        st.write("Statut :", str(m.getStatus()))
        st.write("Objectif :", m.getObjVal())

        if str(m.getStatus()) == "optimal":
            st.success("SCIP fonctionne sans limitation de licence !")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
