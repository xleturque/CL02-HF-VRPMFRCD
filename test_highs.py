import streamlit as st
import highspy

st.title("Test HiGHS - CL02")

if st.button("Tester HiGHS"):
    try:
        m = highspy.Highs()
        m.setOptionValue("output_flag", False)
        m.setOptionValue("time_limit", 10.0)

        x = [
            m.addVariable(lb=0, ub=1, obj=1,
                          type=highspy.HighsVarType.kInteger)
            for i in range(2000)
        ]

        for i in range(2000):
            m.addConstr(x[i] <= 1)
            m.addConstr(x[i] >= 0)

        m.setMaximize()
        m.run()

        statut = m.getModelStatus()

        st.write("Version HiGHS :", m.version())
        st.write("Variables :", m.getNumCol())
        st.write("Contraintes :", m.getNumRow())
        st.write("Statut :", m.modelStatusToString(statut))
        st.write("Objectif :", m.getObjectiveValue())
        st.write("Temps de calcul :", m.getRunTime())

        if statut == highspy.HighsModelStatus.kOptimal:
            st.success("HiGHS fonctionne sans limitation de licence !")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
