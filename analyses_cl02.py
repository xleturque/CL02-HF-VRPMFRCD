"""Analyses complementaires du dashboard CL02.

Module independant de l'interface existante : aucune modification du solveur.
- Analyse de robustesse multi-executions sur une instance identique.
- Chronologie des cinq processus (Gantt selon les temps du modele).
- Comparaison solution initiale / solution optimisee.

Integration dans app.py, apres le contenu des onglets existants :
    from analyses_cl02 import afficher_analyses
    afficher_analyses(execution)
"""

from __future__ import annotations

import random
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable, Optional

import pandas as pd
import plotly.graph_objects as go

from hf_vrpmfrcd_metaheuristics import Problem, quick_params, solve

COLORS_METHOD = {"SA": "#F28E2B", "SAVND": "#59A14F", "ANSA": "#4E79A7"}
COLORS_PHASE = {"P1": "#E15759", "P2": "#F28E2B", "P3": "#D4A017", "P4": "#269DAB", "P5": "#9467BD"}


def _facteur_minutes(execution: dict) -> float:
    from hf_vrpmfrcd_instances import UNITES_TEMPS
    return float(UNITES_TEMPS[execution["set_id"]][1])


def analyser_gain_initial(execution: dict, seed_recherche: int = 0) -> list[dict]:
    """Reference brute de initial_solution(), avant reparation/recuit.

    Les executions ordinaires de l'app utilisent seed_recherche=0. On compare
    l'objectif penalise meme si la solution initiale est infaisable ; le gain
    de cout de transport n'est montre que si les deux solutions sont faisables.
    """
    prob = Problem(**execution["donnees"])
    a0, _ = prob.initial_solution(random.Random(seed_recherche))
    debut = prob.evaluate(a0, detail=True)
    lignes = []
    for resultat in execution["resultats"]:
        fin = resultat["details"]
        obj0, objf = float(debut["obj"]), float(fin["obj"])
        cout0, coutf = float(debut["cost"]), float(fin["cost"])
        comparable = bool(debut["feasible"] and fin["feasible"])
        historique = resultat.get("historique", [])
        entree_recuit = float(historique[0][1]) if historique else None
        lignes.append({
            "Méthode": resultat["methode"],
            "Initial faisable": bool(debut["feasible"]),
            "Final faisable": bool(fin["feasible"]),
            "Coût initial": cout0,
            "Coût final": coutf,
            "Objectif initial": obj0,
            "Objectif final": objf,
            "Objectif entrée recuit": entree_recuit,
            "Gain recherche (%)": (100.0 * (entree_recuit - objf) / entree_recuit)
                                 if entree_recuit is not None and entree_recuit > 0 else None,
            "Gain objectif (%)": 100.0 * (obj0 - objf) / obj0 if obj0 > 0 else None,
            "Gain transport (%)": 100.0 * (cout0 - coutf) / cout0 if comparable and cout0 > 0 else None,
        })
    return lignes


def construire_chronologie(execution: dict, resultat: dict) -> list[dict]:
    """Reconstitue un diagramme temporel a partir des tau du solveur.

    P1 : toutes les tournées commencent a t=0.
    P2 : depart quand le cross-dock d'origine a fini P1.
    P3 : depart apres tau2 et la preparation zeta.
    P4 : depart quand le cross-dock d'origine a fini P3.
    P5 : depart apres tau4 et la preparation zeta.

    Les camions d'une meme liaison de transfert sont representes par une barre
    agregee (pas une planification individuelle des vehicules).
    """
    inst = execution["donnees"]
    details = resultat["details"]
    f = _facteur_minutes(execution)
    tau = details["tau"]
    zeta = float(inst["zeta"])
    nD = int(inst["nD"])
    lignes = []

    def ajouter(phase: str, activite: str, debut: float, duree: float, dock: str, type_barre="operation"):
        if duree <= 0:
            return
        lignes.append({
            "Phase": phase,
            "Activité": activite,
            "Début (min)": round(debut * f, 6),
            "Fin (min)": round((debut + duree) * f, 6),
            "Durée (min)": round(duree * f, 6),
            "Cross-dock / liaison": dock,
            "Type": type_barre,
        })

    for phase in ("P1", "P3", "P5"):
        for idx, (dock, noeuds, veh, cout, duree) in enumerate(details[phase], 1):
            if phase == "P1":
                debut = 0.0
            elif phase == "P3":
                debut = float(tau["tau2"][dock - 1]) + zeta
            else:
                debut = float(tau["tau4"][dock - 1]) + zeta
            prefix = "C" if phase == "P3" else "S"
            sommets = ", ".join(f"{prefix}{noeud - nD}" for noeud in noeuds)
            ajouter(phase, f"{phase} · D{dock} · V{veh} · tournée {idx} ({sommets})", debut,
                    float(duree), f"D{dock}")

    for phase, tau_origine, camions in (
        ("P2", "tau1", details.get("trucks2", {})),
        ("P4", "tau3", details.get("trucks4", {})),
    ):
        for (origine, destination), nombre in sorted(camions.items()):
            if not nombre:
                continue
            debut = float(tau[tau_origine][origine - 1])
            duree = float(inst["tD"][origine - 1][destination - 1])
            ajouter(phase, f"{phase} · D{origine} → D{destination} · {nombre} camion(s)",
                    debut, duree, f"D{origine} → D{destination}")

    # Les delais de traitement zeta sont une activite distincte, neutre.
    for phase, cle, phase_preparation in (("P3", "tau2", "Préparation P3"), ("P5", "tau4", "Préparation P5")):
        docks_actifs = sorted({tour[0] for tour in details[phase]})
        for dock in docks_actifs:
            ajouter(phase, f"{phase_preparation} · D{dock}", float(tau[cle][dock - 1]),
                    zeta, f"D{dock}", type_barre="preparation")

    ordre = {p: k for k, p in enumerate(COLORS_PHASE)}
    lignes.sort(key=lambda a: (ordre[a["Phase"]], a["Début (min)"], a["Activité"]))
    return lignes


def executer_repetitions(
    execution: dict,
    methodes: list[str],
    repetitions: int = 5,
    secondes: float = 5.0,
    progression: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """Repete les methodes sur la MEME instance avec graines de recherche 0..n-1."""
    valides = {"SA", "SAVND", "ANSA"}
    if not methodes or any(m not in valides for m in methodes):
        raise ValueError("Veuillez choisir au moins une des methodes SA/SAVND/ANSA.")
    if not 2 <= repetitions <= 10:
        raise ValueError("Le nombre de repetitions doit etre compris entre 2 et 10.")
    if not 0.1 <= secondes <= 15:
        raise ValueError("Le budget par execution doit etre compris entre 0,1 et 15 secondes.")
    if repetitions * secondes * len(methodes) > 150:
        raise ValueError("Budget cumule trop eleve pour l'hebergement public (maximum 150 s).")

    prob = Problem(**execution["donnees"])
    lignes = []
    total = repetitions * len(methodes)
    n = 0
    # Par repetition, comparer les trois methodes avec la meme graine de recherche.
    for seed in range(repetitions):
        for methode in methodes:
            params = quick_params(prob, seed=seed, time_limit=float(secondes))
            debut = perf_counter()
            solution, score, hist, stats = solve(prob, methode.lower(), params)
            duree = perf_counter() - debut
            details = prob.evaluate(solution, detail=True)
            lignes.append({
                "Méthode": methode,
                "Graine recherche": seed,
                "Faisable": bool(details["feasible"]),
                "Coût transport": float(details["cost"]),
                "Objectif pénalisé": float(details["obj"]),
                "Calcul (s)": float(duree),
                "Itérations": int(stats.get("iterations", 0)),
            })
            n += 1
            if progression:
                progression(n, total, f"{methode}, essai {seed + 1}/{repetitions}")
    return lignes


def resumer_repetitions(lignes: list[dict]) -> list[dict]:
    """Stats de cout calculees UNIQUEMENT sur les repetitions faisables."""
    sortie = []
    for methode in ("SA", "SAVND", "ANSA"):
        essais = [r for r in lignes if r["Méthode"] == methode]
        if not essais:
            continue
        valides = [r["Coût transport"] for r in essais if r["Faisable"]]
        sortie.append({
            "Méthode": methode,
            "Répétitions": len(essais),
            "Faisabilité (%)": 100 * len(valides) / len(essais),
            "Coût moyen faisable": mean(valides) if valides else None,
            "Meilleur coût faisable": min(valides) if valides else None,
            "Écart-type coût": pstdev(valides) if len(valides) >= 2 else (0.0 if valides else None),
            "Temps moyen (s)": mean(r["Calcul (s)"] for r in essais),
            "Objectif moyen pénalisé": mean(r["Objectif pénalisé"] for r in essais),
        })
    return sortie


def afficher_analyses(execution: dict) -> None:
    """Affichage de 3 onglets supplementaires sans remplacer l'UI existante."""
    import streamlit as st
    if not execution or not execution.get("resultats"):
        return
    st.markdown("---")
    st.subheader("Analyses complémentaires")
    st.caption("Robustesse statistique, chronologie des opérations et efficacité de l'optimisation.")
    tab_exp, tab_temps, tab_gain = st.tabs([
        "Analyse expérimentale", "Chronologie P1–P5", "Gain de l'optimisation"
    ])

    with tab_exp:
        st.markdown("**Comparer la régularité des trois métaheuristiques**")
        st.caption(
            "L'instance (Set et graine de génération) reste fixe. Seule la graine interne "
            "du recuit change d'une répétition à l'autre. Les coûts statistiques ne "
            "concernent que les solutions déclarées faisables."
        )
        presentes = [r["methode"] for r in execution["resultats"]]
        with st.form("cl02_form_analyse_experimentale"):
            c1, c2, c3 = st.columns([1.1, 1, 1])
            with c1:
                methodes = st.multiselect("Méthodes", presentes, default=presentes,
                                         key="adv_methodes")
            with c2:
                repetitions = st.select_slider("Répétitions par méthode", options=[3, 5, 10],
                                               value=5, key="adv_repetitions")
            with c3:
                budget = st.select_slider("Budget par essai (s)", options=[3, 5, 10, 15],
                                          value=5, key="adv_budget")
            total = repetitions * len(methodes) * budget
            st.caption(f"Budget théorique maximal cumulé : {total} s (plafond : 150 s).")
            lancer = st.form_submit_button("Lancer les répétitions", type="primary")
        signature = (execution["set_id"], execution["graine"], tuple(methodes), repetitions, budget)
        if lancer:
            if not methodes:
                st.warning("Choisissez au moins une méthode.")
            elif total > 150:
                st.warning("Réduisez le nombre de répétitions, le temps ou le nombre de méthodes (maximum 150 s cumulées).")
            else:
                st.session_state.pop("cl02_repetitions_avancees", None)
                bar = st.progress(0.0, text="Calculs expérimentaux")
                texte = st.empty()
                def maj(n, maxi, msg):
                    bar.progress(n / maxi, text=f"{n}/{maxi} essais terminés")
                    texte.caption(msg)
                try:
                    essais = executer_repetitions(execution, methodes, repetitions, budget, maj)
                    st.session_state["cl02_repetitions_avancees"] = {"signature": signature, "essais": essais}
                    st.success("Campagne expérimentale terminée.")
                except Exception as exc:
                    st.error(f"Analyse interrompue : {exc}")
        session = st.session_state.get("cl02_repetitions_avancees")
        if session and session["signature"] == signature:
            essais = session["essais"]
            stats = resumer_repetitions(essais)
            st.dataframe(pd.DataFrame(stats).round(2), hide_index=True, use_container_width=True)
            fig = go.Figure()
            for nom in methodes:
                vals = [x["Coût transport"] for x in essais if x["Méthode"] == nom and x["Faisable"]]
                if vals:
                    fig.add_trace(go.Box(y=vals, name=nom, boxpoints="all", jitter=0.15,
                                         marker_color=COLORS_METHOD[nom], line_color=COLORS_METHOD[nom]))
            if fig.data:
                fig.update_layout(title="Dispersion des coûts des solutions faisables",
                                  yaxis_title="Coût transport", height=380, margin=dict(l=8, r=8, t=45, b=15),
                                  paper_bgcolor="white", plot_bgcolor="white", font_color="#202124")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune solution faisable : aucun coût n'apparaît dans le boxplot.")
            st.download_button("Exporter les essais (.csv)",
                               pd.DataFrame(essais).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               file_name=f"Analyse_Set{execution['set_id']}_graine{execution['graine']}.csv",
                               mime="text/csv")
            st.caption("L'écart-type est calculé sur les coûts faisables uniquement ; avec un seul essai faisable il vaut 0 par convention.")
        elif session:
            st.info("Les paramètres ont changé : relancez les répétitions pour obtenir des résultats correspondants.")
        else:
            st.info("Définissez une petite campagne expérimentale puis lancez-la ; les résultats de l'optimisation principale restent inchangés.")

    with tab_temps:
        resultats = execution["resultats"]
        methode = st.selectbox("Méthode à représenter", [r["methode"] for r in resultats],
                               key="adv_methode_gantt")
        r = next(x for x in resultats if x["methode"] == methode)
        d = r["details"]
        inst = execution["donnees"]
        f = _facteur_minutes(execution)
        tmax = float(inst["tmax"]) * f
        duree = float(d["duree_operations"]) * f
        marge = tmax - duree
        x1, x2, x3 = st.columns(3)
        x1.metric("Durée des opérations", f"{duree:,.1f} min")
        x2.metric("Horizon Tmax", f"{tmax:,.1f} min")
        x3.metric("Marge avant Tmax", f"{marge:,.1f} min", delta=f"{marge:+.1f} min")
        chronologie = construire_chronologie(execution, r)
        if not chronologie:
            st.info("Aucune opération à représenter.")
        else:
            limite = 80
            # Sur les grands Sets, repartir le quota entre les cinq phases.
            if len(chronologie) > limite:
                affichees = []
                for phase in COLORS_PHASE:
                    affichees.extend([x for x in chronologie if x["Phase"] == phase][:limite // 5])
                affichees = affichees[:limite]
                st.warning(f"Diagramme limité à {len(affichees)} barres (réparties entre les phases) pour rester lisible ; le tableau contient les {len(chronologie)} opérations.")
            else:
                affichees = chronologie
            fig = go.Figure()
            for phase, color in COLORS_PHASE.items():
                for typ, opacity in (("operation", 1.0), ("preparation", 0.36)):
                    lignes = [x for x in affichees if x["Phase"] == phase and x["Type"] == typ]
                    if not lignes:
                        continue
                    nom = phase if typ == "operation" else f"{phase} · préparation"
                    fig.add_trace(go.Bar(
                        x=[a["Durée (min)"] for a in lignes],
                        base=[a["Début (min)"] for a in lignes],
                        y=[a["Activité"] for a in lignes],
                        orientation="h", name=nom,
                        marker_color=color, opacity=opacity,
                        customdata=[[a["Début (min)"], a["Fin (min)"], a["Cross-dock / liaison"]] for a in lignes],
                        hovertemplate="%{y}<br>Début : %{customdata[0]:.1f} min<br>Fin : %{customdata[1]:.1f} min<br>%{customdata[2]}<extra></extra>",
                    ))
            fig.add_vline(x=tmax, line_dash="dash", line_color="#C9394E", line_width=2,
                          annotation_text="Tmax", annotation_position="top")
            fig.update_layout(
                barmode="overlay", height=min(1700, max(385, 26 * len(affichees) + 120)),
                xaxis_title="Temps écoulé depuis le début de P1 (min)",
                yaxis_title="", legend_title="Phase", margin=dict(l=10, r=12, t=28, b=20),
                paper_bgcolor="white", plot_bgcolor="white", font_color="#202124",
            )
            fig.update_yaxes(autorange="reversed", tickfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Chronologie reconstituée avec tau1–tau4, les durées des tournées, les transferts et zeta. "
                "Une barre P2/P4 regroupe les camions d'une même liaison ; ce n'est pas un ordonnancement individuel. "
                "La ligne rouge matérialise Tmax."
            )
            with st.expander("Afficher toutes les opérations et leurs horaires"):
                st.dataframe(pd.DataFrame(chronologie).drop(columns="Type").round(2),
                             hide_index=True, use_container_width=True)

    with tab_gain:
        st.markdown("**Mesurer l'amélioration depuis la construction initiale**")
        st.caption("Trois repères : construction initiale (avant réparation), entrée du recuit (après réparation, premier point de convergence) et solution finale. Graine de recherche 0, comme le calcul principal.")
        lignes = analyser_gain_initial(execution, seed_recherche=0)
        affichage = []
        for x in lignes:
            affichage.append({
                "Méthode": x["Méthode"],
                "Initial faisable": "Oui" if x["Initial faisable"] else "Non",
                "Final faisable": "Oui" if x["Final faisable"] else "Non",
                "Obj. initial": round(x["Objectif initial"], 2),
                "Obj. entrée recuit": round(x["Objectif entrée recuit"], 2) if x["Objectif entrée recuit"] is not None else None,
                "Obj. final": round(x["Objectif final"], 2),
                "Gain recherche (%)": round(x["Gain recherche (%)"], 2) if x["Gain recherche (%)"] is not None else None,
                "Gain objectif (%)": round(x["Gain objectif (%)"], 2) if x["Gain objectif (%)"] is not None else None,
                "Gain coût (%)": round(x["Gain transport (%)"], 2) if x["Gain transport (%)"] is not None else None,
            })
        st.dataframe(pd.DataFrame(affichage), hide_index=True, use_container_width=True)
        fig = go.Figure()
        for nom in [x["Méthode"] for x in lignes]:
            ligne = next(x for x in lignes if x["Méthode"] == nom)
            fig.add_trace(go.Bar(name=f"{nom} · initial", x=[nom], y=[ligne["Objectif initial"]],
                                 marker_color="#C7D0D8", offsetgroup=0))
            if ligne["Objectif entrée recuit"] is not None:
                fig.add_trace(go.Bar(name=f"{nom} · entrée recuit", x=[nom],
                                     y=[ligne["Objectif entrée recuit"]],
                                     marker_color="#8E9EA9", offsetgroup=1))
            fig.add_trace(go.Bar(name=f"{nom} · final", x=[nom], y=[ligne["Objectif final"]],
                                 marker_color=COLORS_METHOD[nom], offsetgroup=2))
        fig.update_layout(barmode="group", height=355, yaxis_title="Objectif pénalisé",
                          margin=dict(l=8, r=8, t=15, b=15), paper_bgcolor="white",
                          plot_bgcolor="white", font_color="#202124")
        st.plotly_chart(fig, use_container_width=True)
        if any(not x["Initial faisable"] for x in lignes):
            st.info("Au moins une solution initiale est infaisable. Le gain total sur l'objectif comprend la réduction des pénalités : il ne correspond pas à une économie de transport. Le gain 'recherche' isole l'évolution depuis l'entrée du recuit (après réparation), mais reste un gain d'objectif pénalisé. Le gain de coût n'est affiché que lorsque les deux solutions sont faisables.")
        else:
            st.caption("Lorsque les deux solutions sont faisables, le gain de coût de transport est directement comparable.")
