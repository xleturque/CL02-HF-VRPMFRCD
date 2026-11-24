"""Analyses complementaires du dashboard CL02.

Module independant de l'interface existante : aucune modification du solveur.
- Analyse de robustesse multi-executions sur une instance identique.
- Chronologie agregee P1-P5 avec preparation P3/P5.
- Amelioration objectif penalise : entree recuit vs resultat final.

Importer afficher_chronologie, afficher_gain_optimisation,
afficher_analyse_experimentale puis les appeler dans les onglets centraux.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Callable, Optional

import pandas as pd
import plotly.graph_objects as go

from hf_vrpmfrcd_metaheuristics import Problem
from optimisation import (HEURISTIQUES, SOLVEUR_EXACT, COULEURS_METHODES as COLORS_METHOD,
                          COULEURS_PROCESSUS as COLORS_PHASE, facteur_minutes,
                          executer_heuristique)


def _facteur_minutes(execution):
    return facteur_minutes(execution['set_id'])


def analyser_gain_initial(execution: dict) -> list[dict]:
    """Reference brute de initial_solution(), avant reparation/recuit.

    Les executions ordinaires de l'app utilisent seed_recherche=0. On compare
    l'objectif penalise meme si la solution initiale est infaisable ; le gain
    de cout de transport n'est montre que si les deux solutions sont faisables.
    """
    debut = execution.get("initial")
    if debut is None:
        return []
    lignes = []
    for resultat in execution["resultats"]:
        if resultat["methode"] not in HEURISTIQUES:
            continue
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
    if resultat["methode"] == SOLVEUR_EXACT:
        return []  # Le Gantt heuristique ne décrit pas le MILP.
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
    valides = set(HEURISTIQUES)
    if not methodes or any(m not in valides for m in methodes):
        raise ValueError("Veuillez choisir au moins une des methodes SA/SAVND/ANSA.")
    if not 2 <= repetitions <= 1000:
        raise ValueError("Le nombre de répétitions doit être compris entre 2 et 1000.")
    if not 0 < float(secondes) < float('inf'):
        raise ValueError("Le budget par exécution doit être positif et fini.")

    prob = Problem(**execution["donnees"])
    lignes = []
    total = repetitions * len(methodes)
    n = 0
    # Par repetition, comparer les trois methodes avec la meme graine de recherche.
    for seed in range(repetitions):
        for methode in methodes:
            r = executer_heuristique(prob, methode, secondes, seed)
            details = r['details']
            duree = r['duree_calcul']
            stats = r['statistiques']
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


def afficher_analyse_experimentale(execution: dict) -> None:
    """Analyse de robustesse : repetitions sur une instance commune."""
    import streamlit as st
    if not execution or not execution.get("resultats"):
        st.info("Lancez d'abord une optimisation pour acceder aux analyses.")
        return
    st.subheader("Analyse expérimentale")
    st.markdown("**Comparer la régularité des trois métaheuristiques**")
    st.caption(
        "L'instance (Set et graine de génération) reste fixe. Seule la graine interne "
        "du recuit change d'une répétition à l'autre. Les coûts statistiques ne "
        "concernent que les solutions déclarées faisables."
    )
    presentes = [r["methode"] for r in execution["resultats"] if r["methode"] in HEURISTIQUES]
    if not presentes:
        st.info("Analyse réservée aux métaheuristiques.")
        return
    precedent = st.session_state.get("cl02_repetitions_avancees")
    anciens = precedent["signature"] if precedent else None
    defaut_methodes = [m for m in anciens[2] if m in presentes] if anciens else presentes
    with st.form("cl02_form_analyse_experimentale"):
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            methodes = st.multiselect("Méthodes", presentes, default=defaut_methodes,
                                     key="adv_methodes")
        with c2:
            repetitions = st.number_input("Répétitions par méthode", min_value=2, max_value=1000,
                                           value=int(anciens[3]) if anciens else 5, step=1, key="adv_repetitions")
        with c3:
            budget = st.number_input("Budget par essai (s)", min_value=0.1,
                                      value=float(anciens[4]) if anciens else 5.0, step=1.0, key="adv_budget")
        total = repetitions * len(methodes) * budget
        st.caption(f"Budget théorique maximal cumulé : {total} s hors préparation et post-traitement.")
        lancer = st.form_submit_button("Lancer les répétitions", type="primary")
    signature = (execution["set_id"], execution["graine"], tuple(methodes), repetitions, budget)
    if lancer:
        if not methodes:
            st.warning("Choisissez au moins une méthode.")
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
        fonds_box = {
            "SA": "rgba(242,142,43,0.30)",
            "SAVND": "rgba(89,161,79,0.30)",
            "ANSA": "rgba(78,121,167,0.30)",
        }
        for nom in methodes:
            vals = [x["Coût transport"] for x in essais if x["Méthode"] == nom and x["Faisable"]]
            if vals:
                fig.add_trace(go.Box(
                    x=vals, y=[nom] * len(vals), orientation="h", name=nom,
                    boxpoints="all", jitter=0.22, pointpos=0,
                    marker=dict(color=COLORS_METHOD[nom], size=7),
                    line=dict(color=COLORS_METHOD[nom], width=2),
                    fillcolor=fonds_box[nom], showlegend=False,
                    hovertemplate="Coût de transport : %{x:.2f}<extra></extra>",
                ))
        if fig.data:
            fig.update_layout(
                title="Dispersion des coûts des solutions faisables",
                xaxis_title="Coût de transport", yaxis_title="",
                height=330, margin=dict(l=75, r=25, t=55, b=45),
                showlegend=False, paper_bgcolor="white", plot_bgcolor="white",
                font_color="#202124",
                xaxis=dict(showgrid=True, gridcolor="#E7EDF3", zeroline=False),
                yaxis=dict(categoryorder="array", categoryarray=["SA", "SAVND", "ANSA"],
                           autorange="reversed", showgrid=False),
            )
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


def construire_gantt_simplifie(execution: dict, resultat: dict) -> list[dict]:
    """Agrège en 5 phases et 2 préparations, sans détailler les tournées.

    Il s'agit de l'enveloppe temporelle des activités de chaque phase :
    les barres ne représentent PAS des ressources mobilisées en continu.
    """
    activites = construire_chronologie(execution, resultat)
    ordre = [
        ("P1", "operation", "P1 · Collecte"),
        ("P2", "operation", "P2 · Transfert aller"),
        ("P3", "preparation", "Préparation P3"),
        ("P3", "operation", "P3 · Livraison / retours"),
        ("P4", "operation", "P4 · Transfert retour"),
        ("P5", "preparation", "Préparation P5"),
        ("P5", "operation", "P5 · Restitution"),
    ]
    sortie = []
    for phase, typ, nom in ordre:
        lignes = [a for a in activites if a["Phase"] == phase and a["Type"] == typ]
        if not lignes:
            continue
        debut = min(a["Début (min)"] for a in lignes)
        fin = max(a["Fin (min)"] for a in lignes)
        if fin <= debut:
            continue
        sortie.append({
            "Phase": phase,
            "Étape": nom,
            "Début (min)": debut,
            "Fin (min)": fin,
            "Durée de l'enveloppe (min)": fin - debut,
            "Type": typ,
            "Nombre d'activités": len(lignes),
        })
    return sortie


def afficher_chronologie(execution: dict) -> None:
    """Onglet central : chronologie macro des phases P1–P5."""
    import streamlit as st
    st.subheader("Chronologie des processus logistiques")
    st.caption("Chronologie des métaheuristiques ou jalons temporels du MILP.")
    resultats = [r for r in execution["resultats"] if r["solution_disponible"]]
    if not resultats:
        st.info("Aucune solution à représenter.")
        return
    methode = st.selectbox("Méthode à représenter", [r["methode"] for r in resultats],
                           key="cl02_methode_gantt_simple")
    resultat = next(r for r in resultats if r["methode"] == methode)
    details = resultat["details"]
    f = _facteur_minutes(execution)
    tmax = float(execution["donnees"]["tmax"]) * f
    if methode == SOLVEUR_EXACT:
        duree = details.get("duree_operations")
        c1, c2 = st.columns(2)
        c1.metric("Fin P5 (post-traitement)", f"{duree*f:,.1f} min" if duree is not None else "—")
        c2.metric("Horizon Tmax", f"{tmax:,.1f} min")
        tau = details.get("tau")
        if tau is None:
            st.info("Dates HiGHS indisponibles : le post-traitement temporel n'a pas fourni de solution.")
            return
        st.caption("Dates tau du post-traitement HiGHS (jalons, pas intervalles d'activité). Les contraintes OPL ne déterminent pas tous les départs et temps d'attente ; aucun Gantt heuristique n'est imputé à HiGHS.")
        fig = go.Figure()
        for k, phase in enumerate(("P1", "P2", "P3", "P4"), 1):
            fig.add_trace(go.Scatter(x=[v*f for v in tau[f"tau{k}"]],
                y=[f"D{i+1}" for i in range(len(tau[f"tau{k}"]))], mode="markers", name=f"tau{k}",
                marker=dict(size=12,color=COLORS_PHASE[phase]), hovertemplate="%{y} : %{x:.2f} min<extra></extra>"))
        fig.add_vline(x=tmax,line_dash="dash",annotation_text="Tmax")
        fig.update_layout(xaxis_title="Temps (min)",height=350,legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(pd.DataFrame({"Cross-dock":[f"D{i+1}" for i in range(len(tau["tau1"]))],
            **{k:[round(v*f,3) for v in vals] for k,vals in tau.items()}}),hide_index=True)
        return
    duree = float(details["duree_operations"]) * f
    marge = tmax - duree
    a, b, c = st.columns(3)
    a.metric("Durée des opérations", f"{duree:,.1f} min".replace(",", " "))
    b.metric("Horizon Tmax", f"{tmax:,.1f} min".replace(",", " "))
    c.metric("Marge avant Tmax", f"{marge:+,.1f} min".replace(",", " "))
    lignes = construire_gantt_simplifie(execution, resultat)
    if not lignes:
        st.info("Aucune opération à représenter.")
        return
    fig = go.Figure()
    # Afficher aussi P2/P4 sur l'axe même s'il n'y a aucun transfert.
    etiquettes = [
        "P1 · Collecte", "P2 · Transfert aller", "Préparation P3",
        "P3 · Livraison / retours", "P4 · Transfert retour",
        "Préparation P5", "P5 · Restitution",
    ]
    fig.add_trace(go.Scatter(
        x=[0] * len(etiquettes), y=etiquettes, mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    for ligne in lignes:
        phase = ligne["Phase"]
        prep = ligne["Type"] == "preparation"
        fig.add_trace(go.Bar(
            x=[ligne["Durée de l'enveloppe (min)"]],
            base=[ligne["Début (min)"]],
            y=[ligne["Étape"]],
            orientation="h",
            name=ligne["Étape"],
            marker_color="#8E9EA9" if prep else COLORS_PHASE[phase],
            customdata=[[
                ligne["Début (min)"], ligne["Fin (min)"],
                ligne["Nombre d'activités"],
            ]],
            hovertemplate=(
                "%{y}<br>Début : %{customdata[0]:.1f} min"
                "<br>Fin : %{customdata[1]:.1f} min"
                "<br>Activités regroupées : %{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.add_vline(x=tmax, line_dash="dash", line_color="#C9394E", line_width=2,
                  annotation_text="Tmax", annotation_position="top")
    fig.update_layout(
        height=400, barmode="overlay", xaxis_title="Temps écoulé (min)",
        yaxis_title="", margin=dict(l=12, r=25, t=32, b=25),
        paper_bgcolor="white", plot_bgcolor="white", font_color="#202124",
        xaxis=dict(range=[0, max(tmax, max(a["Fin (min)"] for a in lignes)) * 1.07]),
    )
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=etiquettes)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Chaque barre représente l'intervalle entre le premier début et la dernière fin "
        "d'une phase (activités parallèles regroupées). Elle ne signifie pas que l'activité "
        "est continue pendant tout l'intervalle. P2 et P4 restent visibles "
        "sans barre lorsqu'aucun transfert n'est nécessaire."
    )


def afficher_gain_optimisation(execution: dict) -> None:
    """Onglet central : initial non faisable en métrique, recuit vs final en barres."""
    import streamlit as st
    st.subheader("Gain apporté par l'optimisation")
    st.caption("Construction initiale → réparation → recuit. Les valeurs sont des objectifs pénalisés, pas nécessairement des coûts de transport.")
    lignes = analyser_gain_initial(execution)
    if not lignes:
        st.info("Aucune méthode à comparer.")
        return
    initial = lignes[0]
    gains = [l["Gain recherche (%)"] for l in lignes if l["Gain recherche (%)"] is not None]
    c1, c2, c3 = st.columns(3)
    c1.metric("Objectif initial (avant réparation)", f"{initial['Objectif initial']:,.2f}".replace(",", " "))
    c2.metric("Solution initiale faisable", "Oui" if initial["Initial faisable"] else "Non")
    if gains:
        c3.metric("Meilleur gain pendant le recuit", f"{max(gains):.2f} %")
    else:
        c3.metric("Meilleur gain pendant le recuit", "—")
    affichage = []
    contient_gain_transport = any(l["Gain transport (%)"] is not None for l in lignes)
    for l in lignes:
        x = {
            "Méthode": l["Méthode"],
            "Initial faisable": "Oui" if l["Initial faisable"] else "Non",
            "Final faisable": "Oui" if l["Final faisable"] else "Non",
            "Obj. initial": round(l["Objectif initial"], 2),
            "Obj. entrée recuit": round(l["Objectif entrée recuit"], 2) if l["Objectif entrée recuit"] is not None else "—",
            "Obj. final": round(l["Objectif final"], 2),
            "Gain recherche (%)": round(l["Gain recherche (%)"], 2) if l["Gain recherche (%)"] is not None else "—",
            "Gain objectif (%)": round(l["Gain objectif (%)"], 2) if l["Gain objectif (%)"] is not None else "—",
        }
        if contient_gain_transport:
            x["Gain coût (%)"] = round(l["Gain transport (%)"], 2) if l["Gain transport (%)"] is not None else "—"
        affichage.append(x)
    st.dataframe(pd.DataFrame(affichage), hide_index=True, use_container_width=True)
    st.markdown("**Amélioration pendant le recuit : entrée vs final**")
    fig = go.Figure()
    with_entree = [l for l in lignes if l["Objectif entrée recuit"] is not None]
    if with_entree:
        fig.add_trace(go.Bar(
            name="Entrée du recuit", x=[l["Méthode"] for l in with_entree],
            y=[l["Objectif entrée recuit"] for l in with_entree],
            marker_color="#8E9EA9", offsetgroup="entree",
            text=[f"{l['Objectif entrée recuit']:.1f}" for l in with_entree],
            textposition="outside",
            hovertemplate="%{x}<br>Entrée du recuit : %{y:.2f}<extra></extra>",
        ))
    for l in lignes:
        nom = l["Méthode"]
        fig.add_trace(go.Bar(
            name=f"Final · {nom}", x=[nom], y=[l["Objectif final"]],
            marker_color=COLORS_METHOD[nom], offsetgroup="final",
            text=[f"{l['Objectif final']:.1f}"], textposition="outside",
            hovertemplate=f"{nom}<br>Objectif final : %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=370, yaxis_title="Objectif pénalisé",
        yaxis=dict(rangemode="tozero"),
        margin=dict(l=10, r=12, t=28, b=12),
        paper_bgcolor="white", plot_bgcolor="white", font_color="#202124",
        legend=dict(orientation="h", y=1.18, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    if any(not l["Initial faisable"] for l in lignes):
        st.info(
            "La solution initiale est non faisable : son objectif élevé comporte des pénalités. "
            "Le graphique exclut volontairement cette valeur pour rester lisible. "
            "La diminution de l'objectif ne correspond donc pas automatiquement "
            "à une économie de transport."
        )
    else:
        st.caption("La baisse de l'objectif correspond à une réduction des coûts lorsque les solutions comparées sont toutes faisables.")