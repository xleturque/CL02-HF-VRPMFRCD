"""Analyses complementaires du dashboard CL02.

- Analyse de robustesse multi-executions sur une instance identique.
- Chronologie agregee P1-P5 avec preparation P3/P5.
- Amelioration de l'objectif penalise : entree recuit vs resultat final.
"""

from __future__ import annotations

import random
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable, Optional

import pandas as pd
import plotly.graph_objects as go

from hf_vrpmfrcd_metaheuristics import Problem, quick_params, solve

COLORS_METHOD = {
    "SA": "#F28E2B",
    "SAVND": "#59A14F",
    "ANSA": "#4E79A7",
}
COLORS_PHASE = {
    "P1": "#E15759",
    "P2": "#F28E2B",
    "P3": "#D4A017",
    "P4": "#269DAB",
    "P5": "#9467BD",
}


def _facteur_minutes(execution: dict) -> float:
    from hf_vrpmfrcd_instances import UNITES_TEMPS
    return float(UNITES_TEMPS[execution["set_id"]][1])


def analyser_gain_initial(execution: dict, seed_recherche: int = 0) -> list[dict]:
    """Reference brute avant reparation et recuit."""
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
            "Gain recherche (%)": (
                100.0 * (entree_recuit - objf) / entree_recuit
                if entree_recuit is not None and entree_recuit > 0 else None
            ),
            "Gain objectif (%)": (
                100.0 * (obj0 - objf) / obj0 if obj0 > 0 else None
            ),
            "Gain transport (%)": (
                100.0 * (cout0 - coutf) / cout0
                if comparable and cout0 > 0 else None
            ),
        })
    return lignes


def construire_chronologie(execution: dict, resultat: dict) -> list[dict]:
    """Reconstruit les intervalles temporels des operations."""
    inst = execution["donnees"]
    details = resultat["details"]
    f = _facteur_minutes(execution)
    tau = details["tau"]
    zeta = float(inst["zeta"])
    nD = int(inst["nD"])
    lignes = []

    def ajouter(
        phase: str, activite: str, debut: float, duree: float,
        dock: str, type_barre="operation"
    ):
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
        for idx, (dock, noeuds, veh, cout, duree) in enumerate(
            details[phase], 1
        ):
            if phase == "P1":
                debut = 0.0
            elif phase == "P3":
                debut = float(tau["tau2"][dock - 1]) + zeta
            else:
                debut = float(tau["tau4"][dock - 1]) + zeta

            prefix = "C" if phase == "P3" else "S"
            sommets = ", ".join(
                f"{prefix}{noeud - nD}" for noeud in noeuds
            )
            ajouter(
                phase,
                f"{phase} · D{dock} · V{veh} · tournée {idx} ({sommets})",
                debut, float(duree), f"D{dock}"
            )

    for phase, tau_origine, camions in (
        ("P2", "tau1", details.get("trucks2", {})),
        ("P4", "tau3", details.get("trucks4", {})),
    ):
        for (origine, destination), nombre in sorted(camions.items()):
            if not nombre:
                continue
            debut = float(tau[tau_origine][origine - 1])
            duree = float(inst["tD"][origine - 1][destination - 1])
            ajouter(
                phase,
                f"{phase} · D{origine} → D{destination} · {nombre} camion(s)",
                debut, duree, f"D{origine} → D{destination}"
            )

    for phase, cle, phase_preparation in (
        ("P3", "tau2", "Préparation P3"),
        ("P5", "tau4", "Préparation P5"),
    ):
        docks_actifs = sorted({
            tour[0] for tour in details[phase]
        })
        for dock in docks_actifs:
            ajouter(
                phase,
                f"{phase_preparation} · D{dock}",
                float(tau[cle][dock - 1]),
                zeta, f"D{dock}",
                type_barre="preparation"
            )

    ordre = {p: k for k, p in enumerate(COLORS_PHASE)}
    lignes.sort(
        key=lambda a: (
            ordre[a["Phase"]],
            a["Début (min)"],
            a["Activité"]
        )
    )
    return lignes


def executer_repetitions(
    execution: dict,
    methodes: list[str],
    repetitions: int = 5,
    secondes: float = 5.0,
    progression: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """Repete les methodes sur la meme instance."""
    valides = {"SA", "SAVND", "ANSA"}

    if not methodes or any(m not in valides for m in methodes):
        raise ValueError("Choisir SA, SAVND ou ANSA.")
    if not 2 <= repetitions <= 10:
        raise ValueError("Nombre de repetitions compris entre 2 et 10.")
    if not 0.1 <= secondes <= 15:
        raise ValueError("Budget par essai compris entre 0,1 et 15 s.")
    if repetitions * secondes * len(methodes) > 150:
        raise ValueError("Budget cumule maximal : 150 secondes.")

    prob = Problem(**execution["donnees"])
    lignes = []
    total = repetitions * len(methodes)
    n = 0

    for seed in range(repetitions):
        for methode in methodes:
            params = quick_params(
                prob, seed=seed, time_limit=float(secondes)
            )
            debut = perf_counter()
            solution, score, hist, stats = solve(
                prob, methode.lower(), params
            )
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
                progression(
                    n, total,
                    f"{methode}, essai {seed + 1}/{repetitions}"
                )
    return lignes


def resumer_repetitions(lignes: list[dict]) -> list[dict]:
    """Statistiques sur les solutions faisables uniquement."""
    sortie = []

    for methode in ("SA", "SAVND", "ANSA"):
        essais = [
            r for r in lignes if r["Méthode"] == methode
        ]
        if not essais:
            continue

        valides = [
            r["Coût transport"] for r in essais if r["Faisable"]
        ]
        sortie.append({
            "Méthode": methode,
            "Répétitions": len(essais),
            "Faisabilité (%)": 100 * len(valides) / len(essais),
            "Coût moyen faisable": mean(valides) if valides else None,
            "Meilleur coût faisable": min(valides) if valides else None,
            "Écart-type coût": (
                pstdev(valides) if len(valides) >= 2
                else (0.0 if valides else None)
            ),
            "Temps moyen (s)": mean(
                r["Calcul (s)"] for r in essais
            ),
            "Objectif moyen pénalisé": mean(
                r["Objectif pénalisé"] for r in essais
            ),
        })
    return sortie


def afficher_analyse_experimentale(execution: dict) -> None:
    """Analyse de robustesse des metaheuristiques."""
    import streamlit as st

    if not execution or not execution.get("resultats"):
        st.info("Lancez d'abord une optimisation.")
        return

    st.subheader("Analyse expérimentale")
    st.markdown("**Comparer la régularité des trois métaheuristiques**")
    st.caption(
        "L'instance reste identique. Seule la graine interne du "
        "recuit change. Les statistiques de coût portent sur "
        "les solutions déclarées faisables."
    )

    presentes = [
        r["methode"] for r in execution["resultats"]
    ]

    with st.form("cl02_form_analyse_experimentale"):
        c1, c2, c3 = st.columns([1.1, 1, 1])

        with c1:
            methodes = st.multiselect(
                "Méthodes", presentes, default=presentes,
                key="adv_methodes"
            )
        with c2:
            repetitions = st.select_slider(
                "Répétitions par méthode",
                options=[3, 5, 10],
                value=5,
                key="adv_repetitions"
            )
        with c3:
            budget = st.select_slider(
                "Budget par essai (s)",
                options=[3, 5, 10, 15],
                value=5,
                key="adv_budget"
            )

        total = repetitions * len(methodes) * budget
        st.caption(
            f"Budget théorique maximal : {total} s "
            "(plafond : 150 s)."
        )
        lancer = st.form_submit_button(
            "Lancer les répétitions", type="primary"
        )

    signature = (
        execution["set_id"], execution["graine"],
        tuple(methodes), repetitions, budget
    )

    if lancer:
        if not methodes:
            st.warning("Choisissez au moins une méthode.")
        elif total > 150:
            st.warning(
                "Réduisez le nombre de répétitions ou le temps."
            )
        else:
            st.session_state.pop(
                "cl02_repetitions_avancees", None
            )
            barre = st.progress(
                0.0, text="Calculs expérimentaux"
            )
            texte = st.empty()

            def maj(n, maxi, msg):
                barre.progress(
                    n / maxi,
                    text=f"{n}/{maxi} essais terminés"
                )
                texte.caption(msg)

            try:
                essais = executer_repetitions(
                    execution, methodes,
                    repetitions, budget, maj
                )
                st.session_state[
                    "cl02_repetitions_avancees"
                ] = {
                    "signature": signature,
                    "essais": essais
                }
                st.success(
                    "Campagne expérimentale terminée."
                )
            except Exception as exc:
                st.error(
                    f"Analyse interrompue : {exc}"
                )

    session = st.session_state.get(
        "cl02_repetitions_avancees"
    )

    if session and session["signature"] == signature:
        essais = session["essais"]
        stats = resumer_repetitions(essais)

        st.dataframe(
            pd.DataFrame(stats).round(2),
            hide_index=True,
            use_container_width=True
        )

        fig = go.Figure()
        for nom in methodes:
            valeurs = [
                x["Coût transport"] for x in essais
                if x["Méthode"] == nom and x["Faisable"]
            ]
            if valeurs:
                fig.add_trace(go.Box(
                    y=valeurs,
                    name=nom,
                    boxpoints="all",
                    jitter=0.15,
                    marker_color=COLORS_METHOD[nom],
                    line_color=COLORS_METHOD[nom],
                ))

        if fig.data:
            fig.update_layout(
                title="Dispersion des coûts faisables",
                yaxis_title="Coût transport",
                height=380,
                margin=dict(l=8, r=8, t=45, b=15),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font_color="#202124",
            )
            st.plotly_chart(
                fig, use_container_width=True
            )
        else:
            st.warning(
                "Aucune solution faisable pour le boxplot."
            )

        csv = pd.DataFrame(essais).to_csv(
            index=False, sep=";", decimal=","
        ).encode("utf-8-sig")
        st.download_button(
            "Exporter les essais (.csv)",
            data=csv,
            file_name=(
                f"Analyse_Set{execution['set_id']}"
                f"_graine{execution['graine']}.csv"
            ),
            mime="text/csv",
        )
        st.caption(
            "L'écart-type est calculé sur les coûts "
            "faisables uniquement."
        )
    elif session:
        st.info(
            "Les paramètres ont changé : relancez les répétitions."
        )
    else:
        st.info(
            "Définissez votre campagne expérimentale, "
            "puis lancez les répétitions."
        )


def construire_gantt_simplifie(
    execution: dict, resultat: dict
) -> list[dict]:
    """Regroupe les operations en cinq phases et deux preparations."""
    activites = construire_chronologie(
        execution, resultat
    )
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
        lignes = [
            a for a in activites
            if a["Phase"] == phase and a["Type"] == typ
        ]
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
    """Onglet Chronologie."""
    import streamlit as st

    st.subheader(
        "Chronologie des processus logistiques"
    )
    st.caption(
        "Cinq phases et préparations P3/P5."
    )

    resultats = execution["resultats"]
    methode = st.selectbox(
        "Méthode à représenter",
        [r["methode"] for r in resultats],
        key="cl02_methode_gantt_simple",
    )
    resultat = next(
        r for r in resultats
        if r["methode"] == methode
    )

    details = resultat["details"]
    facteur = _facteur_minutes(execution)
    tmax = (
        float(execution["donnees"]["tmax"]) * facteur
    )
    duree = (
        float(details["duree_operations"]) * facteur
    )
    marge = tmax - duree

    a, b, c = st.columns(3)
    a.metric(
        "Durée des opérations",
        f"{duree:,.1f} min".replace(",", " ")
    )
    b.metric(
        "Horizon Tmax",
        f"{tmax:,.1f} min".replace(",", " ")
    )
    c.metric(
        "Marge avant Tmax",
        f"{marge:+,.1f} min".replace(",", " ")
    )

    lignes = construire_gantt_simplifie(
        execution, resultat
    )

    if not lignes:
        st.info("Aucune opération à représenter.")
        return

    etiquettes = [
        "P1 · Collecte",
        "P2 · Transfert aller",
        "Préparation P3",
        "P3 · Livraison / retours",
        "P4 · Transfert retour",
        "Préparation P5",
        "P5 · Restitution",
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0] * len(etiquettes),
        y=etiquettes,
        mode="markers",
        marker=dict(
            size=1, color="rgba(0,0,0,0)"
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    for ligne in lignes:
        phase = ligne["Phase"]
        preparation = (
            ligne["Type"] == "preparation"
        )

        fig.add_trace(go.Bar(
            x=[ligne["Durée de l'enveloppe (min)"]],
            base=[ligne["Début (min)"]],
            y=[ligne["Étape"]],
            orientation="h",
            name=ligne["Étape"],
            marker_color=(
                "#8E9EA9" if preparation
                else COLORS_PHASE[phase]
            ),
            customdata=[[
                ligne["Début (min)"],
                ligne["Fin (min)"],
                ligne["Nombre d'activités"],
            ]],
            hovertemplate=(
                "%{y}"
                "<br>Début : %{customdata[0]:.1f} min"
                "<br>Fin : %{customdata[1]:.1f} min"
                "<br>Activités regroupées : %{customdata[2]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fin_max = max(
        ligne["Fin (min)"] for ligne in lignes
    )

    fig.add_vline(
        x=tmax,
        line_dash="dash",
        line_color="#C9394E",
        line_width=2,
        annotation_text="Tmax",
        annotation_position="top",
    )

    fig.update_layout(
        height=400,
        barmode="overlay",
        xaxis_title="Temps écoulé (min)",
        yaxis_title="",
        margin=dict(l=12, r=25, t=32, b=25),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#202124",
        xaxis=dict(
            range=[0, max(tmax, fin_max) * 1.07]
        ),
    )
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=etiquettes,
    )

    st.plotly_chart(
        fig, use_container_width=True
    )
    st.caption(
        "Chaque barre représente une enveloppe temporelle "
        "regroupant les activités d'une phase. "
        "Ce n'est pas une occupation continue des véhicules. "
        "P2 et P4 restent affichées sans barre si aucun "
        "transfert n'est nécessaire."
    )


def afficher_gain_optimisation(execution: dict) -> None:
    """Onglet Gain de l'optimisation."""
    import streamlit as st

    st.subheader(
        "Gain apporté par l'optimisation"
    )
    st.caption(
        "Construction initiale → réparation → recuit. "
        "Les valeurs correspondent à des objectifs pénalisés."
    )

    lignes = analyser_gain_initial(
        execution, seed_recherche=0
    )

    if not lignes:
        st.info("Aucune méthode à comparer.")
        return

    initial = lignes[0]
    gains = [
        l["Gain recherche (%)"] for l in lignes
        if l["Gain recherche (%)"] is not None
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Objectif initial (avant réparation)",
        f"{initial['Objectif initial']:,.2f}".replace(",", " ")
    )
    c2.metric(
        "Solution initiale faisable",
        "Oui" if initial["Initial faisable"] else "Non"
    )
    c3.metric(
        "Meilleur gain pendant le recuit",
        f"{max(gains):.2f} %" if gains else "—"
    )

    affichage = []
    afficher_gain_cout = any(
        l["Gain transport (%)"] is not None
        for l in lignes
    )

    for l in lignes:
        ligne = {
            "Méthode": l["Méthode"],
            "Initial faisable": (
                "Oui" if l["Initial faisable"] else "Non"
            ),
            "Final faisable": (
                "Oui" if l["Final faisable"] else "Non"
            ),
            "Obj. initial": round(
                l["Objectif initial"], 2
            ),
            "Obj. entrée recuit": (
                round(l["Objectif entrée recuit"], 2)
                if l["Objectif entrée recuit"] is not None
                else "—"
            ),
            "Obj. final": round(
                l["Objectif final"], 2
            ),
            "Gain recherche (%)": (
                round(l["Gain recherche (%)"], 2)
                if l["Gain recherche (%)"] is not None
                else "—"
            ),
            "Gain objectif (%)": (
                round(l["Gain objectif (%)"], 2)
                if l["Gain objectif (%)"] is not None
                else "—"
            ),
        }
        if afficher_gain_cout:
            ligne["Gain coût (%)"] = (
                round(l["Gain transport (%)"], 2)
                if l["Gain transport (%)"] is not None
                else "—"
            )
        affichage.append(ligne)

    st.dataframe(
        pd.DataFrame(affichage),
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "**Amélioration pendant le recuit : entrée vs final**"
    )
    fig = go.Figure()
    entrees = [
        l for l in lignes
        if l["Objectif entrée recuit"] is not None
    ]

    if entrees:
        fig.add_trace(go.Bar(
            name="Entrée du recuit",
            x=[l["Méthode"] for l in entrees],
            y=[l["Objectif entrée recuit"] for l in entrees],
            marker_color="#8E9EA9",
            offsetgroup="entree",
            text=[
                f"{l['Objectif entrée recuit']:.1f}"
                for l in entrees
            ],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>Entrée du recuit : %{y:.2f}"
                "<extra></extra>"
            ),
        ))

    for l in lignes:
        nom = l["Méthode"]
        fig.add_trace(go.Bar(
            name=f"Final · {nom}",
            x=[nom],
            y=[l["Objectif final"]],
            marker_color=COLORS_METHOD[nom],
            offsetgroup="final",
            text=[f"{l['Objectif final']:.1f}"],
            textposition="outside",
            hovertemplate=(
                f"{nom}<br>Objectif final : "
                "%{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        height=370,
        yaxis_title="Objectif pénalisé",
        yaxis=dict(rangemode="tozero"),
        margin=dict(l=10, r=12, t=28, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#202124",
        legend=dict(
            orientation="h", y=1.18, x=0
        ),
    )

    st.plotly_chart(
        fig, use_container_width=True
    )

    if any(
        not l["Initial faisable"] for l in lignes
    ):
        st.info(
            "La solution initiale est non faisable : "
            "son objectif élevé contient des pénalités. "
            "Cette valeur a volontairement été exclue "
            "du graphique pour rester lisible. "
            "La réduction de l'objectif ne représente "
            "pas nécessairement une économie de transport."
        )
    else:
        st.caption(
            "Lorsque les solutions comparées sont faisables, "
            "la diminution de l'objectif correspond "
            "à une réduction des coûts."
        )


def afficher_analyses(execution: dict) -> None:
    """Compatibilite avec l'ancien appel."""
    afficher_analyse_experimentale(execution)
