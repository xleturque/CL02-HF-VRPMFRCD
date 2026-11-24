"""Interface Streamlit du projet HF-VRPMFRCD (CL02)."""
import math
from collections import defaultdict
from analyses_cl02 import (
    afficher_chronologie,
    afficher_gain_optimisation,
    afficher_analyse_experimentale,
)
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimisation import (
    HEURISTIQUES, SOLVEUR_EXACT, COULEURS_METHODES, COULEURS_PROCESSUS, methodes_disponibles,
    classement, executer_comparaison, facteur_minutes, itineraires,
    graphe_flux_simplifie, tableau_comparaison, tableau_tournees, tableau_transferts
)

# =====================================================================
# COULEURS
# =====================================================================
NOMS_PROCESSUS = {
    "P1": "P1 · Collecte fournisseurs",
    "P2": "P2 · Transferts aller",
    "P3": "P3 · Livraison et collecte clients",
    "P4": "P4 · Transferts retour",
    "P5": "P5 · Retour fournisseurs"
}
COULEURS_NOEUDS = {"S": "#64748B", "D": "#147D78", "C": "#D18B46"}

# =====================================================================
# CONFIGURATION ET PRESENTATION
# =====================================================================
st.set_page_config(page_title="CL02 | HF-VRPMFRCD", page_icon="🚚", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp { background-color: #F7F9FB; }
[data-testid="stSidebar"] { background-color: #FFFFFF; }
.block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 3rem; }
.eyebrow { font-size: .76rem; font-weight: 700; letter-spacing: .14em;
           color: #147D78; text-transform: uppercase; margin-bottom: .5rem; }
.app-title { font-weight: 760; letter-spacing: -.035em; font-size: 2.2rem;
             margin: 0; line-height: 1.15; color: #182B3A; }
.app-subtitle { color: #526578; margin-top: .65rem; margin-bottom: 1.75rem;
                max-width: 820px; }
.stMetric { padding: 1.05rem 1.2rem; background: white;
            border: 1px solid #E5EBF0; border-radius: 12px; }
[data-testid="stMetricLabel"] { color: #627284; font-size: .88rem; }
[data-testid="stMetricValue"] { font-size: 1.55rem; color: #192D3C; }
div[data-testid="stDataFrame"] { border: 1px solid #E5EBF0; border-radius: 10px; }
h2, h3 { color: #192D3C; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="eyebrow">Projet CL02 · Optimisation logistique</div>
<h1 class="app-title">HF-VRPMFRCD <span style="font-weight:400;color:#80909D;">/</span> Dashboard</h1>
<p class="app-subtitle">Comparaison de SA, SAVND, ANSA et du solveur HiGHS (Sets 1–2) sur un problème
de tournées de véhicules avec flotte hétérogène et cross-docks aller / retour.</p>
""", unsafe_allow_html=True)

# =====================================================================
# FONCTIONS DU GRAPHE : TOURNEES REELLES ET COURBES
# =====================================================================
def ajouter_liaison(fig, positions, depart, arrivee, phase, etiquette="", description="", repetition=0):
    """Trace une liaison orientee avec une courbe de Bezier et une fleche."""
    if depart not in positions or arrivee not in positions or depart == arrivee:
        return

    x1, y1 = positions[depart]
    x2, y2 = positions[arrivee]

    # Perpendiculaire stable : meme orientation pour les deux sens d'une liaison.
    a, b = sorted((depart, arrivee))
    vx = positions[b][0] - positions[a][0]
    vy = positions[b][1] - positions[a][1]
    longueur = max(math.hypot(vx, vy), 1e-9)
    px, py = -vy / longueur, vx / longueur

    # Separation des processus sur des courbes distinctes.
    courbures = {"P1": -0.30, "P2": -0.22, "P3": 0.24, "P4": 0.22, "P5": 0.32}
    courbure = courbures[phase]

    # Si plusieurs tournees utilisent la meme liaison, elles sont decalees.
    if repetition > 0:
        signe = -1 if repetition % 2 else 1
        courbure += signe * math.ceil(repetition / 2) * 0.12

    # Point de controle de la courbe de Bezier quadratique.
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = xm + px * courbure, ym + py * courbure

    xs, ys = [], []
    for k in range(41):
        t = k / 40
        x = (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
        y = (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
        xs.append(x)
        ys.append(y)

    couleur = COULEURS_PROCESSUS[phase]
    tirets = "dash" if phase in ("P2", "P4") else "solid"

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=couleur, width=2.5, dash=tirets),
        text=[description] * len(xs),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ))

    # Pointe de fleche selon la tangente de la courbe, avant le noeud d'arrivee.
    fig.add_annotation(
        x=xs[-4], y=ys[-4], ax=xs[-10], ay=ys[-10],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=2.5, arrowcolor=couleur
    )

    if etiquette:
        fig.add_annotation(
            x=xs[20], y=ys[20], text=etiquette, showarrow=False,
            font=dict(size=10, color=couleur),
            bgcolor="rgba(255,255,255,0.90)", borderpad=3
        )


def construire_graphe_tournees(execution, resultat, phase_choisie, afficher_vehicules=True):
    """Graphe des trajets complets avec courbes et separation des arcs reutilises."""
    inst = execution["donnees"]
    schema = graphe_flux_simplifie(execution, resultat)
    if not schema["actif"]:
        return None, schema["raison"]

    positions = schema["positions"]
    fig = go.Figure()
    compteurs = defaultdict(int)

    # Les chemins OPL de HiGHS ne sont jamais reconstruits en tournées heuristiques.
    for route in itineraires(execution, resultat):
        phase = route['phase']
        if phase_choisie not in ('Toutes les phases', phase):
            continue
        veh = route['veh']; chemin = route['labels']
        description = (f"{phase} · Tournée {route['index']}<br>Véhicule V{veh} · Capacité {inst['qV'][veh-1]}"
                       f"<br>{' → '.join(chemin)}<br>Coût : {route['cost']:.3f}"
                       f"<br>Temps de trajet : {route['minutes']:.3f} min")
        for indice_arc, (depart, arrivee) in enumerate(zip(chemin, chemin[1:])):
            cle = (phase, *sorted((depart, arrivee)))
            repetition = compteurs[cle]; compteurs[cle] += 1
            montrer = afficher_vehicules and (phase_choisie != 'Toutes les phases' or indice_arc == 0)
            ajouter_liaison(fig, positions, depart, arrivee, phase, f'V{veh}' if montrer else '', description, repetition)

    for phase in ('P2','P4'):
        if phase_choisie not in ('Toutes les phases', phase):
            continue
        for liaison in tableau_transferts(execution, resultat, phase):
            depart, arrivee = liaison['Origine'], liaison['Destination']
            q = liaison['Quantité transférée']; n = liaison['Camions']
            cle = (phase, *sorted((depart, arrivee)))
            repetition = compteurs[cle]; compteurs[cle] += 1
            qt = f'{q:g}' if q is not None else '—'
            description = f'{phase} · {depart} → {arrivee}<br>Quantité : {qt}<br>Camions : {n}'
            if liaison.get("Véhicules"):
                description += "<br>" + liaison["Véhicules"]
            ajouter_liaison(fig, positions, depart, arrivee, phase, f'Q={qt}', description, repetition)

    # Legende des processus.
    for phase in ("P1", "P2", "P3", "P4", "P5"):
        if phase_choisie not in ("Toutes les phases", phase):
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=NOMS_PROCESSUS[phase],
            line=dict(color=COULEURS_PROCESSUS[phase], width=3,
                      dash="dash" if phase in ("P2", "P4") else "solid")
        ))

    # Noeuds : fournisseurs en losange, docks en carre, clients en cercle.
    for prefixe, nom, symbole in (
        ("S", "Fournisseurs", "diamond"),
        ("D", "Cross-docks", "square"),
        ("C", "Clients", "circle")
    ):
        noeuds = [n for n in positions if n.startswith(prefixe)]
        fig.add_trace(go.Scatter(
            x=[positions[n][0] for n in noeuds],
            y=[positions[n][1] for n in noeuds],
            mode="markers+text", name=nom, text=noeuds,
            textposition="top center", textfont=dict(size=12, color="#1F2937"),
            marker=dict(size=24, symbol=symbole, color="#FFFFFF",
                        line=dict(width=2.5, color=COULEURS_NOEUDS[prefixe])),
            hovertemplate="%{text}<extra></extra>"
        ))

    valeurs_y = [y for x, y in positions.values()]
    fig.update_layout(
        height=580, hovermode="closest",
        margin=dict(l=35, r=35, t=70, b=25),
        xaxis=dict(visible=False, range=[-0.45, 3.45]),
        yaxis=dict(visible=False, range=[min(valeurs_y) - 1, max(valeurs_y) + 1]),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig, None

# =====================================================================
# BARRE LATERALE
# =====================================================================
with st.sidebar:
    st.markdown("### Configuration")
    st.caption("Définissez une instance commune et les durées de recherche.")
    set_id = st.selectbox("Famille d'instances", [1, 2, 3, 4, 5], index=0, format_func=lambda x: f"Set {x}")
    disponibles = methodes_disponibles(set_id)
    methodes = st.multiselect("Méthodes à comparer", disponibles, default=list(HEURISTIQUES),
                               key=f"cl02_highs_methodes_{'petits' if set_id <= 2 else 'grands'}")
    if set_id >= 3:
        st.caption("Sets 3–5 : métaheuristiques uniquement.")
    if any(m in HEURISTIQUES for m in methodes):
        secondes = st.number_input("Temps maximal par métaheuristique (s)", min_value=1, value=20, step=5, format="%d")
    else:
        secondes = 20
    if SOLVEUR_EXACT in methodes:
        secondes_highs = st.number_input("Temps maximal HiGHS (s)", min_value=1, value=120, step=10, format="%d")
        st.caption("Limite de résolution, hors construction et post-traitement.")
    else:
        secondes_highs = 120
    graine = st.number_input("Graine de génération", min_value=0, max_value=2_147_483_647, value=11, step=1)
    budget_total = sum(secondes for m in methodes if m in HEURISTIQUES) + (secondes_highs if SOLVEUR_EXACT in methodes else 0)
    st.caption(f"Durée maximale de recherche : {budget_total} s.")
    lancer = st.button("Lancer l'optimisation", type="primary", use_container_width=True)
    st.divider()
    st.caption("**Méthode de calcul**")
    st.caption("Une exécution par méthode, sur la même instance.")
    st.caption("Les durées logistiques sont affichées en minutes.")

# =====================================================================
# EXECUTION DE L'OPTIMISATION
# =====================================================================
if lancer:
    if not methodes:
        st.warning("Sélectionnez au moins une méthode avant de lancer le calcul.")
    else:
        try:
            with st.status("Optimisation en cours", expanded=True) as statut:
                st.write(f"Génération du Set {set_id} avec la graine {graine}…")
                progression = st.progress(0, text="Préparation des calculs")
                suivi = st.empty()

                def afficher_progres(numero, total, methode):
                    progression.progress(numero / total, text=f"{numero}/{total} méthodes terminées")
                    suivi.write(f"{methode} : traitement terminé")

                execution = executer_comparaison(
                    set_id=int(set_id), graine=int(graine), methodes=methodes,
                    secondes=float(secondes), progression=afficher_progres,
                    secondes_highs=float(secondes_highs)
                )
                execution["comparaison"] = tableau_comparaison(execution)
                st.session_state["execution_cl02_highs"] = execution
                st.session_state.pop("cl02_repetitions_avancees", None)
                st.session_state["cl02_figures"] = {}
                st.session_state.pop("methode_tournees", None)
                st.session_state.pop("cl02_methode_gantt_simple", None)
                st.session_state.pop("adv_methodes", None)
                st.session_state.pop("adv_repetitions", None)
                st.session_state.pop("adv_budget", None)
                statut.update(label="Optimisation terminée", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Le calcul n'a pas pu aboutir : {exc}")

execution = st.session_state.get("execution_cl02_highs")
if execution is None:
    st.subheader("Votre espace d'optimisation")
    st.info("Choisissez un set et une graine dans le panneau de gauche, puis lancez l'optimisation.")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**01 · Instance**")
        st.caption("Réseau, flottes, demandes et retours.")
    with b:
        st.markdown("**02 · Recherche**")
        st.caption("SA, SAVND, ANSA et le solveur HiGHS (Sets 1–2) sur les mêmes données.")
    with c:
        st.markdown("**03 · Analyse**")
        st.caption("Coûts, convergence, tournées et données de l’instance.")
    st.stop()

inst = execution["donnees"]
resultats = execution["resultats"]
meilleur = classement(resultats)
facteur = facteur_minutes(execution["set_id"])
tmax_min = inst["tmax"] * facteur
aller, retour = sum(map(sum, inst["df"])), sum(map(sum, inst["pr"]))

# =====================================================================
# INDICATEURS GENERAUX
# =====================================================================
st.markdown(f"**Résultats · Set {execution['set_id']} · Graine {execution['graine']}**")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Clients", inst["nC"])
col2.metric("Fournisseurs", inst["nS"])
col3.metric("Cross-docks", inst["nD"])
col4.metric("Véhicules", inst["nV"])
col5.metric("Flux aller / retour", f"{aller:,.0f} / {retour:,.0f}")
col6.metric("Tmax", f"{tmax_min:,.0f} min".replace(",", " "))
st.caption(
    f"Unités internes Python : {execution['unite_python']} · " +
    (f"Temps maximal : {execution['secondes']:g} s par métaheuristique · " if any(r["methode"] in HEURISTIQUES for r in resultats) else "") +
    (f"Limite HiGHS : {execution['secondes_highs']:g} s · " if any(r["methode"] == SOLVEUR_EXACT for r in resultats) else "") +
    "Durées logistiques affichées en minutes."
)

for methode, erreur in execution.get('erreurs', {}).items():
    st.error(f"{methode} : {erreur}")
if not resultats:
    st.warning("Aucun résultat disponible. Vérifiez les erreurs ci-dessus.")

# Un seul onglet calculé à chaque interaction, contrairement à st.tabs classique.
ONGLETS = ("Vue d'ensemble", "Convergence", "Tournées et transferts",
           "Chronologie", "Gain de l'optimisation", "Données de l'instance")
vue = st.radio("Navigation", ONGLETS, horizontal=True, label_visibility="collapsed", key="cl02_navigation")

# =====================================================================
# ONGLET 1 - VUE D'ENSEMBLE
# =====================================================================
if vue == ONGLETS[0]:
    st.subheader("Comparaison des méthodes")
    st.dataframe(pd.DataFrame(execution["comparaison"]), hide_index=True, use_container_width=True)
    if meilleur is None:
        st.warning("Aucune solution disponible pour le classement.")
    elif meilleur['details']['feasible']:
        st.success(f"Plus faible coût parmi les solutions faisables selon leur modèle : **{meilleur['methode']} — {meilleur['details']['cost']:,.3f}**")
    else:
        st.warning(f"Aucune solution faisable trouvée. Meilleur objectif pénalisé : **{meilleur['methode']}**.")
    st.caption("La faisabilité dépend du modèle utilisé ; un résultat heuristique n'est pas certifié par le MILP.")
    comparables = [r for r in resultats if r['details']['feasible'] and r['details']['cost'] is not None]
    gauche, droite = st.columns(2, gap='large')
    with gauche:
        st.markdown("**Coûts de transport des solutions faisables**")
        if comparables:
            fig = go.Figure(go.Bar(x=[r['methode'] for r in comparables], y=[r['details']['cost'] for r in comparables],
                marker_color=[COULEURS_METHODES[r['methode']] for r in comparables],
                text=[f"{r['details']['cost']:,.3f}" for r in comparables], textposition='outside'))
            fig.update_layout(height=315, margin=dict(l=8,r=12,t=22,b=12),yaxis_title='Coût de transport',showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Aucun coût faisable à comparer.")
    with droite:
        st.markdown("**Répartition du coût par processus**")
        if meilleur is not None:
            couts = meilleur['details']['G']
            st.caption(meilleur['methode'] + ('' if meilleur['details']['feasible'] else ' · solution non faisable'))
            fig = go.Figure(go.Bar(x=list(couts),y=list(couts.values()),marker_color=[COULEURS_PROCESSUS[p] for p in couts],
                text=[f'{v:,.3f}' for v in couts.values()],textposition='outside'))
            fig.update_layout(height=315,margin=dict(l=8,r=12,t=22,b=12),yaxis_title='Coût par processus',showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
    highs = next((r for r in resultats if r['methode']==SOLVEUR_EXACT),None)
    if highs is not None:
        st.divider(); st.subheader("Référence HiGHS")
        st.write(f"Statut : **{highs['statut']}** — {highs.get('statut_brut') or '—'}")
        def fmt(x, suffix=''):
            return '—' if x is None else f'{x:,.3f}{suffix}'
        cols=st.columns(3)
        cols[0].metric('Coût HiGHS',fmt(highs['details']['cost']))
        cols[1].metric('Meilleure borne',fmt(highs['borne']))
        cols[2].metric('Gap relatif',fmt(highs['gap']*100 if highs['gap'] is not None else None,' %'))
        cols=st.columns(4)
        for col,cle,nom in zip(cols,('temps_construction','duree_calcul','temps_posttraitement','duree_totale'),('Construction','Résolution coût','Post-traitement','Total réel')):
            col.metric(nom,fmt(highs.get(cle),' s'))
        st.caption("Le total inclut la construction, la résolution, le post-traitement et les surcoûts Python.")
        ref=highs['details']['cost']
        if ref is not None and ref > 0:
            rows=[{'Méthode':r['methode'],'Coût':round(r['details']['cost'],3),
                   'Écart au coût HiGHS (%)':round(100*(r['details']['cost']-ref)/ref,4)}
                  for r in comparables if r['methode'] in HEURISTIQUES]
            if rows:
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                st.caption("Écart descriptif au coût HiGHS, même si l'optimum MILP est prouvé : l'équivalence des domaines faisables n'est pas établie. Ce n'est pas une certification d'optimalité heuristique.")

    # Analyse statistique de robustesse, conservee en bas de la Vue d'ensemble.
    st.divider()
    afficher_analyse_experimentale(execution)

# =====================================================================
# ONGLET 2 - CONVERGENCE
# =====================================================================
if vue == ONGLETS[1]:
    st.subheader("Évolution du meilleur objectif pénalisé")
    st.caption("Un point est enregistré lorsqu'une nouvelle meilleure solution est trouvée.")
    fig = go.Figure()
    for r in resultats:
        historique = r["historique"]
        if not historique:
            continue
        t = [float(p[0]) for p in historique] + [float(r["duree_calcul"])]
        v = [float(p[1]) for p in historique] + [float(historique[-1][1])]
        fig.add_trace(go.Scatter(
            x=t, y=v, mode="lines+markers",
            line=dict(shape="hv", width=2.5, color=COULEURS_METHODES[r["methode"]]),
            marker=dict(size=4), name=r["methode"],
            hovertemplate="Temps : %{x:.2f} s<br>Objectif : %{y:.2f}<extra></extra>"
        ))
    fig.update_layout(
        xaxis_title="Temps de calcul (s)", yaxis_title="Meilleur objectif pénalisé",
        height=440, hovermode="x unified", margin=dict(l=10, r=10, t=12, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    if fig.data:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun historique de recuit disponible.")
    highs = next((r for r in resultats if r["methode"] == SOLVEUR_EXACT), None)
    if highs is not None:
        st.caption(f"HiGHS : {highs['statut']} · pas d’historique de convergence collecté.")
        if highs["details"]["cost"] is not None:
            st.metric("Coût final HiGHS", f"{highs['details']['cost']:,.3f}")
    st.info("La convergence représente l'objectif pénalisé utilisé pendant l'optimisation, et non le seul coût de transport.")

# =====================================================================
# ONGLET 3 - TOURNEES ET TRANSFERTS
# =====================================================================
if vue == ONGLETS[2]:
    st.subheader("Détail de la solution")
    avec_solution = [r for r in resultats if r['solution_disponible']]
    if not avec_solution:
        st.info("Aucune solution disponible.")
    else:
        noms_methodes = [r["methode"] for r in avec_solution]
        choix = st.selectbox("Méthode à examiner", noms_methodes,
                             index=noms_methodes.index(meilleur["methode"]) if meilleur and meilleur["methode"] in noms_methodes else 0, key="methode_tournees")
        r = next(x for x in resultats if x["methode"] == choix)
        d = r["details"]

        a, b, c, dd = st.columns(4)
        a.metric("Coût transport", f"{d['cost']:,.2f}")
        b.metric("Objectif pénalisé", f"{d['obj']:,.2f}" if choix != SOLVEUR_EXACT else "—")
        c.metric("Fin P5" if choix == SOLVEUR_EXACT else "Durée des opérations", f"{d['duree_operations'] * facteur:,.1f} min" if d["duree_operations"] is not None else "—")
        dd.metric("Fin P5 ≤ Tmax" if choix == SOLVEUR_EXACT else "Horizon respecté", "—" if d["respecte_tmax"] is None else ("Oui" if d["respecte_tmax"] else "Non"))

        st.markdown("**Représentation des tournées et des flux**")
        if inst["nC"] + inst["nS"] + inst["nD"] <= 20:
            col_phase, col_veh = st.columns([2, 1])
            with col_phase:
                phase_a_voir = st.selectbox(
                    "Processus à visualiser", ["Toutes les phases", "P1", "P2", "P3", "P4", "P5"],
                    key="phase_graphe"
                )
            with col_veh:
                afficher_vehicules = st.checkbox("Afficher les véhicules", value=True, key="afficher_vehicules")

            cache = st.session_state.setdefault("cl02_figures", {})
            cle = (choix, phase_a_voir, afficher_vehicules)
            if cle not in cache:
                cache[cle] = construire_graphe_tournees(execution, r, phase_a_voir, afficher_vehicules)
            fig_reseau, erreur = cache[cle]
            if fig_reseau is not None:
                st.plotly_chart(fig_reseau, use_container_width=True)
                st.caption(
                    "Les flèches courbées représentent les liaisons successives des tournées calculées. "
                    "V1, V2… indiquent les véhicules ; Q indique les quantités transférées. "
                    "Survolez les courbes pour consulter les détails. Le schéma n'est pas à l'échelle."
                )
            else:
                st.info(erreur)
        else:
            st.info("Schéma réservé aux petites instances pour préserver la lisibilité. Les tournées restent consultables dans le tableau.")

        st.markdown("**Tournées P1, P3 et P5**")
        tours = tableau_tournees(execution, r)
        if tours:
            st.dataframe(pd.DataFrame(tours), hide_index=True, use_container_width=True)
        else:
            st.caption("Aucune tournée enregistrée.")

        gauche, droite = st.columns(2, gap="large")
        with gauche:
            st.markdown("**P2 · Transferts aller**")
            transferts = tableau_transferts(execution, r, "P2")
            if transferts:
                st.dataframe(pd.DataFrame(transferts), hide_index=True, use_container_width=True)
            else:
                st.caption("Aucun transfert entre cross-docks.")
        with droite:
            st.markdown("**P4 · Transferts retour**")
            transferts = tableau_transferts(execution, r, "P4")
            if transferts:
                st.dataframe(pd.DataFrame(transferts), hide_index=True, use_container_width=True)
            else:
                st.caption("Aucun transfert entre cross-docks.")

        if d.get("tau") is not None:
            with st.expander("Temps de fin des processus par cross-dock"):
                tau = d["tau"]
                tableau_tau = [
                    {"Cross-dock": f"D{i}", **{
                        phase: round(tau[phase][i - 1] * facteur, 2)
                        for phase in ("tau1", "tau2", "tau3", "tau4")
                    }}
                    for i in range(1, inst["nD"] + 1)
                ]
                st.dataframe(pd.DataFrame(tableau_tau), hide_index=True, use_container_width=True)
                st.caption("Temps exprimés en minutes depuis le début des opérations.")

# =====================================================================
# ONGLET 4 - CHRONOLOGIE
# =====================================================================
if vue == ONGLETS[3]:
    afficher_chronologie(execution)

# =====================================================================
# ONGLET 5 - GAIN DE L'OPTIMISATION
# =====================================================================
if vue == ONGLETS[4]:
    afficher_gain_optimisation(execution)

# =====================================================================
# ONGLET 6 - DONNEES ET EXPORT
# =====================================================================
if vue == ONGLETS[5]:
    st.subheader("Données de l'instance et téléchargements")
    st.caption("Les résultats et matrices correspondent exactement à l'instance commune utilisée.")
    csv = pd.DataFrame(execution["comparaison"]).to_csv(
        index=False, sep=";", decimal=","
    ).encode("utf-8-sig")
    st.download_button(
        "Télécharger la comparaison (.csv)", data=csv,
        file_name=f"Comparaison_Set{execution['set_id']}_graine{execution['graine']}.csv",
        mime="text/csv", use_container_width=True
    )

    with st.expander("Consulter les matrices de flux de l'instance"):
        if inst["nC"] * inst["nS"] > 10_000:
            st.caption("Matrices volumineuses : téléchargez les flux en CSV ci-dessous.")
        else:
            colonnes = [f"S{i}" for i in range(1, inst["nS"] + 1)]
            lignes_flux = [f"C{i}" for i in range(1, inst["nC"] + 1)]
            st.markdown("**Demandes (flux aller)**")
            st.dataframe(pd.DataFrame(inst["df"], columns=colonnes, index=lignes_flux), use_container_width=True)
            st.markdown("**Retours**")
            st.dataframe(pd.DataFrame(inst["pr"], columns=colonnes, index=lignes_flux), use_container_width=True)

    # Export des données utiles, sans conversion du modèle ni fichier temporaire.
    colonnes = [f"S{i}" for i in range(1, inst["nS"] + 1)]
    lignes_flux = [f"C{i}" for i in range(1, inst["nC"] + 1)]
    for cle, libelle in (("df", "Demandes"), ("pr", "Retours")):
        st.download_button(f"Télécharger {libelle.lower()} (.csv)",
            pd.DataFrame(inst[cle],columns=colonnes,index=lignes_flux).to_csv(sep=";",decimal=",").encode("utf-8-sig"),
            file_name=f"{libelle}_Set{execution['set_id']}_graine{execution['graine']}.csv",mime="text/csv")

st.divider()
st.caption(
    "CL02 · HF-VRPMFRCD · Résultats obtenus par les fonctions Python existantes. "
    "L'application ne remplace pas une vérification indépendante des contraintes du modèle exact."
)