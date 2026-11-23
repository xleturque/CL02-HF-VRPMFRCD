"""Adaptateur entre l'interface web et les deux modules scientifiques du projet CL02.

Le générateur et les algorithmes sont importés sans modification de leur code.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable, Optional, Sequence

from hf_vrpmfrcd_instances import UNITES_TEMPS, generate_instance
from hf_vrpmfrcd_metaheuristics import Problem, exporter_dat_pour_cplex, quick_params, solve

METHODES = ("SA", "SAVND", "ANSA")


def executer_comparaison(
    set_id: int,
    graine: int,
    methodes: Sequence[str] = METHODES,
    secondes: float = 30.0,
    progression: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Une exécution par méthode sur une même instance, comme dans main()."""
    if set_id not in UNITES_TEMPS:
        raise ValueError("Le set doit être compris entre 1 et 5.")
    if graine < 0:
        raise ValueError("La graine doit être positive ou nulle.")
    if secondes <= 0:
        raise ValueError("La durée de calcul doit être positive.")
    if not methodes or any(m not in METHODES for m in methodes):
        raise ValueError("Sélectionner au moins une méthode connue.")

    donnees = generate_instance(set_id, seed=graine)
    prob = Problem(**donnees)
    resultats = []

    for numero, methode in enumerate(methodes, start=1):
        # Les métaheuristiques utilisent la graine 0, comme la console originale.
        reglages = quick_params(prob, seed=0, time_limit=float(secondes))
        debut = perf_counter()
        solution, score, historique, statistiques = solve(
            prob, methode.lower(), reglages
        )
        duree = perf_counter() - debut
        details = prob.evaluate(solution, detail=True)
        resultats.append(
            {
                "methode": methode,
                "solution": solution,
                "score": score,
                "historique": historique,
                "statistiques": statistiques,
                "duree_calcul": duree,
                "details": details,
            }
        )
        if progression is not None:
            progression(numero, len(methodes), methode)

    return {
        "set_id": set_id,
        "graine": graine,
        "secondes": secondes,
        "unite_python": UNITES_TEMPS[set_id][0],
        "donnees": donnees,
        "resultats": resultats,
    }


def facteur_minutes(set_id: int) -> float:
    """Convertit les temps Python du set en minutes pour les vues et CPLEX."""
    return float(UNITES_TEMPS[set_id][1])


def classement(resultats: Sequence[dict]) -> dict:
    """Même règle que dans main() : faisable d'abord, sinon objectif pénalisé."""
    valides = [r for r in resultats if r["details"]["feasible"]]
    if valides:
        return min(valides, key=lambda r: r["details"]["cost"])
    return min(resultats, key=lambda r: r["details"]["obj"])


def tableau_comparaison(execution: dict) -> list[dict]:
    facteur = facteur_minutes(execution["set_id"])
    lignes = []
    for r in execution["resultats"]:
        d = r["details"]
        lignes.append(
            {
                "Méthode": r["methode"],
                "Coût transport": round(d["cost"], 2),
                "Objectif pénalisé": round(d["obj"], 2),
                "Faisable": "Oui" if d["feasible"] else "Non",
                "Calcul (s)": round(r["duree_calcul"], 2),
                "Opérations (min)": round(d["duree_operations"] * facteur, 2),
                "Tmax (min)": round(execution["donnees"]["tmax"] * facteur, 2),
                "Itérations": r["statistiques"].get("iterations", 0),
            }
        )
    return lignes


def tableau_tournees(execution: dict, resultat: dict) -> list[dict]:
    """Libellés C1..Cn ou S1..Sn à la place des indices internes."""
    nD = execution["donnees"]["nD"]
    facteur = facteur_minutes(execution["set_id"])
    lignes = []
    for phase in ("P1", "P3", "P5"):
        for index, (dock, noeuds, veh, cout, duree) in enumerate(
            resultat["details"][phase], start=1
        ):
            prefixe = "C" if phase == "P3" else "S"
            trajet = " → ".join(
                [f"D{dock}"]
                + [f"{prefixe}{n - nD}" for n in noeuds]
                + [f"D{dock}"]
            )
            lignes.append(
                {
                    "Processus": phase,
                    "Tournée": index,
                    "Véhicule": veh,
                    "Cross-dock": f"D{dock}",
                    "Parcours": trajet,
                    "Coût": round(cout, 2),
                    "Durée (min)": round(duree * facteur, 2),
                }
            )
    return lignes


def tableau_transferts(execution: dict, resultat: dict, phase: str) -> list[dict]:
    """
    Détaille les transferts inter-cross-docks pour P2 ou P4.
    """
    details = resultat["details"]
    donnees = execution["donnees"]
    facteur = facteur_minutes(execution["set_id"])

    if phase == "P2":
        camions = details.get("trucks2", {})
        flux = details.get("flow2", {})
    else:
        camions = details.get("trucks4", {})
        flux = details.get("flow4", {})

    lignes = []
    for (i, j), nombre_camions in sorted(camions.items()):
        quantite = flux.get((i, j), 0.0)
        lignes.append(
            {
                "Origine": f"D{i}",
                "Destination": f"D{j}",
                "Quantité transférée": round(quantite, 2),
                "Camions": nombre_camions,
                "Durée liaison (min)": round(donnees["tD"][i-1][j-1] * facteur, 2),
                "Coût liaison": round(donnees["cD"][i-1][j-1] * donnees["usetW"] * nombre_camions, 2),
            }
        )
    return lignes

def generer_dat(execution: dict) -> tuple[str, bytes]:
    """Produit le .dat CPLEX en minutes sans laisser de fichier sur le serveur."""
    set_id, graine = execution["set_id"], execution["graine"]
    nom = f"Set{set_id}_graine{graine}_CPLEX.dat"
    with TemporaryDirectory(prefix="cl02_export_") as dossier:
        chemin = Path(dossier) / nom
        exporter_dat_pour_cplex(execution["donnees"], chemin, set_id, graine)
        contenu = chemin.read_bytes()
    return nom, contenu

def graphe_flux_simplifie(execution: dict, resultat: dict) -> dict:
    """
    Prépare un graphe simplifié des flux pour l'affichage Streamlit.
    Utilisable surtout pour les petits sets.
    """
    inst = execution["donnees"]
    nD, nS, nC = inst["nD"], inst["nS"], inst["nC"]

    # On limite l'affichage aux petites instances
    if nD + nS + nC > 20:
        return {"actif": False, "raison": "Instance trop grande pour un schéma lisible."}

    positions = {}
    etiquettes = {}

    # Fournisseurs à gauche
    for s in range(1, nS + 1):
        positions[f"S{s}"] = (0, -s)
        etiquettes[f"S{s}"] = f"S{s}"
    # Docks au centre
    for d in range(1, nD + 1):
        positions[f"D{d}"] = (1.5, -d * 2)
        etiquettes[f"D{d}"] = f"D{d}"
    # Clients à droite
    for c in range(1, nC + 1):
        positions[f"C{c}"] = (3, -c)
        etiquettes[f"C{c}"] = f"C{c}"
    aretes = []
    # P1 : D -> S -> D (on simplifie en D-S)
    for dock, noeuds, veh, cout, duree in resultat["details"]["P1"]:
        for n in noeuds:
            aretes.append(("P1", f"D{dock}", f"S{n - nD}"))
    # P3 : D -> C -> D
    for dock, noeuds, veh, cout, duree in resultat["details"]["P3"]:
        for n in noeuds:
            aretes.append(("P3", f"D{dock}", f"C{n - nD}"))
    # P5 : D -> S -> D
    for dock, noeuds, veh, cout, duree in resultat["details"]["P5"]:
        for n in noeuds:
            aretes.append(("P5", f"D{dock}", f"S{n - nD}"))
    # P2 transferts docks
    for (i, j), q in resultat["details"].get("flow2", {}).items():
        if q > 0:
            aretes.append(("P2", f"D{i}", f"D{j}"))
    # P4 transferts docks
    for (i, j), q in resultat["details"].get("flow4", {}).items():
        if q > 0:
            aretes.append(("P4", f"D{i}", f"D{j}"))
    return {
        "actif": True,
        "positions": positions,
        "etiquettes": etiquettes,
        "aretes": aretes,
    }