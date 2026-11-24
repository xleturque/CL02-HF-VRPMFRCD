"""
hf_vrpmfrcd_instances.py : generateur des instances 1 a 5 de l'article
(Yu, Anh, Gunawan, Han, 2024, Expert Systems With Applications 237, 121631 -- Tableau 2).

    from hf_vrpmfrcd_instances import generate_instance
    inst = generate_instance(2, seed=1)             # Set 2 ; renvoie un dict aux noms OPL

    python hf_vrpmfrcd_instances.py --set 3 --seed 1

Le dict renvoye contient : nC nD nS nV nW  bigM tmax zeta  c1 t1 c2 t2 cD tD  df pr  hD
qV uV qsetW usetW  (matrices en listes de listes 0-indexees). Il se passe tel quel a
Problem(**inst) (metaheuristiques) et a resoudre_highs(inst, limite_temps=...) (HiGHS).

DONNEES DE L'ARTICLE (Tableau 2) -- les 30 instances reelles ne sont pas publiees : on regenere
des instances tirees selon les memes lois (graine = reproductibilite).
HYPOTHESES (non donnees, ou ambigues dans l'article) -- chacune est modifiable :
  H1  Set 4 : le tableau indique |C| = "90 ou 85" et |S| = "85 ou 15", ce qui ne donne pas
      100 noeuds. On prend les deux combinaisons a 100 noeuds : variant=0 -> (85 clients,
      15 fournisseurs) ; variant=1 -> (15, 85).
  H2  Set 5 : |C| = 180, |S| = 20 comme dans le tableau (le texte parle de 300 noeuds ; pour
      cela il faudrait |S| = 120 : generate_instance(5, nS=120)).
  H3  tmax = horizon ETENDU T'max (modele avec retours) ; reverse=False donne l'horizon aller.
  H4  Sets 4-5 : unites en heures (t = c/60, zeta = 2 h, T'max = 24 h) ; flotte de 3 types
      (L, M, H) avec n vehicules par type ; capacites et couts du tableau (note de bas de tableau).
  H5  Demande d'un client : somme_j d_ij ~ U(.,.) arrondie a l'entier, repartie au hasard
      entre les fournisseurs (repartition non precisee).
  H6  Retours : d'_j ~ U(.,.) du tableau, repartis au hasard entre les clients (p'_ij)
      (returns="table"). Variante "text" : p'_ij ~ U(0, max_i sum_j d_ij / 2) (texte de 5.1).
  H7  Distances / temps : tirages independants par arc (matrices non symetriques). Les trajets
      entre cross-docks sont identiques dans G1 et G2 ; cD, tD = bloc des docks de c1, t1.
  H8  Flotte de transfert (non donnee) : nW = 3 ; Sets 1-3 : qsetW = 120, usetW = 1 ;
      Sets 4-5 : qsetW = q_M, usetW = u_M.
  H9  Capacite des docks hD (non donnee) : non contraignante (= flux total le plus grand).
  H10 bigM = tmax + plus grand temps de trajet (valeur sure).
"""
import argparse
import math
import random

# ----------------------------------------------------------------------
# Tableau 2 (une ligne par instance)
#   sizes   : liste de (|C|, |S|) possibles (variant)       nD : |D| fixe ou (min, max) tire
#   fleet   : ("random", min, max) -> |V| ~ U, q ~ U(70,120), u = q/100
#             ("typed", n)        -> n vehicules par type L, M, H
#   dist    : c_ij ~ U(.,.)    time : t_ij ~ U(.,.) ou None (-> c_ij / 60)
#   demand  : somme_j d_ij ~ U(.,.)     ret : d'_j ~ U(.,.)     zeta, tmax (etendu), tmax_fwd
# ----------------------------------------------------------------------
TYPED_FLEET = dict(q=(4000.0, 12500.0, 26000.0), u=(12.00, 13.66, 16.86))     # types L, M, H

# Unites utilisees pour les temps de trajet, zeta, tmax et bigM en Python.
# Le tuple conserve l acces UNITES_TEMPS[set_id][0] des metaheuristiques.
UNITES_TEMPS = {
    1: ("min",), 2: ("min",), 3: ("min",),
    4: ("h",), 5: ("h",),
}

SETS = {
    1: dict(sizes=[(4, 6)],             nD=2,      fleet=("random", 2, 15), dist=(20, 200), time=(48, 560),
            demand=(3, 25), ret=(3, 25), zeta=50, tmax=1920, tmax_fwd=960,  transfer=(3, 120.0, 1.0)),
    2: dict(sizes=[(7, 23)],            nD=3,      fleet=("random", 2, 15), dist=(20, 200), time=(48, 480),
            demand=(3, 10), ret=(3, 10), zeta=50, tmax=2880, tmax_fwd=960,  transfer=(3, 120.0, 1.0)),
    3: dict(sizes=[(12, 38)],           nD=5,      fleet=("random", 2, 15), dist=(20, 200), time=(48, 560),
            demand=(3, 15), ret=(3, 15), zeta=50, tmax=2880, tmax_fwd=960,  transfer=(3, 120.0, 1.0)),
    4: dict(sizes=[(85, 15), (15, 85)], nD=(1, 3), fleet=("typed", 6),      dist=(0, 30),   time=None,
            demand=(0, 20), ret=(0, 10), zeta=2,  tmax=24,   tmax_fwd=12,   transfer=(3, 12500.0, 13.66)),
    5: dict(sizes=[(180, 20)],          nD=(1, 3), fleet=("typed", 25),     dist=(0, 300),  time=None,
            demand=(0, 20), ret=(0, 10), zeta=2,  tmax=24,   tmax_fwd=12,   transfer=(3, 12500.0, 13.66)),
}


# ----------------------------------------------------------------------
# Briques de generation (une par bloc de donnees)
# ----------------------------------------------------------------------
def _matrix(rg, n, lo, hi, nd=1):
    """Matrice n x n, diagonale nulle, valeurs ~ U(lo, hi) arrondies."""
    # Meme ordre des tirages (ligne puis colonne, sans tirage diagonal).
    uniform = rg.uniform
    indices = range(n)
    return [[0.0 if i == j else round(uniform(lo, hi), nd) for j in indices]
            for i in indices]


def _partition(rg, total, k):
    """Entier -> k entiers >= 0 de somme total (repartition aleatoire)."""
    cuts = [rg.randint(0, total) for _ in range(k - 1)]
    cuts.sort()
    cuts.append(total)
    parts = [b - a for a, b in zip([0] + cuts, cuts)]
    rg.shuffle(parts)
    return parts


def _fleet(rg, spec, nV=None):
    """Flotte heterogene : (qV, uV)."""
    kind = spec["fleet"]
    if kind[0] == "random":
        n = nV or rg.randint(kind[1], kind[2])
        q = [float(round(rg.uniform(70, 120))) for _ in range(n)]
        return q, [round(x / 100, 2) for x in q]
    n = kind[1]
    return ([x for x in TYPED_FLEET["q"] for _ in range(n)],
            [x for x in TYPED_FLEET["u"] for _ in range(n)])


def _network(rg, spec, nD, nS, nC):
    """Couts / temps de trajet des graphes G1 (docks + fournisseurs), G2 (docks + clients), docks."""
    lo, hi = spec["dist"]
    def graph(n):
        c = _matrix(rg, n, lo, hi)
        if spec["time"]:
            t = _matrix(rg, n, *spec["time"])
        else:
            # Les distances sont arrondies au dixieme. De nombreuses valeurs
            # se repetent : ne calculer la conversion en heures qu'une fois.
            conversion = {x: round(x / 60, 4) for x in {x for row in c for x in row}}
            t = [[conversion[x] for x in row] for row in c]
        return c, t
    c1, t1 = graph(nD + nS)
    c2, t2 = graph(nD + nC)
    for i in range(nD):                         # memes trajets entre cross-docks dans G1 et G2 (H7)
        for j in range(nD):
            c2[i][j], t2[i][j] = c1[i][j], t1[i][j]
    cD = [row[:nD] for row in c1[:nD]]
    tD = [row[:nD] for row in t1[:nD]]
    return c1, t1, c2, t2, cD, tD


def _demand(rg, spec, nC, nS):
    """df[c][s] : somme_s df[c][s] ~ U(demand), repartie entre les fournisseurs (H5)."""
    return [_partition(rg, round(rg.uniform(*spec["demand"])), nS) for _ in range(nC)]



def _returns(rg, spec, df, nC, nS, mode):
    """
    Génération des quantités retournées par les clients.
    Pour chaque client i :
    p'_ij ~ U_discrete(0, floor(max_j(d_ij) / 2))
    Les quantités retournées sont des entiers.
    """
    randint = rg.randint
    indices = range(nS)
    pr = []
    for i in range(nC):
        retour_max = max(df[i]) // 2
        pr.append([randint(0, retour_max) for _ in indices])
    return pr


def _check(inst):
    """Controle de coherence des dimensions des instances."""
    nC, nD, nS, nV = inst["nC"], inst["nD"], inst["nS"], inst["nV"]
    shapes = dict(c1=(nD + nS,) * 2, t1=(nD + nS,) * 2, c2=(nD + nC,) * 2, t2=(nD + nC,) * 2,
                  cD=(nD, nD), tD=(nD, nD), df=(nC, nS), pr=(nC, nS))
    for k, (r, c) in shapes.items():
        M = inst[k]
        if len(M) != r or any(len(row) != c for row in M):
            raise ValueError(f"{k} : taille {len(M)} x {len(M[0]) if M else 0} au lieu de {r} x {c}")
    for k, n in (("qV", nV), ("uV", nV), ("hD", nD)):
        if len(inst[k]) != n:
            raise ValueError(f"{k} : taille {len(inst[k])} au lieu de {n}")


# ----------------------------------------------------------------------
# Fonction principale
# ----------------------------------------------------------------------
def generate_instance(set_id, seed=1, variant=0, reverse=True, returns="text", **overrides):
    """Instance `set_id` (1 a 5) de l'article, aux noms de parametres OPL.

    seed      : graine (meme graine -> memes donnees pour HiGHS et heuristiques)
    variant   : Set 4 uniquement : 0 -> (85 clients, 15 fournisseurs), 1 -> (15, 85)   [H1]
    reverse   : True -> horizon etendu T'max (modele avec retours) ; False -> horizon aller  [H3]
    returns   : "table" (d'_j ~ U du Tableau 2) ou "text" (p'_ij ~ U(0, max sum_j d_ij / 2))  [H6]
    overrides : tout parametre force a la main. Structure (avant tirage) : nC, nS, nD, nV.
                Autres (apres tirage) : qV, uV, nW, qsetW, usetW, hD, tmax, zeta, bigM, ...
                ex. generate_instance(1, 1, nV=2, qV=[70, 120], uV=[0.7, 1.2])
    """
    if set_id not in SETS:
        raise ValueError(f"set_id doit etre dans {sorted(SETS)}")
    spec = SETS[set_id]
    rg = random.Random(seed)
    nC, nS = spec["sizes"][min(variant, len(spec["sizes"]) - 1)]
    nC, nS = overrides.pop("nC", nC), overrides.pop("nS", nS)
    nD = overrides.pop("nD", None) or (spec["nD"] if isinstance(spec["nD"], int) else rg.randint(*spec["nD"]))

    qV, uV = _fleet(rg, spec, overrides.pop("nV", None))
    c1, t1, c2, t2, cD, tD = _network(rg, spec, nD, nS, nC)
    df = _demand(rg, spec, nC, nS)
    pr = _returns(rg, spec, df, nC, nS, returns)
    nW, qsetW, usetW = spec["transfer"]

    inst = dict(nC=nC, nD=nD, nS=nS, nV=len(qV), nW=nW,
                tmax=float(spec["tmax"] if reverse else spec["tmax_fwd"]), zeta=float(spec["zeta"]),
                c1=c1, t1=t1, c2=c2, t2=t2, cD=cD, tD=tD, df=df, pr=pr,
                qV=qV, uV=uV, qsetW=qsetW, usetW=usetW)
    flow = max(sum(map(sum, df)), sum(map(sum, pr)))                                   # H9
    inst["hD"] = [float(math.ceil(flow))] * nD
    inst.update(overrides)
    if "nV" in overrides or "qV" in overrides:
        inst["nV"] = len(inst["qV"])
    if "bigM" not in overrides:                                                         # H10
        inst["bigM"] = float(math.ceil(inst["tmax"] + max(max(map(max, inst["t1"])), max(map(max, inst["t2"])))))
    _check(inst)
    return inst


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------
def summary(inst, set_id=None):
    """Resume d'une instance (taille, totaux et unite des temps)."""
    unite = f" {UNITES_TEMPS[set_id][0]}" if set_id in UNITES_TEMPS else ""
    return (f"|D|={inst['nD']} |S|={inst['nS']} |C|={inst['nC']} |V|={inst['nV']} |W|={inst['nW']}  "
            f"tmax={inst['tmax']:g}{unite} zeta={inst['zeta']:g}{unite} "
            f"bigM={inst['bigM']:g}{unite}  "
            f"flux aller={sum(map(sum, inst['df'])):g}  retours={sum(map(sum, inst['pr'])):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generateur des instances 1 a 5 de l'article")
    ap.add_argument("--set", type=int, default=1, choices=sorted(SETS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", type=int, default=0, help="Set 4 : 0 = (85 C, 15 S), 1 = (15 C, 85 S)")
    ap.add_argument("--returns", choices=["text"], default="text")
    a = ap.parse_args()
    inst = generate_instance(a.set, a.seed, a.variant, returns=a.returns)
    print(f"Set {a.set}, graine {a.seed} : {summary(inst, a.set)}")