"""
Projet CL02 - HF-VRPMFRCD
Optimisation par recuit simule (SA), recuit simule avec descente a voisinages
variables (SAVND) et recuit simule avec voisinages adaptatifs (ANSA).

Reference : Yu, Anh, Gunawan et Han (2024), Expert Systems with Applications.

Mode d'emploi dans Spyder :
    - Executer ce fichier (F5) sans argument de ligne de commande.
    - Saisir le numero du set (1 a 5) et la graine de generation.
    - Consulter les resultats des trois methodes sur la meme instance.
    - En fin d'execution, choisir ou non d'exporter les donnees vers CPLEX.

La comparaison interactive utilise une execution par methode, avec un budget
de 30 secondes par methode. Les parametres complets de l'article sont definis
par paper_params() pour des campagnes experimentales approfondies.
"""
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

EPS = 1e-9
VERSION_CONTROLE = "CL02-TMAX-2026-11-23-v2"

# =====================================================================
# 1. DONNEES ET FONCTION OBJECTIF
# =====================================================================
def _vec(x, n, name):
    try:
        lst = [x[i] for i in range(1, n + 1)] if isinstance(x, dict) else list(x)
    except KeyError as e:
        raise ValueError(f"{name} : indice {e} manquant")
    if len(lst) != n:
        raise ValueError(f"{name} : taille {len(lst)} au lieu de {n}")
    return lst

def _mat(M, r, c, name):
    try:
        rows = ([[M[i, j] for j in range(1, c + 1)] for i in range(1, r + 1)]
                if isinstance(M, dict) else [list(row) for row in M])
    except KeyError as e:
        raise ValueError(f"{name} : indice {e} manquant")
    if len(rows) != r or any(len(row) != c for row in rows):
        raise ValueError(f"{name} : dimensions incorrectes")
    return rows

class Problem:
    def __init__(self, nC, nD, nS, nV, nW, tmax, zeta, c1, t1, c2, t2, cD, tD,
                 df, pr, hD, qV, uV, qsetW, usetW, bigM=None,
                 pen_load=1000.0, pen_time=100.0):
        self.nD, self.nS, self.nC, self.nV, self.nW = nD, nS, nC, nV, nW
        n1, n2 = nD + nS, nD + nC
        self.n1, self.n2 = n1, n2
        self.Tmax, self.zeta, self.QW, self.uW = tmax, zeta, qsetW, usetW
        self.pen_load, self.pen_time = pen_load, pen_time
        pad = lambda v: [0] + list(v)
        padm = lambda M: [[0] * (len(M[0]) + 1)] + [[0] + list(r) for r in M]
        self.Q = pad(_vec(qV, nV, "qV"))
        self.u = pad(_vec(uV, nV, "uV"))
        self.h = pad(_vec(hD, nD, "hD"))
        self.d = padm(_mat(df, nC, nS, "df"))
        self.pret = padm(_mat(pr, nC, nS, "pr"))
        self.c1, self.t1 = padm(_mat(c1, n1, n1, "c1")), padm(_mat(t1, n1, n1, "t1"))
        self.c2, self.t2 = padm(_mat(c2, n2, n2, "c2")), padm(_mat(t2, n2, n2, "t2"))
        self.cD, self.tD = padm(_mat(cD, nD, nD, "cD")), padm(_mat(tD, nD, nD, "tD"))
        S, C = range(1, nS + 1), range(1, nC + 1)
        self.p = [0] + [sum(self.d[c][k] for c in C) for k in S]
        self.dret = [0] + [sum(self.pret[c][k] for c in C) for k in S]
        self.dtot = [0] + [sum(self.d[c][k] for k in S) for c in C]
        self.load1 = [0.0] * (n1 + 1)
        self.load5 = [0.0] * (n1 + 1)
        self.load3 = [0.0] * (n2 + 1)
        for s in S:
            self.load1[nD + s] = self.p[s]
            self.load5[nD + s] = self.dret[s]
        for c in C:
            self.load3[nD + c] = self.dtot[c]
        self.nZ = nV
        self._cache_phases = {1: {}, 3: {}, 5: {}}

    def _decode(self, r, vl, load):
        nD, Q, nveh = self.nD, self.Q, len(vl)
        routes, cur, cl, viol, ptr = [], [], 0.0, 0.0, 0
        dock = r[0] if 1 <= r[0] <= nD else 1
        for e in r:
            if e <= nD:
                if cur:
                    routes.append((dock, cur, vl[ptr])); ptr += 1; cur = []; cl = 0.0
                if e:
                    dock = e
            else:
                ld = load[e]
                if ptr >= nveh:
                    viol += ld
                    continue
                cap = Q[vl[ptr]]
                if cur and cl + ld > cap:
                    routes.append((dock, cur, vl[ptr])); ptr += 1; cur = []; cl = 0.0
                    if ptr >= nveh:
                        viol += ld
                        continue
                    cap = Q[vl[ptr]]
                if not cur and ld > cap:
                    viol += ld - cap
                cur.append(e)
                cl += ld
        if cur:
            routes.append((dock, cur, vl[ptr]))
        return routes, viol

    @staticmethod
    def _route_cd(dock, nodes, dist, tm):
        c = t = 0.0
        prev = dock
        for n in nodes:
            c += dist[prev][n]
            t += tm[prev][n]
            prev = n
        return c + dist[prev][dock], t + tm[prev][dock]

    def _transfer(self, theta, xi):
        """Transferts P2/P4 : coût, camions et quantités par liaison."""
        nD, nS, cD = self.nD, self.nS, self.cD
        delta = [[theta[i][j] - xi[i][j] for j in range(nS + 1)] for i in range(nD + 1)]
        flow = {}
        for i in range(1, nD + 1):
            for j in range(1, nS + 1):
                while delta[i][j] > EPS:
                    h, bd = None, 1e18
                    for k in range(1, nD + 1):
                        if k != i and delta[k][j] < -EPS and cD[i][k] < bd:
                            h, bd = k, cD[i][k]
                    if h is None:
                        break
                    beta = 0.0
                    for pp in range(1, nS + 1):
                        val = min(delta[i][pp], -delta[h][pp])
                        if val > 0:
                            beta += val
                            delta[i][pp] -= val
                            delta[h][pp] += val
                    flow[(i, h)] = flow.get((i, h), 0.0) + beta
        cost, trucks = 0.0, {}
        for (i, h), b in flow.items():
            n = math.ceil(b / self.QW - 1e-9)
            trucks[(i, h)] = n
            cost += cD[i][h] * self.uW * n
        return cost, trucks, flow

    def _phase(self, numero, routes_tableau, vehicules_tableau):
        """Evaluation mise en cache de P1, P3 ou P5."""
        cache = self._cache_phases[numero]
        cle = (tuple(routes_tableau), tuple(vehicules_tableau))
        if cle in cache:
            return cache[cle]
        nD, nS = self.nD, self.nS
        charge = self.load3 if numero == 3 else self.load1 if numero == 1 else self.load5
        R, violations = self._decode(routes_tableau, vehicules_tableau, charge)
        distances, temps = (self.c2, self.t2) if numero == 3 else (self.c1, self.t1)
        flux = [[0.0] * (nS + 1) for _ in range(nD + 1)]
        retours = [[0.0] * (nS + 1) for _ in range(nD + 1)] if numero == 3 else None
        chargements_dock = [0.0] * (nD + 1)
        fin_tournee = [0.0] * (nD + 1)
        infos = []
        cout_total = 0.0
        for dock, noeuds, veh in R:
            distance, duree = self._route_cd(dock, noeuds, distances, temps)
            cout = distance * self.u[veh]
            cout_total += cout
            fin_tournee[dock] = max(fin_tournee[dock], duree)
            infos.append((dock, noeuds, veh, cout, duree))
            for noeud in noeuds:
                j = noeud - nD
                if numero == 1:
                    q = self.p[j]
                    flux[dock][j] += q
                    chargements_dock[dock] += q
                elif numero == 5:
                    q = self.dret[j]
                    flux[dock][j] += q
                    chargements_dock[dock] += q
                else:
                    chargements_dock[dock] += self.dtot[j]
                    commandes, invendus = self.d[j], self.pret[j]
                    for k in range(1, nS + 1):
                        flux[dock][k] += commandes[k]
                        retours[dock][k] += invendus[k]
        result = (cout_total, infos, fin_tournee, flux, retours, chargements_dock, violations)
        if len(cache) >= 256:
            cache.clear()
        cache[cle] = result
        return result

    def evaluate(self, A, detail=False):
        """Fonction objectif : coûts P1-P5 et pénalités."""
        nD, nW, zeta, Tmax = self.nD, self.nW, self.zeta, self.Tmax
        E1 = self._phase(1, A[0], A[1])
        E3 = self._phase(3, A[2], A[3])
        E5 = self._phase(5, A[4], A[5])
        G1, I1, tau1, theta2, _, ld1, w1 = E1
        G3, I3, _, xi2, theta4, ld3, w3 = E3
        G5, I5, _, xi4, _, ld5, w5 = E5
        pen_units = w1 + w3 + w5

        G2, trucks2, flow2 = self._transfer(theta2, xi2)
        tau2 = tau1[:]
        for (i, k), nt in trucks2.items():
            tau2[k] = max(tau2[k], tau1[i] + self.tD[i][k])
            if nt > nW:
                pen_units += nt - nW

        tau3 = [0.0] * (nD + 1)
        for dock, nodes, veh, cout, duree in I3:
            fin = tau2[dock] + zeta + duree
            tau3[dock] = max(tau3[dock], fin)

        G4, trucks4, flow4 = self._transfer(theta4, xi4)
        tau4 = tau3[:]
        for (i, k), nt in trucks4.items():
            tau4[k] = max(tau4[k], tau3[i] + self.tD[i][k])
            if nt > nW:
                pen_units += nt - nW

        # Une seule definition de l'horizon, commune a la penalite et au rapport.
        # Inclure P1/P2/P3/P4 meme si aucune tournee P5 ne part d'un dock.
        fins_p5 = [tau4[dock] + zeta + duree
                   for dock, nodes, veh, cout, duree in I5]
        duree_operations = max([0.0] + tau1[1:] + tau2[1:]
                               + tau3[1:] + tau4[1:] + fins_p5)
        if not math.isfinite(duree_operations) or not math.isfinite(Tmax):
            raise ValueError("Horizon temporel invalide : NaN ou infini.")
        depassement_brut = max(0.0, duree_operations - Tmax)
        respecte_tmax = depassement_brut <= EPS
        pen_time = 0.0 if respecte_tmax else depassement_brut
        for i in range(1, nD + 1):
            h = self.h[i]
            pen_units += max(0.0, ld1[i] - h) + max(0.0, ld3[i] - h) + max(0.0, ld5[i] - h)

        total = G1 + G2 + G3 + G4 + G5
        self.last_pen = self.pen_load * pen_units + self.pen_time * pen_time
        obj = total + self.last_pen
        if not detail:
            return obj

        feasible = (pen_units < EPS and respecte_tmax)
        # Verification explicite de la coherence avant toute diffusion des resultats.
        if feasible != (pen_units < EPS and duree_operations <= Tmax + EPS):
            raise RuntimeError("Faisabilite incoherente avec les contraintes")
        if duree_operations > Tmax + EPS and pen_time <= EPS:
            raise RuntimeError("Depassement de Tmax sans penalite temporelle")
        return dict(
            obj=obj, cost=total, feasible=feasible,
            duree_operations=duree_operations, respecte_tmax=respecte_tmax,
            P1=I1, P3=I3, P5=I5,
            G=dict(P1=G1, P2=G2, P3=G3, P4=G4, P5=G5),
            trucks2=trucks2, trucks4=trucks4, flow2=flow2, flow4=flow4,
            tau=dict(tau1=tau1[1:], tau2=tau2[1:], tau3=tau3[1:], tau4=tau4[1:]),
            viol_units=pen_units, viol_time=pen_time,
            depassement_tmax=pen_time,
        )

    def initial_solution(self, rng):
        """Solution initiale : plus proche voisin modifié."""
        nD, nV, Tmax = self.nD, self.nV, self.Tmax
        procs = [
            (range(nD + 1, nD + self.nS + 1), self.c1, self.t1, self.load1),
            (range(nD + 1, nD + self.nC + 1), self.c2, self.t2, self.load3),
            (range(nD + 1, nD + self.nS + 1), self.c1, self.t1, self.load5)
        ]
        best, fbest = None, float("inf")
        for i in range(1, nD + 1):
            A = []
            for nodes, dist, tm, load in procs:
                VR = list(range(1, nV + 1)); NR = set(nodes); ZR = self.nZ
                eta, cl, rt = i, 0.0, 0.0
                v = rng.choice(VR); VR.remove(v)
                Sr, Sv = [i], [v]
                while NR:
                    cand = [n for n in NR if cl + load[n] <= self.Q[v]
                            and rt + tm[eta][n] + tm[n][i] <= Tmax]
                    if cand:
                        n = min(cand, key=lambda x: dist[eta][x])
                    elif VR and ZR > 0:
                        v = rng.choice(VR); VR.remove(v); ZR -= 1
                        Sr.append(0); Sv.append(v)
                        eta, cl, rt = i, 0.0, 0.0
                        continue
                    else:
                        n = min(NR, key=lambda x: dist[eta][x])
                    Sr.append(n); NR.remove(n)
                    cl += load[n]; rt += tm[eta][n]; eta = n
                Sr += [x for x in range(1, nD + 1) if x != i] + [0] * ZR
                Sv += VR
                A += [Sr, Sv]
            f = self.evaluate(A)
            if f < fbest:
                best, fbest = A, f
        return best, fbest

    def neighbor(self, A, k, rng):
        idx = rng.randrange(6)
        a = A[idx][:]
        if not MOVES[k](a, rng):
            return None
        if idx % 2 == 0 and not (1 <= a[0] <= self.nD):
            return None
        B = list(A)
        B[idx] = a
        return B

# =====================================================================
# 2. VOISINAGES
# =====================================================================
def swap11(a, rng):
    if len(a) < 2: return False
    i, j = rng.sample(range(len(a)), 2)
    a[i], a[j] = a[j], a[i]
    return True

def swap22(a, rng):
    n = len(a)
    if n < 4: return False
    i, j = rng.randrange(n - 1), rng.randrange(n - 1)
    if abs(i - j) < 2: return False
    a[i:i + 2], a[j:j + 2] = a[j:j + 2], a[i:i + 2]
    return True

def two_opt(a, rng):
    if len(a) < 2: return False
    i, j = sorted(rng.sample(range(len(a)), 2))
    a[i:j + 1] = a[i:j + 1][::-1]
    return True

def shift10(a, rng):
    if len(a) < 2: return False
    x = a.pop(rng.randrange(len(a)))
    a.insert(rng.randrange(len(a) + 1), x)
    return True

def shift20(a, rng):
    if len(a) < 3: return False
    i = rng.randrange(len(a) - 1)
    blk = a[i:i + 2]; del a[i:i + 2]
    j = rng.randrange(len(a) + 1)
    a[j:j] = blk
    return True

MOVES = [swap11, swap22, two_opt, shift10, shift20]
MOVE_NAMES = ["Swap(1,1)", "Swap(2,2)", "2-Opt", "Shift(1,0)", "Shift(2,0)"]

# =====================================================================
# 3. ALGORITHMES SA / SAVND / ANSA
# =====================================================================
@dataclass
class Params:
    T0: float = 50.0
    Tf: float = 35.0
    alpha: float = 0.997
    eta_sa: int = 1000
    K: float = 1.0
    eta_ls: int = 45
    eta_ni: int = 7
    eps: float = 0.015
    time_limit: Optional[float] = None
    seed: Optional[int] = 0
    verbose: bool = False
    segment: Optional[int] = None
    rho: float = 0.3
    p_min: float = 0.05
    reward_improve: float = 1.0
    reward_best: float = 2.0
    repair_iters: int = 20000
    repair_tries: int = 8

def paper_params(prob, method="savnd"):
    B = prob.nS + prob.nC
    if method in ("sa", "ansa"):
        return Params(T0=50, Tf=25, alpha=0.999, eta_sa=3000 * B)
    return Params(T0=50, Tf=35, alpha=0.997, eta_sa=2500 * B, eta_ls=45, eta_ni=7, eps=0.015)

def quick_params(prob, **kw):
    return Params(T0=50, Tf=35, alpha=0.99, eta_sa=200 * (prob.nS + prob.nC), **kw)

def repair(prob, X, rng, max_iter, deadline=None, noise=0.1):
    """Réparation des violations de capacité et d'horizon."""
    prob.evaluate(X); v = prob.last_pen
    bestX, bestv = X, v
    it = 0
    while bestv > EPS and it < max_iter:
        it += 1
        if deadline and (it & 255) == 0 and time.time() > deadline:
            break
        Y = prob.neighbor(X, rng.randrange(len(MOVES)), rng)
        if Y is None:
            continue
        prob.evaluate(Y); vY = prob.last_pen
        if vY <= v or rng.random() < math.exp(-(vY - v) / max(noise * v, 1e-6)):
            X, v = Y, vY
            if v < bestv:
                bestX, bestv = X, v
    return bestX

def localsearch_vnd(prob, Y, fY, P, rng, deadline=None):
    """Descente à voisinages variables."""
    Xp, fXp = Y, fY
    Xv, fXv = Y, fY
    k = 0
    while k < len(MOVES):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        Yp, fYp = Xp, fXp
        ni = 0
        for index_essai in range(P.eta_ls):
            if deadline is not None and (index_essai & 15) == 0 and time.perf_counter() >= deadline:
                return Xv, fXv
            Z = prob.neighbor(Yp, k, rng)
            if Z is None:
                continue
            Yp, fYp = Z, prob.evaluate(Z)
            if fYp < fXp:
                Xp, fXp, ni = Yp, fYp, 0
                break
            elif ni < P.eta_ni and (fYp - fXp) / max(abs(fXp), EPS) < P.eps:
                Xp, fXp, ni = Yp, fYp, ni + 1
            elif ni >= P.eta_ni:
                Yp, fYp, ni = Xp, fXp, 0
        if fXp < fXv:
            Xv, fXv, k = Xp, fXp, 0
        else:
            k += 1
    return Xv, fXv

def solve(prob, method="savnd", P=None):
    """Retourne (solution, objectif, historique, statistiques)."""
    P = P or paper_params(prob, method)
    rng = random.Random(P.seed)
    use_vnd = (method == "savnd")
    adaptive = (method == "ansa")
    nM = len(MOVES)
    weights = [1.0 / nM] * nM
    probs = weights[:]
    used, score = [0] * nM, [0.0] * nM
    seg = P.segment or P.eta_sa
    t_start = time.perf_counter()
    deadline_vnd = t_start + P.time_limit if P.time_limit else None
    X, fX = prob.initial_solution(rng)

    if P.repair_iters:
        dl = time.time() + 0.3 * P.time_limit if P.time_limit else None
        Xb, vb = None, float("inf")
        for attempt in range(P.repair_tries):
            if attempt:
                X, _ = prob.initial_solution(rng)
            prob.evaluate(X)
            if prob.last_pen > EPS:
                X = repair(prob, X, rng, P.repair_iters, dl)
                prob.evaluate(X)
            if prob.last_pen < vb:
                Xb, vb = X, prob.last_pen
            if vb <= EPS or (dl and time.time() > dl):
                break
        X = Xb
        fX = prob.evaluate(X)

    best, fbest = X, fX
    hist = [(0.0, fbest)]
    T, it = P.T0, 0
    while T > P.Tf:
        it += 1
        k = rng.choices(range(nM), weights=probs)[0] if adaptive else rng.randrange(nM)
        used[k] += 1
        Y = prob.neighbor(X, k, rng)
        if Y is not None:
            fY = prob.evaluate(Y)
            if fY < fX:
                if use_vnd:
                    Y, fY = localsearch_vnd(prob, Y, fY, P, rng, deadline_vnd)
                X, fX = Y, fY
                score[k] += P.reward_improve
                if fX < fbest:
                    best, fbest = X, fX
                    score[k] += P.reward_best
                    hist.append((time.perf_counter() - t_start, fbest))
            elif rng.random() < math.exp(-(fY - fX) / (P.K * T)):
                X, fX = Y, fY

        if adaptive and it % seg == 0:
            for j in range(nM):
                if used[j]:
                    weights[j] = (1 - P.rho) * weights[j] + P.rho * score[j] / used[j]
            tot = sum(weights)
            weights = [w / tot for w in weights] if tot > 0 else [1.0 / nM] * nM
            probs = [max(P.p_min, w) for w in weights]
            sp = sum(probs)
            probs = [x / sp for x in probs]
            used, score = [0] * nM, [0.0] * nM
        if it % P.eta_sa == 0:
            T *= P.alpha
            if P.verbose:
                extra = "  p=" + str([round(x, 2) for x in probs]) if adaptive else ""
                print(f"T={T:.3f} it={it} courant={fX:.2f} meilleur={fbest:.2f}{extra}")
        if P.time_limit and (it & 255) == 0 and time.perf_counter() - t_start > P.time_limit:
            break

    info = {"iterations": it, "move_probs": dict(zip(MOVE_NAMES, probs)) if adaptive else None}
    return best, fbest, hist, info

# =====================================================================
# 4. AFFICHAGE ET EXPORT
# =====================================================================
def report(prob, A, unite_temps=""):
    I = prob.evaluate(A, detail=True)
    print(f"Objectif = {I['obj']:.2f} | Coût = {I['cost']:.2f} | Faisable = {I['feasible']}")
    print("Coûts par phase :", {k: round(v, 2) for k, v in I["G"].items()})
    for name in ("P1", "P3", "P5"):
        for dock, nodes, veh, c, t in I[name]:
            print(f"{name} : D{dock} -> {nodes} -> D{dock} (véhicule {veh}, coût {c:.2f}, durée {t:.2f})")
    for phase in ("P2", "P4"):
        print(f"{phase} : camions = {I['trucks2' if phase == 'P2' else 'trucks4']}")
        print(f"{phase} : quantités = {I['flow2' if phase == 'P2' else 'flow4']}")
    print("Temps de fin :", {k: [round(x, 2) for x in v] for k, v in I["tau"].items()})
    print(f"Durée totale = {I['duree_operations']:.2f} {unite_temps} / Tmax = {prob.Tmax:g}")
    if not I["feasible"]:
        print(f"Violations : charge={I['viol_units']:.1f}, temps={I['viol_time']:.1f}")
    return I

SECONDES_PAR_METHODE = 20

def exporter_dat_pour_cplex(donnees, chemin, set_id, graine):
    """Export CPLEX avec conversion des temps en minutes."""
    from hf_vrpmfrcd_instances import UNITES_TEMPS, write_dat
    import re
    unite, facteur = UNITES_TEMPS[set_id]
    write_dat(donnees, str(chemin), f"Set {set_id}, graine {graine}",
              facteur_temps=facteur, unite_source=unite)
    texte = chemin.read_text(encoding="utf-8")
    correspondance = {
        "bigM": "BigM", "tmax": "Tmax", "qV": "Q", "uV": "u",
        "qsetW": "QW", "usetW": "uW", "hD": "h", "df": "d", "pr": "pret"
    }
    for origine, destination in correspondance.items():
        texte = re.sub(r"\b" + re.escape(origine) + r"(?=\s*=)", destination, texte)
    chemin.write_text(texte, encoding="utf-8")

def demander_entier(invite, minimum, maximum=None):
    while True:
        try:
            n = int(input(invite).strip())
            if n >= minimum and (maximum is None or n <= maximum):
                return n
        except ValueError:
            pass
        print("Saisie incorrecte : veuillez entrer un entier autorisé.")

def demander_oui_non(message):
    while True:
        r = input(message).strip().lower()
        if r in ("o", "oui", "y", "yes"): return True
        if r in ("n", "non", "", "no"): return False
        print("Répondre par o (oui) ou n (non).")

def main():
    from hf_vrpmfrcd_instances import UNITES_TEMPS, generate_instance, summary
    print("=" * 65)
    print("CL02 - Comparaison SA / SAVND / ANSA")
    print("Version controle :", VERSION_CONTROLE)
    print("Fichier execute :", Path(__file__).resolve())
    print("=" * 65)
    set_id = demander_entier("Choisir un Set (1 à 5, 0 pour quitter) : ", 0, 5)
    if set_id == 0:
        return
    seed = demander_entier("Graine de génération (entier >= 0) : ", 0)
    donnees = generate_instance(set_id, seed=seed)
    prob = Problem(**donnees)
    unite_temps = UNITES_TEMPS[set_id][0]
    print("Instance :", summary(donnees, set_id))
    resultats = []

    for methode in ("sa", "savnd", "ansa"):
        reglages = quick_params(prob, seed=0, time_limit=SECONDES_PAR_METHODE)
        debut = time.perf_counter()
        solution, score, _, statistiques = solve(prob, methode, reglages)
        duree = time.perf_counter() - debut
        details = prob.evaluate(solution, detail=True)
        # Double controle des valeurs qui alimentent la console et Streamlit.
        attendu = (details["viol_units"] < EPS and
                   details["duree_operations"] <= prob.Tmax + EPS)
        if details["feasible"] != attendu:
            raise RuntimeError("Evaluation finale incoherente avec Tmax")
        if not math.isclose(details["obj"], score, rel_tol=1e-8, abs_tol=1e-6):
            raise RuntimeError("Score du solveur different du score reevalue")
        if details["duree_operations"] > prob.Tmax + EPS and (details["viol_time"] <= EPS or details["feasible"]):
            raise RuntimeError("Depassement de Tmax non penalise")
        resultats.append((methode, solution, details, duree, statistiques))
        print(
            f"{methode.upper():6s} | coût={details['cost']:.2f} | "
            f"objectif={details['obj']:.2f} | faisable={details['feasible']} | "
            f"opérations={details['duree_operations']:.1f} {unite_temps} "
            f"/ Tmax={prob.Tmax:g} {unite_temps} | "
            f"dépassement={details['viol_time']:.1f} {unite_temps} | "
            f"calcul={duree:.2f} s"
        )

    valides = [res for res in resultats if res[2]["feasible"]]
    meilleur = min(valides, key=lambda res: res[2]["cost"]) if valides else min(resultats, key=lambda res: res[2]["obj"])
    print(f"\nMeilleure méthode : {meilleur[0].upper()}")
    report(prob, meilleur[1], unite_temps=unite_temps)

    if demander_oui_non("\nExporter les données vers CPLEX ? (o/n) : "):
        chemin = Path(__file__).resolve().parent / f"Set{set_id}_graine{seed}_CPLEX.dat"
        exporter_dat_pour_cplex(donnees, chemin, set_id, seed)
        print("Fichier créé :", chemin)

if __name__ == "__main__":
    main()
