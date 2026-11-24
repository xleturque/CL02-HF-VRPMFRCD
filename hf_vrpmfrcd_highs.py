"""MILP HF-VRPMFRCD, transcription numerotee (1)-(41) de Projet_CL02.mod.
Executez dans Spyder avec hf_vrpmfrcd_instances.py dans le meme dossier.
ATTENTION : transcription de la formulation OPL fournie, non preuve de validite
mathematique de l'article ; les limitations de cette formulation sont documentees.
Optimisation de construction Python uniquement : memes variables, domaines,
coefficients, contraintes et ordre de creation que le modele MILP de reference.
Aucune solution initiale externe, aucune contrainte supplementaire.
"""
import time
import highspy
import math
from hf_vrpmfrcd_instances import generate_instance, summary

# HiGHS cherche d abord une preuve d optimalite stricte (gap cible 0 %).
# Une solution est acceptee comme reference si son gap final <= 0,1 %.
GAP_CIBLE_HIGHS = 0
TOLERANCE_ACCEPTATION = 0.001


def demander_entier(message, minimum=None, maximum=None, defaut=None):
    while True:
        texte = input(message).strip()
        if not texte and defaut is not None:
            return defaut
        try:
            valeur = int(texte)
            if minimum is not None and valeur < minimum:
                raise ValueError
            if maximum is not None and valeur > maximum:
                raise ValueError
            return valeur
        except ValueError:
            print("Saisie invalide : veuillez entrer un entier autorise.")


def demander_limite():
    """La limite est obligatoire pour eviter une recherche sans fin."""
    while True:
        texte = input("Duree maximale de calcul en secondes : ").strip()
        if not texte:
            print("Veuillez saisir une duree maximale strictement positive.")
            continue
        try:
            valeur = float(texte.replace(",", "."))
            if valeur > 0 and valeur < float("inf"):
                return valeur
        except ValueError:
            pass
        print("Entrez une duree strictement positive (en secondes).")


def ordonner_arcs(arcs):
    """Reconstruit les chaines/cycles de chaque vehicule sans perdre d'arcs."""
    restants = list(arcs)
    trajets = []
    while restants:
        departs = {i for i, _ in restants}
        arrivees = {j for _, j in restants}
        origine = next((i for i in departs if i not in arrivees), restants[0][0])
        trajet = [origine]
        while True:
            position = next((k for k, (i, _) in enumerate(restants)
                             if i == trajet[-1]), None)
            if position is None:
                break
            _, destination = restants.pop(position)
            trajet.append(destination)
            if destination == origine:
                break
        trajets.append(trajet)
    return trajets


def resoudre_highs(instance, limite_temps, gap_relatif=GAP_CIBLE_HIGHS,
                   tolerance_acceptation=TOLERANCE_ACCEPTATION,
                   tmax=None, details_interface=False):
    debut_construction = time.perf_counter()
    if limite_temps is None or not (0 < float(limite_temps) < float("inf")):
        raise ValueError("Une duree maximale de calcul positive est obligatoire.")
    a = dict(instance)
    if tmax is not None:
        a['tmax'] = float(tmax)
    nd, ns, nc, nv, nw = (a[k] for k in ('nD','nS','nC','nV','nW'))
    D, S, C, V, W = range(nd), range(ns), range(nc), range(nv), range(nw)
    N1, N2 = range(nd+ns), range(nd+nc)
    p = [sum(a['df'][c][s] for c in C) for s in S]
    ret = [sum(a['pr'][c][s] for c in C) for s in S]
    # Agregats constants : ne pas reconstruire les memes coefficients pour
    # chaque arc / vehicule. Les donnees generees sont des entiers.
    df, pr = a['df'], a['pr']
    demande_client = [sum(df[c]) for c in C]
    bilan_client = [sum(pr[c][k] - df[c][k] for k in S) for c in C]
    # Pour les donnees externes non entieres, garder les sommes originales
    # de (17), (21) et (24) afin de ne pas changer les arrondis numeriques.
    flux_entiers = all(isinstance(q, int) or (isinstance(q, float) and q.is_integer())
                       for ligne in df for q in ligne)
    M = float(a['bigM'])
    T = float(a['tmax'])
    zeta = float(a['zeta'])
    m = highspy.Highs()
    m.setOptionValue("time_limit", float(limite_temps))
    m.setOptionValue("mip_rel_gap", float(gap_relatif))
    m.setOptionValue("mip_abs_gap", 0.0)
    m.setOptionValue("output_flag", False)
    def add(expr, ctname=""):
        return m.addConstr(expr, name=ctname)
    sm = m.qsum
    indices_entiers = []
    def var_dict(indices, name, vtype):
        variables = {k: m.addVariable(name=name + "_" + "_".join(map(str, k if isinstance(k, tuple) else (k,))),
                            lb=0.0, ub=1.0 if vtype == "B" else highspy.kHighsInf,
                            type=highspy.HighsVarType.kContinuous if vtype == "C" else highspy.HighsVarType.kInteger) for k in indices}
        if vtype != "C":
            indices_entiers.extend(var.index for var in variables.values())
        return variables
    # Indices Python 0-based ; indices OPL 1-based. Sites: nd + index.
    x1 = var_dict(((i,j,v) for i in N1 for j in N1 if i!=j for v in V), name='x1', vtype='B')
    x3 = var_dict(((i,j,v) for i in N2 for j in N2 if i!=j for v in V), name='x3', vtype='B')
    x5 = var_dict(((i,j,v) for i in N1 for j in N1 if i!=j for v in V), name='x5', vtype='B')
    x2 = var_dict(((i,j,w) for i in D for j in D if i!=j for w in W), name='x2', vtype='B')
    x4 = var_dict(((i,j,w) for i in D for j in D if i!=j for w in W), name='x4', vtype='B')
    z1 = var_dict(((i,s) for i in D for s in S), name='z1', vtype='B')
    z3 = var_dict(((i,c) for i in D for c in C), name='z3', vtype='B')
    z5 = var_dict(((i,s) for i in D for s in S), name='z5', vtype='B')
    y2 = var_dict(((i,j,s) for i in D for j in D if i!=j for s in S), name='y2', vtype='C')
    y4 = var_dict(((i,j,s) for i in D for j in D if i!=j for s in S), name='y4', vtype='C')
    n2 = var_dict(((i,j) for i in D for j in D if i!=j), name='n2truck', vtype='I')
    n4 = var_dict(((i,j) for i in D for j in D if i!=j), name='n4truck', vtype='I')
    s1 = var_dict(((i,v) for i in N1 for v in V), name='s1', vtype='C')
    s3 = var_dict(((i,v) for i in N2 for v in V), name='s3', vtype='C')
    s5 = var_dict(((i,v) for i in N1 for v in V), name='s5', vtype='C')
    tau1 = var_dict(D, name='tau1', vtype='C')
    tau2 = var_dict(D, name='tau2', vtype='C')
    tau3 = var_dict(D, name='tau3', vtype='C')
    tau4 = var_dict(D, name='tau4', vtype='C')
    b3 = var_dict(((i,v) for i in N2 for v in V), name='b3', vtype='C')
    # Le produit est cree seulement pour les arcs entrants client (i != nd+c),
    # car x3[nd+c,nd+c,v] = 0 dans le .mod.
    prod = var_dict(((l,i,c,v) for l in D for i in N2 for c in C
                               for v in V if i!=nd+c), name='produit3', vtype='B')
    def X(x,i,j,v):
        return 0 if i==j else x[i,j,v]
    def Y(y,i,j,k):
        return 0 if i==j else y[i,j,k]
    # Memes sommes lineaires que dans le modele de reference. Elles sont
    # construites une seule fois, puis reutilisees pour les contraintes de
    # conservation et d'affectation. Aucun changement de formulation.
    def degres(x, N):
        entrants = {(l,v): sm(x[i,l,v] for i in N if i != l)
                    for l in N for v in V}
        sortants = {(l,v): sm(x[l,j,v] for j in N if j != l)
                    for l in N for v in V}
        return entrants, sortants
    ent1, sor1 = degres(x1, N1)
    ent3, sor3 = degres(x3, N2)
    ent5, sor5 = degres(x5, N1)
    # (1) Fonction objectif.
    couts = {
      'P1':sm(a['c1'][i][j]*a['uV'][v]*var for (i,j,v),var in x1.items()),
      'P2':sm(a['cD'][i][j]*a['usetW']*var for (i,j),var in n2.items()),
      'P3':sm(a['c2'][i][j]*a['uV'][v]*var for (i,j,v),var in x3.items()),
      'P4':sm(a['cD'][i][j]*a['usetW']*var for (i,j),var in n4.items()),
      'P5':sm(a['c1'][i][j]*a['uV'][v]*var for (i,j,v),var in x5.items()),
    }
    m.setObjective(sm(couts.values()), highspy.ObjSense.kMinimize)
    # P1 (2)-(11)
    for j in S: add(sm(x1[i,nd+j,v] for i in N1 if i!=nd+j for v in V)==1, ctname=f'c02_{j}')
    for v in V: add(sm(p[j]*x1[i,nd+j,v] for i in N1 for j in S if i!=nd+j)<=a['qV'][v],ctname=f'c03_{v}')
    for i in S:
      for j in S:
       if i!=j:
        for v in V: add(s1[nd+i,v]+a['t1'][nd+i][nd+j]-M*(1-x1[nd+i,nd+j,v])<=s1[nd+j,v]) # (4)
    for i in D:
     for j in S:
      for v in V: add(a['t1'][i][nd+j]*x1[i,nd+j,v]<=s1[nd+j,v]) # (5)
    for i in D:
     for v in V:
      add(s1[i,v]==0); add(s3[i,v]==0); add(s5[i,v]==0) # (6)
    for l in N1:
     for v in V: add(ent1[l,v]==sor1[l,v]) # (7)
    for i in D: add(sm(p[j]*z1[i,j] for j in S)<=a['hD'][i]) # (8)
    for i in D:
     for j in S:
      for v in V: add(-z1[i,j]+sor1[i,v]+ent1[nd+j,v]<=1) # (9)
    for i in D:
     for j in S:
      for v in V: add(tau1[i]>=s1[nd+j,v]+a['t1'][nd+j][i]-M*(1-x1[nd+j,i,v])) # (10)
    for j in S: add(sm(z1[i,j] for i in D)<=1) # (11)
    # P2 (12)-(15)
    for i in D:
     for j in D:
      if i!=j:
       add(sm(y2[i,j,k] for k in S)<=a['qsetW']*n2[i,j]) # (12)
       add(sm(x2[i,j,w] for w in W)==n2[i,j]) # (13)
       for w in W: add(tau2[j]>=tau1[i]+a['tD'][i][j]-M*(1-x2[i,j,w])) # (14)
    for i in D: add(tau2[i]>=tau1[i]) # (15)
    # P3 (16)-(27)
    for c in C: add(sm(x3[i,nd+c,v] for i in N2 if i!=nd+c for v in V)==1) # (16)
    for v in V:
     if flux_entiers:
      add(sm(demande_client[c]*x3[i,nd+c,v] for i in N2 for c in C if i!=nd+c)<=a['qV'][v]) # (17)
     else:
      add(sm(df[c][k]*X(x3,i,nd+c,v) for i in N2 for c in C for k in S)<=a['qV'][v]) # (17)
    for i in C:
     for j in C:
      if i!=j:
       for v in V: add(s3[nd+i,v]+a['t2'][nd+i][nd+j]-M*(1-x3[nd+i,nd+j,v])<=s3[nd+j,v]) # (18)
    for i in D:
     for j in C:
      for v in V: add(a['t2'][i][nd+j]*x3[i,nd+j,v]+tau2[i]+zeta<=s3[nd+j,v]) # (19) OPL littéral
    for l in N2:
     for v in V: add(ent3[l,v]==sor3[l,v]) # (20)
    for i in D:
     if flux_entiers:
      add(sm(demande_client[c]*z3[i,c] for c in C)<=a['hD'][i]) # (21)
     else:
      add(sm(df[c][k]*z3[i,c] for c in C for k in S)<=a['hD'][i]) # (21)
    for i in D:
     for k in S: add(p[k]*z1[i,k]+sm(Y(y2,j,i,k) for j in D)-sm(Y(y2,i,j,k) for j in D)==sm(a['df'][c][k]*z3[i,c] for c in C)) # (22)
    for i in D:
     for j in C:
      for v in V: add(-z3[i,j]+sor3[i,v]+ent3[nd+j,v]<=1) # (23)
    for (l,i,c,v),b in prod.items():
        add(b<=x3[i,nd+c,v]); add(b<=z3[l,c]); add(b>=x3[i,nd+c,v]+z3[l,c]-1) # linearisation (24)
    for l in D:
     for v in V:
      if flux_entiers:
       add(b3[l,v]==sm(demande_client[c]*prod[l,i,c,v] for i in N2 for c in C if i!=nd+c)) # (24)
      else:
       add(b3[l,v]==sm(df[c][k]*prod[l,i,c,v] for i in N2 for c in C for k in S if i!=nd+c)) # (24)
    for i in N2:
     for j in C:
      if i!=nd+j:
       for v in V: add(b3[nd+j,v]>=b3[i,v]+bilan_client[j]-M*(1-x3[i,nd+j,v])) # (25)
    for i in D:
     for j in C:
      for v in V: add(tau3[i]>=s3[nd+j,v]+a['t2'][nd+j][i]-M*(1-x3[nd+j,i,v])) # (26)
    for j in C: add(sm(z3[i,j] for i in D)<=1) # (27)
    # P4 (28)-(31)
    for i in D:
     for j in D:
      if i!=j:
       add(sm(y4[i,j,k] for k in S)<=a['qsetW']*n4[i,j]) # (28)
       add(sm(x4[i,j,w] for w in W)==n4[i,j]) # (29)
       for w in W: add(tau4[j]>=tau3[i]+a['tD'][i][j]-M*(1-x4[i,j,w])) # (30)
    for i in D: add(tau4[i]>=tau3[i]) # (31)
    # P5 (32)-(41)
    for j in S: add(sm(x5[i,nd+j,v] for i in N1 if i!=nd+j for v in V)==1) # (32)
    for v in V: add(sm(ret[j]*x5[i,nd+j,v] for i in N1 for j in S if i!=nd+j)<=a['qV'][v]) # (33)
    for i in S:
     for j in S:
      if i!=j:
       for v in V: add(s5[nd+i,v]+a['t1'][nd+i][nd+j]-M*(1-x5[nd+i,nd+j,v])<=s5[nd+j,v]) # (34)
    for i in D:
     for j in S:
      for v in V: add(a['t1'][i][nd+j]*x5[i,nd+j,v]+tau4[i]+zeta<=s5[nd+j,v]) # (35) OPL littéral
    for l in N1:
     for v in V: add(ent5[l,v]==sor5[l,v]) # (36)
    for i in D: add(sm(ret[j]*z5[i,j] for j in S)<=a['hD'][i]) # (37)
    for i in D:
     for k in S: add(sm(a['pr'][c][k]*z3[i,c] for c in C)+sm(Y(y4,l,i,k) for l in D)-sm(Y(y4,i,j,k) for j in D)==ret[k]*z5[i,k]) # (38)
    for i in D:
     for j in S:
      for v in V: add(-z5[i,j]+sor5[i,v]+ent5[nd+j,v]<=1) # (39)
    for i in D:
     for j in S:
      for v in V: add(s5[nd+j,v]+a['t1'][nd+j][i]-M*(1-x5[nd+j,i,v])<=T) # (40)
    for j in S: add(sm(z5[i,j] for i in D)<=1) # (41)
    print("\nResolution HiGHS en cours...", flush=True)
    debut = time.perf_counter()
    temps_construction = debut - debut_construction
    nb_variables = m.getNumCol()
    nb_contraintes = m.getNumRow()
    m.run()
    duree = time.perf_counter() - debut
    statut_brut = m.getModelStatus()
    statut = m.modelStatusToString(statut_brut)
    info = m.getInfo()
    faisable = int(info.primal_solution_status) == 2 and m.getSolution().value_valid
    sol = m.getSolution() if faisable else None
    def fini(valeur):
        return float(valeur) if math.isfinite(valeur) and abs(valeur) < highspy.kHighsInf else None
    gap_highs = fini(info.mip_gap) if faisable else None
    borne = fini(info.mip_dual_bound)
    objectif = m.getObjectiveValue() if faisable else None
    gap = (max(0.0, objectif - borne) / (abs(objectif) + 1e-10)
           if faisable and borne is not None else None)
    statut_optimal = statut_brut == highspy.HighsModelStatus.kOptimal
    certifie = (faisable and
                (statut_optimal or
                 (gap is not None and float(gap) <= tolerance_acceptation + 1e-9)))
    resultat = dict(statut=statut, temps_s=round(duree, 3),
                    variables=nb_variables,
                    contraintes=nb_contraintes,
                    objectif=None, borne=borne, gap=gap, gap_highs=gap_highs,
                    couts={}, affectations={}, tournees={}, duree_operations=None,
                    certifie=certifie, faisable=faisable,
                    niveau=("OPTIMAL" if statut_optimal and faisable else
                            "OPTIMAL_TOLERANCE" if certifie else
                            "FEASIBLE" if faisable else "NO_SOLUTION"),
                    temps_posttraitement_s=None)

    if details_interface:
        resultat["temps_construction_s"] = temps_construction

    if faisable:
        # Memoriser les resultats AVANT toute seconde resolution.
        resultat['objectif'] = objectif
        resultat['couts'] = {k: m.val(expr) for k, expr in couts.items()}
        resultat['affectations'] = {
            'P1': {s+1: [i+1 for i in D if m.val(z1[i,s]) > .5] for s in S},
            'P3': {c+1: [i+1 for i in D if m.val(z3[i,c]) > .5] for c in C},
            'P5': {s+1: [i+1 for i in D if m.val(z5[i,s]) > .5] for s in S}}
        # Extraire les tournees depuis la solution de cout, avant le post-traitement.
        # Les numeros affiches correspondent aux indices OPL (a partir de 1).
        arcs_retour = []
        for nom, variables in (('P1', x1), ('P3', x3), ('P5', x5)):
            par_vehicule = {}
            for (i, j, v), arc in variables.items():
                if m.val(arc) > 0.5:
                    par_vehicule.setdefault(v + 1, []).append((i + 1, j + 1))
                    if nom == 'P5' and i >= nd and j < nd:
                        arcs_retour.append((i, j, v))
            resultat['tournees'][nom] = {
                v: ordonner_arcs(arcs) for v, arcs in sorted(par_vehicule.items())
            }
        for nom, variables in (('P2', x2), ('P4', x4)):
            transferts = {}
            for (i, j, w), arc in variables.items():
                if m.val(arc) > 0.5:
                    transferts.setdefault(w + 1, []).append((i + 1, j + 1))
            resultat['tournees'][nom] = transferts
        if details_interface:
            # Extraire les flux de la solution de cout. Aucun changement du MILP.
            resultat['flux_transferts'] = {
                nom: {(i + 1, j + 1): sum(m.val(variables[i,j,k]) for k in S)
                      for i in D for j in D if i != j}
                for nom, variables in (('P2', y2), ('P4', y4))}
        if arcs_retour:
            # Deuxieme optimisation : conserver toutes les decisions discretes
            # et minimiser la date de fin du processus P5.
            decisions = [(i, round(sol.col_value[i]))
                         for i in indices_entiers]
            for i, valeur in decisions:
                m.changeColBounds(i, valeur, valeur)
            fin = m.addVariable(name="fin_operations_P5", lb=0.0)
            for j,i,v in arcs_retour:
                m.addConstr(fin >= s5[j,v] + a['t1'][j][i])
            m.setObjective(fin, highspy.ObjSense.kMinimize)
            debut_post = time.perf_counter()
            m.clearSolver()
            m.setOptionValue("time_limit", 5.0)
            m.run()
            sol_temps = m.getSolution() if int(m.getInfo().primal_solution_status) == 2 and m.getSolution().value_valid else None
            resultat['temps_posttraitement_s'] = round(time.perf_counter()-debut_post, 3)
            if sol_temps is not None:
                resultat['duree_operations'] = m.val(fin)
                if details_interface:
                    # Dates de la meme solution temporelle que la fin P5.
                    resultat['tau'] = {nom: [m.val(var[i]) for i in D]
                                       for nom, var in (('tau1', tau1), ('tau2', tau2),
                                                        ('tau3', tau3), ('tau4', tau4))}
                    resultat['statut_posttraitement'] = m.modelStatusToString(m.getModelStatus())

    print("\n" + "=" * 56)
    if faisable:
        if statut_optimal:
            print("SOLUTION OPTIMALE PROUVEE")
        elif certifie:
            print("SOLUTION OPTIMALE A LA TOLERANCE DE {:.3f} %".format(100*tolerance_acceptation))
        else:
            print("MEILLEURE SOLUTION FAISABLE — OPTIMALITE NON PROUVEE")
        print("Cout total : {:.3f}".format(resultat['objectif']))
        if resultat['duree_operations'] is not None:
            print("Duree des operations (fin P5) : {:.3f} {}".format(
                resultat['duree_operations'], a.get('_unite_temps', 'unites de temps')))
        else:
            print("Duree des operations : indisponible")
        print("Couts par processus :")
        for nom, valeur in resultat['couts'].items():
            print("  {} : {:.3f}".format(nom, valeur))
        print("Tournees et transferts (numerotation OPL) :")
        for nom in ('P1', 'P2', 'P3', 'P4', 'P5'):
            groupes = resultat['tournees'].get(nom, {})
            lignes = []
            for vehicule, trajets in groupes.items():
                if nom in ('P2', 'P4'):
                    for depart, arrivee in trajets:
                        lignes.append("Vehicule W{} : Dock {} -> Dock {}".format(
                            vehicule, depart, arrivee))
                else:
                    for trajet in trajets:
                        lignes.append("Vehicule V{} : {}".format(
                            vehicule, ' -> '.join(
                                'D{}'.format(noeud) if noeud <= nd else
                                '{}{}'.format('C' if nom == 'P3' else 'S', noeud - nd)
                                for noeud in trajet)))
            for ligne in lignes or ["Aucun trajet"]:
                print("  {} : {}".format(nom, ligne))
        if borne is not None:
            print("Meilleure borne Branch-and-Cut HiGHS : {:.3f}".format(borne))
        if borne is not None:
            print("Ecart absolu avec la borne : {:.3f}".format(max(0.0, resultat["objectif"] - float(borne))))
        if gap is not None:
            print("Gap relatif final : {:.4f} %".format(100*gap))
    else:
        print("AUCUNE SOLUTION FAISABLE TROUVEE — voir statut")
    print("Temps de calcul HiGHS (cout) : {:.3f} s".format(resultat['temps_s']))
    print("Statut HiGHS :", statut)
    print("=" * 56)
    m.clear()
    return resultat


if __name__ == '__main__':
    print("=" * 56)
    print("       HF-VRPMFRCD — SOLVEUR HiGHS")
    print("=" * 56)
    set_id = demander_entier("Choisir le Set (1 a 5) : ", 1, 5)
    graine = demander_entier("Choisir la graine : ", minimum=0)
    limite = demander_limite()
    instance = generate_instance(set_id, seed=graine)
    instance['_unite_temps'] = 'min' if set_id <= 3 else 'h'
    print("\nINSTANCE SELECTIONNEE — Set {} / graine {}".format(set_id, graine))
    print(summary(instance, set_id))
    print("Limite de calcul : {} s".format(limite))
    print("Objectif HiGHS : prouver l optimalite (gap cible 0 %)")
    print("Tolerance d acceptation : {:.3f} %".format(100 * TOLERANCE_ACCEPTATION))
    resultat = resoudre_highs(instance, limite_temps=limite,
                             gap_relatif=GAP_CIBLE_HIGHS,
                             tolerance_acceptation=TOLERANCE_ACCEPTATION)
