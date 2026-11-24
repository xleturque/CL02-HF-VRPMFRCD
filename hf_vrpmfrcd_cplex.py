"""MILP HF-VRPMFRCD, transcription numerotee (1)-(41) de Projet_CL02.mod.
Executez dans Spyder avec hf_vrpmfrcd_instances.py dans le meme dossier.
ATTENTION : transcription de la formulation OPL fournie, non preuve de validite
mathematique de l'article ; les limitations de cette formulation sont documentees.
"""
import time
from docplex.mp.model import Model
from hf_vrpmfrcd_instances import generate_instance, summary

# CPLEX cherche d abord une preuve d optimalite stricte (gap cible 0 %).
# Une solution est acceptee comme reference si son gap final <= 0,1 %.
GAP_CIBLE_CPLEX = 0
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


def resoudre_cplex(instance, limite_temps, gap_relatif=GAP_CIBLE_CPLEX,
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
    m = Model('HF_VRPMFRCD_OPL_1_41')
    m.parameters.timelimit = float(limite_temps)
    m.parameters.mip.tolerances.mipgap = float(gap_relatif)
    add = m.add_constraint
    sm = m.sum
    # Indices Python 0-based ; indices OPL 1-based. Sites: nd + index.
    x1 = m.binary_var_dict(((i,j,v) for i in N1 for j in N1 if i!=j for v in V), name='x1')
    x3 = m.binary_var_dict(((i,j,v) for i in N2 for j in N2 if i!=j for v in V), name='x3')
    x5 = m.binary_var_dict(((i,j,v) for i in N1 for j in N1 if i!=j for v in V), name='x5')
    x2 = m.binary_var_dict(((i,j,w) for i in D for j in D if i!=j for w in W), name='x2')
    x4 = m.binary_var_dict(((i,j,w) for i in D for j in D if i!=j for w in W), name='x4')
    z1 = m.binary_var_dict(((i,s) for i in D for s in S), name='z1')
    z3 = m.binary_var_dict(((i,c) for i in D for c in C), name='z3')
    z5 = m.binary_var_dict(((i,s) for i in D for s in S), name='z5')
    y2 = m.continuous_var_dict(((i,j,s) for i in D for j in D if i!=j for s in S), name='y2')
    y4 = m.continuous_var_dict(((i,j,s) for i in D for j in D if i!=j for s in S), name='y4')
    n2 = m.integer_var_dict(((i,j) for i in D for j in D if i!=j), name='n2truck')
    n4 = m.integer_var_dict(((i,j) for i in D for j in D if i!=j), name='n4truck')
    s1 = m.continuous_var_dict(((i,v) for i in N1 for v in V), name='s1')
    s3 = m.continuous_var_dict(((i,v) for i in N2 for v in V), name='s3')
    s5 = m.continuous_var_dict(((i,v) for i in N1 for v in V), name='s5')
    tau1 = m.continuous_var_dict(D, name='tau1')
    tau2 = m.continuous_var_dict(D, name='tau2')
    tau3 = m.continuous_var_dict(D, name='tau3')
    tau4 = m.continuous_var_dict(D, name='tau4')
    b3 = m.continuous_var_dict(((i,v) for i in N2 for v in V), name='b3')
    # Le produit est cree seulement pour les arcs entrants client (i != nd+c),
    # car x3[nd+c,nd+c,v] = 0 dans le .mod.
    prod = m.binary_var_dict(((l,i,c,v) for l in D for i in N2 for c in C
                               for v in V if i!=nd+c), name='produit3')
    def X(x,i,j,v):
        return 0 if i==j else x[i,j,v]
    def Y(y,i,j,k):
        return 0 if i==j else y[i,j,k]
    # (1) Fonction objectif.
    couts = {
      'P1':sm(a['c1'][i][j]*a['uV'][v]*var for (i,j,v),var in x1.items()),
      'P2':sm(a['cD'][i][j]*a['usetW']*var for (i,j),var in n2.items()),
      'P3':sm(a['c2'][i][j]*a['uV'][v]*var for (i,j,v),var in x3.items()),
      'P4':sm(a['cD'][i][j]*a['usetW']*var for (i,j),var in n4.items()),
      'P5':sm(a['c1'][i][j]*a['uV'][v]*var for (i,j,v),var in x5.items()),
    }
    m.minimize(sm(couts.values()))
    # P1 (2)-(11)
    for j in S: add(sm(X(x1,i,nd+j,v) for i in N1 for v in V)==1, ctname=f'c02_{j}')
    for v in V: add(sm(p[j]*X(x1,i,nd+j,v) for i in N1 for j in S)<=a['qV'][v],ctname=f'c03_{v}')
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
     for v in V: add(sm(X(x1,i,l,v) for i in N1)==sm(X(x1,l,j,v) for j in N1)) # (7)
    for i in D: add(sm(p[j]*z1[i,j] for j in S)<=a['hD'][i]) # (8)
    for i in D:
     for j in S:
      for v in V: add(-z1[i,j]+sm(X(x1,i,l,v) for l in N1)+sm(X(x1,l,nd+j,v) for l in N1)<=1) # (9)
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
    for c in C: add(sm(X(x3,i,nd+c,v) for i in N2 for v in V)==1) # (16)
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
     for v in V: add(sm(X(x3,i,l,v) for i in N2)==sm(X(x3,l,j,v) for j in N2)) # (20)
    for i in D:
     if flux_entiers:
      add(sm(demande_client[c]*z3[i,c] for c in C)<=a['hD'][i]) # (21)
     else:
      add(sm(df[c][k]*z3[i,c] for c in C for k in S)<=a['hD'][i]) # (21)
    for i in D:
     for k in S: add(p[k]*z1[i,k]+sm(Y(y2,j,i,k) for j in D)-sm(Y(y2,i,j,k) for j in D)==sm(a['df'][c][k]*z3[i,c] for c in C)) # (22)
    for i in D:
     for j in C:
      for v in V: add(-z3[i,j]+sm(X(x3,i,l,v) for l in N2)+sm(X(x3,l,nd+j,v) for l in N2)<=1) # (23)
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
    for j in S: add(sm(X(x5,i,nd+j,v) for i in N1 for v in V)==1) # (32)
    for v in V: add(sm(ret[j]*X(x5,i,nd+j,v) for i in N1 for j in S)<=a['qV'][v]) # (33)
    for i in S:
     for j in S:
      if i!=j:
       for v in V: add(s5[nd+i,v]+a['t1'][nd+i][nd+j]-M*(1-x5[nd+i,nd+j,v])<=s5[nd+j,v]) # (34)
    for i in D:
     for j in S:
      for v in V: add(a['t1'][i][nd+j]*x5[i,nd+j,v]+tau4[i]+zeta<=s5[nd+j,v]) # (35) OPL littéral
    for l in N1:
     for v in V: add(sm(X(x5,i,l,v) for i in N1)==sm(X(x5,l,j,v) for j in N1)) # (36)
    for i in D: add(sm(ret[j]*z5[i,j] for j in S)<=a['hD'][i]) # (37)
    for i in D:
     for k in S: add(sm(a['pr'][c][k]*z3[i,c] for c in C)+sm(Y(y4,l,i,k) for l in D)-sm(Y(y4,i,j,k) for j in D)==ret[k]*z5[i,k]) # (38)
    for i in D:
     for j in S:
      for v in V: add(-z5[i,j]+sm(X(x5,i,l,v) for l in N1)+sm(X(x5,l,nd+j,v) for l in N1)<=1) # (39)
    for i in D:
     for j in S:
      for v in V: add(s5[nd+j,v]+a['t1'][nd+j][i]-M*(1-x5[nd+j,i,v])<=T) # (40)
    for j in S: add(sm(z5[i,j] for i in D)<=1) # (41)
    print("\nResolution CPLEX en cours...", flush=True)
    debut = time.perf_counter()
    temps_construction = debut - debut_construction
    sol = m.solve(log_output=False)
    duree = time.perf_counter() - debut
    detail = m.solve_details
    statut = str(detail.status)
    gap = getattr(detail, 'mip_relative_gap', None)
    borne = getattr(detail, 'best_bound', None)
    # Toute solution faisable est exploitable ; le gap definit son niveau de certification.
    statut_optimal = ('optimal' in statut.lower() and 'tolerance' not in statut.lower() and 'limit' not in statut.lower())
    faisable = sol is not None
    certifie = (faisable and
                (statut_optimal or
                 (gap is not None and float(gap) <= tolerance_acceptation + 1e-9)))
    resultat = dict(statut=statut, temps_s=round(duree, 3),
                    variables=m.number_of_variables,
                    contraintes=m.number_of_constraints,
                    objectif=None, borne=borne, gap=gap,
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
        resultat['objectif'] = sol.objective_value
        resultat['couts'] = {k: sol.get_value(expr) for k, expr in couts.items()}
        resultat['affectations'] = {
            'P1': {s+1: [i+1 for i in D if sol.get_value(z1[i,s]) > .5] for s in S},
            'P3': {c+1: [i+1 for i in D if sol.get_value(z3[i,c]) > .5] for c in C},
            'P5': {s+1: [i+1 for i in D if sol.get_value(z5[i,s]) > .5] for s in S}}
        # Extraire les tournees depuis la solution de cout, avant le post-traitement.
        # Les numeros affiches correspondent aux indices OPL (a partir de 1).
        arcs_retour = []
        for nom, variables in (('P1', x1), ('P3', x3), ('P5', x5)):
            par_vehicule = {}
            for (i, j, v), arc in variables.items():
                if sol.get_value(arc) > 0.5:
                    par_vehicule.setdefault(v + 1, []).append((i + 1, j + 1))
                    if nom == 'P5' and i >= nd and j < nd:
                        arcs_retour.append((i, j, v))
            resultat['tournees'][nom] = {
                v: ordonner_arcs(arcs) for v, arcs in sorted(par_vehicule.items())
            }
        for nom, variables in (('P2', x2), ('P4', x4)):
            transferts = {}
            for (i, j, w), arc in variables.items():
                if sol.get_value(arc) > 0.5:
                    transferts.setdefault(w + 1, []).append((i + 1, j + 1))
            resultat['tournees'][nom] = transferts
        if details_interface:
            # Extraire les flux de la solution de cout. Aucun changement du MILP.
            resultat['flux_transferts'] = {
                nom: {(i + 1, j + 1): sum(sol.get_value(variables[i,j,k]) for k in S)
                      for i in D for j in D if i != j}
                for nom, variables in (('P2', y2), ('P4', y4))}
        if arcs_retour:
            # Deuxieme optimisation : conserver toutes les decisions discretes
            # et minimiser la date de fin du processus P5.
            for var in m.iter_variables():
                if var.is_discrete():
                    valeur = round(sol.get_value(var))
                    var.set_lb(valeur)
                    var.set_ub(valeur)
            fin = m.continuous_var(lb=0, name='fin_operations_P5')
            for j,i,v in arcs_retour:
                m.add_constraint(fin >= s5[j,v] + a['t1'][j][i])
            m.minimize(fin)
            debut_post = time.perf_counter()
            # La recherche principale respecte la limite imposee. Le post-traitement
            # temporel dispose d un budget court independant (max. 5 secondes).
            m.parameters.timelimit = 5.0
            sol_temps = m.solve(log_output=False)
            resultat['temps_posttraitement_s'] = round(time.perf_counter()-debut_post, 3)
            if sol_temps is not None:
                resultat['duree_operations'] = sol_temps.get_value(fin)
                if details_interface:
                    # Dates de la meme solution temporelle que la fin P5.
                    resultat['tau'] = {nom: [sol_temps.get_value(var[i]) for i in D]
                                       for nom, var in (('tau1', tau1), ('tau2', tau2),
                                                        ('tau3', tau3), ('tau4', tau4))}
                    resultat['statut_posttraitement'] = str(m.solve_details.status)

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
            print("Meilleure borne Branch-and-Cut CPLEX : {:.3f}".format(borne))
        if borne is not None:
            print("Ecart absolu avec la borne : {:.3f}".format(max(0.0, resultat["objectif"] - float(borne))))
        if gap is not None:
            print("Gap relatif final : {:.4f} %".format(100*gap))
    else:
        print("AUCUNE SOLUTION FAISABLE TROUVEE DANS LE TEMPS IMPARTI")
    print("Temps de calcul CPLEX (cout) : {:.3f} s".format(resultat['temps_s']))
    print("Statut CPLEX :", statut)
    print("=" * 56)
    m.end()
    return resultat


if __name__ == '__main__':
    print("=" * 56)
    print("       HF-VRPMFRCD — SOLVEUR CPLEX")
    print("=" * 56)
    set_id = demander_entier("Choisir le Set (1 a 5) : ", 1, 5)
    graine = demander_entier("Choisir la graine : ", minimum=0)
    limite = demander_limite()
    instance = generate_instance(set_id, seed=graine)
    instance['_unite_temps'] = 'min' if set_id <= 3 else 'h'
    print("\nINSTANCE SELECTIONNEE — Set {} / graine {}".format(set_id, graine))
    print(summary(instance, set_id))
    print("Limite de calcul : {} s".format(limite))
    print("Objectif CPLEX : prouver l optimalite (gap cible 0 %)")
    print("Tolerance d acceptation : {:.3f} %".format(100 * TOLERANCE_ACCEPTATION))
    resultat = resoudre_cplex(instance, limite_temps=limite,
                             gap_relatif=GAP_CIBLE_CPLEX,
                             tolerance_acceptation=TOLERANCE_ACCEPTATION)