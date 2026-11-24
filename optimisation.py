"""Adaptateur CL02 : une instance commune, solveurs inchangés, résultats normalisés."""
import math
import random
from time import perf_counter
from collections import Counter
from typing import Callable, Optional, Sequence

from hf_vrpmfrcd_instances import UNITES_TEMPS, generate_instance
from hf_vrpmfrcd_metaheuristics import EPS, Problem, quick_params, solve

HEURISTIQUES = ("SA", "SAVND", "ANSA")
SOLVEUR_EXACT = "Solveur HiGHS"
METHODES = (*HEURISTIQUES, SOLVEUR_EXACT)
COULEURS_METHODES = dict(SA="#F28E2B", SAVND="#59A14F", ANSA="#4E79A7", **{SOLVEUR_EXACT: "#7C3AED"})
COULEURS_PROCESSUS = dict(P1="#E15759", P2="#F28E2B", P3="#D4A017", P4="#269DAB", P5="#9467BD")
PHASES = tuple(COULEURS_PROCESSUS)


def facteur_minutes(set_id):
    return 60.0 if UNITES_TEMPS[set_id][0] == "h" else 1.0


def methodes_disponibles(set_id):
    return METHODES if set_id in (1, 2) else HEURISTIQUES


def nombre_fini(x):
    return float(x) if x is not None and math.isfinite(float(x)) else None


def verifier_heuristique(prob, details, score):
    if details['feasible'] != (details['viol_units'] < EPS and details['duree_operations'] <= prob.Tmax + EPS):
        raise RuntimeError('Evaluation finale incohérente avec Tmax.')
    if not math.isclose(details['obj'], score, rel_tol=1e-8, abs_tol=1e-6):
        raise RuntimeError('Score du solveur différent de la réévaluation.')
    if details['duree_operations'] > prob.Tmax + EPS and details['viol_time'] <= EPS:
        raise RuntimeError('Dépassement de Tmax non pénalisé.')


def executer_heuristique(prob, methode, secondes, seed=0):
    params = quick_params(prob, seed=seed, time_limit=float(secondes))
    debut = perf_counter()
    solution, score, hist, stats = solve(prob, methode.lower(), params)
    duree = perf_counter() - debut
    details = prob.evaluate(solution, detail=True)
    verifier_heuristique(prob, details, score)
    return dict(methode=methode, score=score, historique=hist, statistiques=stats,
                duree_calcul=duree, duree_totale=perf_counter()-debut, details=details,
                statut='FAISABLE' if details['feasible'] else 'NON FAISABLE',
                solution_disponible=True)


def normaliser_highs(brut, donnees, duree_totale):
    """Pas de décodage heuristique de la solution MILP ; chemins OPL exacts."""
    faisable = bool(brut['faisable'])
    cout = nombre_fini(brut.get('objectif')) if faisable else None
    duree = nombre_fini(brut.get('duree_operations')) if faisable else None
    details = dict(cost=cout, obj=cout, feasible=faisable,
                   duree_operations=duree, respecte_tmax=(duree <= donnees['tmax'] + EPS) if duree is not None else None,
                   G=brut.get('couts', {}), tau=brut.get('tau'),
                   tournees_opl=brut.get('tournees', {}),
                   viol_units=None, viol_time=None)
    for phase, num in [('P2',2), ('P4',4)]:
        groupes = details['tournees_opl'].get(phase,{})
        details[f'trucks{num}'] = dict(Counter(tuple(arc) for arcs in groupes.values() for arc in arcs))
        flux = brut.get('flux_transferts', {}).get(phase)
        details[f'flow{num}'] = None if flux is None else {k: v for k,v in flux.items() if v > EPS or k in details[f'trucks{num}']}
    return dict(methode=SOLVEUR_EXACT, score=cout, historique=[], statistiques={}, details=details,
                statut=brut.get('niveau','NO_SOLUTION'), statut_brut=brut.get('statut'),
                solution_disponible=faisable, duree_calcul=nombre_fini(brut.get('temps_s')),
                duree_totale=duree_totale, temps_construction=brut.get('temps_construction_s'),
                temps_posttraitement=brut.get('temps_posttraitement_s'),
                statut_posttraitement=brut.get('statut_posttraitement'),
                borne=nombre_fini(brut.get('borne')), gap=nombre_fini(brut.get('gap')),
                variables=brut.get('variables'), contraintes=brut.get('contraintes'))


def executer_comparaison(set_id:int, graine:int, methodes:Sequence[str]=HEURISTIQUES,
                         secondes:float=20.0, progression:Optional[Callable]=None,
                         secondes_highs:float=120.0):
    if set_id not in UNITES_TEMPS or graine < 0:
        raise ValueError('Set ou graine invalide.')
    methodes = tuple(dict.fromkeys(methodes))
    if not methodes or any(m not in methodes_disponibles(set_id) for m in methodes):
        raise ValueError('Méthodes invalides : HiGHS est réservé aux Sets 1 et 2.')
    for budget, actif in [(secondes, any(m in HEURISTIQUES for m in methodes)), (secondes_highs, SOLVEUR_EXACT in methodes)]:
        if actif and not (0 < float(budget) < math.inf):
            raise ValueError('Durée maximale strictement positive et finie requise.')
    donnees = generate_instance(set_id, seed=graine)
    prob = Problem(**donnees) if any(m in HEURISTIQUES for m in methodes) else None
    resultats, erreurs = [], {}
    # Exécution séquentielle dans l'ordre console, HiGHS en dernier.
    ordre = [m for m in METHODES if m in methodes]
    for numero, methode in enumerate(ordre, 1):
        try:
            if methode == SOLVEUR_EXACT:
                from hf_vrpmfrcd_highs import resoudre_highs
                debut = perf_counter()
                brut = resoudre_highs(dict(donnees, _unite_temps=UNITES_TEMPS[set_id][0]),
                                      limite_temps=float(secondes_highs), details_interface=True)
                r = normaliser_highs(brut, donnees, perf_counter()-debut)
            else:
                r = executer_heuristique(prob, methode, secondes)
            resultats.append(r)
        except Exception as exc:
            erreurs[methode] = f'{type(exc).__name__}: {exc}'
        if progression:
            progression(numero, len(ordre), methode)
    # Une fois, hors chronométrage et après les solveurs pour ne pas réchauffer leurs caches.
    initial = None
    if any(r['methode'] in HEURISTIQUES for r in resultats):
        try:
            a0, _ = prob.initial_solution(random.Random(0))
            d0 = prob.evaluate(a0, detail=True)
            initial = {k: d0[k] for k in ('obj', 'cost', 'feasible')}
        except Exception as exc:
            erreurs['Référence initiale'] = f'{type(exc).__name__}: {exc}'
    return dict(set_id=set_id, graine=graine, secondes=secondes, secondes_highs=secondes_highs,
                unite_python=UNITES_TEMPS[set_id][0], donnees=donnees, resultats=resultats,
                erreurs=erreurs, initial=initial)


def classement(resultats):
    valides = [r for r in resultats if r['details']['feasible'] and r['details']['cost'] is not None]
    if valides:
        return min(valides, key=lambda r:r['details']['cost'])
    penalises = [r for r in resultats if r['methode'] in HEURISTIQUES and r['details']['obj'] is not None]
    return min(penalises, key=lambda r:r['details']['obj']) if penalises else None


def _arrondi(x, n=3):
    return round(x,n) if x is not None else None


def tableau_comparaison(execution):
    f = facteur_minutes(execution['set_id'])
    lignes=[]
    for r in execution['resultats']:
        d=r['details']; t=d['duree_operations']; gap=r.get('gap')
        lignes.append({'Méthode':r['methode'], 'Coût transport':_arrondi(d['cost']),
                       'Objectif pénalisé':_arrondi(d['obj']) if r['methode']!=SOLVEUR_EXACT else None,
                       'Faisable': 'Oui' if d['feasible'] else ('Non' if r['solution_disponible'] else 'Non trouvée'),
                       'Statut':r['statut'], 'Calcul (s)':_arrondi(r['duree_calcul']),
                       'Total (s)':_arrondi(r['duree_totale']), 'Opérations (min)':_arrondi(t*f) if t is not None else None,
                       'Tmax (min)':execution['donnees']['tmax']*f,
                       'Itérations':r['statistiques'].get('iterations'), 'Borne HiGHS':_arrondi(r.get('borne')),
                       'Gap HiGHS (%)':_arrondi(gap*100,4) if gap is not None else None})
    return lignes


def libelle_noeud(n, nD, phase):
    return f'D{n}' if n<=nD else f'{"C" if phase=="P3" else "S"}{n-nD}'


def itineraires(execution, resultat):
    """Chemins exacts, coûts et temps de trajet (hors attente), communs aux vues."""
    a=execution['donnees']; nD=a['nD']; d=resultat['details']; f=facteur_minutes(execution['set_id'])
    for phase in ('P1','P3','P5'):
        if resultat['methode']==SOLVEUR_EXACT:
            c,t=(a['c2'],a['t2']) if phase=='P3' else (a['c1'],a['t1'])
            routes=((v,list(path),sum(c[i-1][j-1] for i,j in zip(path,path[1:]))*a['uV'][v-1],
                     sum(t[i-1][j-1] for i,j in zip(path,path[1:])))
                    for v, paths in d.get('tournees_opl',{}).get(phase,{}).items() for path in paths)
        else:
            routes=((v,[dock,*nodes,dock],c,t) for dock,nodes,v,c,t in d.get(phase,[]))
        for idx,(v,path,c,t) in enumerate(routes,1):
            yield dict(phase=phase,index=idx,veh=v,path=path, labels=[libelle_noeud(n,nD,phase) for n in path],cost=c,minutes=t*f)


def tableau_tournees(execution,resultat):
    return [{'Processus':r['phase'],'Tournée':r['index'],'Véhicule':f"V{r['veh']}",
             'Parcours':' → '.join(r['labels']), 'Coût':_arrondi(r['cost']), 'Trajet (min)':_arrondi(r['minutes'])}
            for r in itineraires(execution,resultat)]


def tableau_transferts(execution,resultat,phase):
    if phase not in ('P2','P4'): raise ValueError('Phase de transfert invalide.')
    d=resultat['details']; a=execution['donnees']; num=phase[-1]; f=facteur_minutes(execution['set_id'])
    trucks=d.get('trucks'+num,{}); flux=d.get('flow'+num)
    lignes=[]
    for i,j in sorted(set(trucks)|set(flux or {})):
        n=trucks.get((i,j),0); q=flux.get((i,j),0) if flux is not None else None
        if not n and not q: continue
        vehicules = None
        if resultat['methode'] == SOLVEUR_EXACT:
            vehicules = ', '.join(f'W{w}' for w, arcs in d['tournees_opl'].get(phase, {}).items() if (i,j) in map(tuple,arcs))
        lignes.append({'Origine':f'D{i}','Destination':f'D{j}','Quantité transférée':_arrondi(q),'Camions':n, **({'Véhicules':vehicules} if vehicules is not None else {}),
                       'Durée liaison (min)':_arrondi(a['tD'][i-1][j-1]*f),'Coût liaison':_arrondi(a['cD'][i-1][j-1]*a['usetW']*n)})
    return lignes


def graphe_flux_simplifie(execution, resultat=None):
    a=execution['donnees']; nD,nS,nC=a['nD'],a['nS'],a['nC']
    if nD+nS+nC>20: return {'actif':False,'raison':'Instance trop grande pour un schéma lisible.'}
    return {'actif':True,'positions':{**{f'S{s}':(0,-s) for s in range(1,nS+1)},
            **{f'D{d}':(1.5,-d*2) for d in range(1,nD+1)},**{f'C{c}':(3,-c) for c in range(1,nC+1)}}}