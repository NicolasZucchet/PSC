# PSC

L'objectif de ce projet est de <b>reconnaître les activités humaines</b> grâce au données récupérées par une smartwatch.

Les résultats obtenus sont les suivants : en utilisant un CNN, on arrive à <b>95%</b> de bonnes prédictions pour 6 activités et <b>80%</b> pour 11.

Le rapport explicatif du projet est dans le dossier 'Rapports/Rapport final', la définition des différents modèles dans 'Code/Prediction tools/solutions'.


## Structure du repository

Les différents rapports écrits au long du projet sont disponibles dans le dossier "Rapports". Le rapport final est le dernier écrit et comprend une présentation du projet dans son ensemble (sans trop rentrer dans les détails techniques).

Le dossier <b>Code</b> contient tous les briques du code, allant du traitement des données à l'évaluation de la qualité des prédictions :
- data : les données brutes (récupérées sur internet), le code pour structurer les données et l'output
- datareader : récupération des données
- solutions : <b>différents modèles</b> (CNN, SGDC, SVC) utilisés pour faire les comparaisons
Le fichier "solutions/main.py" sert de base.

Le dossier <b>Rapports/Rendus</b> se trouvent les différents rapports rendus durant le projet :
- La page publique contient un résumé du travail et du résultat.
- Le rapport final est le rapport rendu à la fin du travail, qui présente de manière plus détaillée les résultats.
