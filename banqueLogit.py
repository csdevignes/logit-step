'''
Ce programme calcule un modèle logistique de prévision des données DECOUVERT à partir des autres données.
Il évalue ensuite l'efficacité de ce modèle.
Cours: Mathématiques pour le machine learning.
Créer un modèle de prévision logistique à partir de ces données qui permet de prévoir découvert.
Evaluer et sélectionner les variables essentielles et donner la qualité du modèle choisi.
Faire un compte rendu. Pour le 1er février.
'''
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tabulate import tabulate

#Importation des données à partir du fichier banque, séparateur: espace
banque = pd.read_table('banque.txt', sep=" ")
#Définition de la variable de réponse Y et des variables explicatives X
Y = banque['DECOUVERT']
X = banque[['AGE', 'EMP', 'ADRE', 'REVENU', 'CREDIT', 'DEBTINCOME', 'CREDDEBT', 'OTHDEBT']]

# #On définit les fonctions de calcul du modèle de régression logistique avec les packages statsmodels et sklearn:
def calculSM(X, Y) :
     """
     Modélisation des données avec statsmodels: permet d'obtenir le AIC et les p-values
     :param X: données explicatives, tableau pandas
     :param Y: donnée à expliquer, tableau pandas
     :return: l'objet statsmodels resultatSM qui contient les résultats de la régression logistique
     """
     #On rajoute la constante a X pour avoir le beta 0, la colonne est automatiquement appelée const
     Xconst = sm.add_constant(X)
     #On crée le modèle et utilise la méthode fit pour trouver l'estimation des coefficients
     modele_logitSM = sm.Logit(Y,Xconst)
     resultatSM = modele_logitSM.fit(disp=0) #On met disp=0 pour éviter qu'il affiche des messages à chaque calcul
     return resultatSM

def calculSK(X, Y, cv=10) :
     """
     Modélisation des données avec sk-learn: permet d'obtenir l'accuracy
     :param X: données explicatives, tableau pandas
     :param Y: donnée à expliquer, tableau pandas
     :param cv: nombre de paquets de données, par défaut 5
     :return: scores la liste des accuracy obtenues avec cv=5 combinaisons de paquets
     """
     #Il faut ajouter un solver qui précise la méthode utilisée pour trouver les coefficients.
     modele_logitSK = LogisticRegression(solver='newton-cg')
     #cv = 5 est le nombre de paquets. Retourne 5 valeurs dont il faut faire la moyenne pour obtenir l'accuracy
     scores = cross_val_score(modele_logitSK, X, Y, cv=cv, scoring='accuracy')
     return scores

colToDelete = ""
while colToDelete != "STOP":
    #Calcul du modèle actuel
    resultatFull = calculSM(X, Y)
    scoresFull = calculSK(X, Y)
    ##Affichage des résultats
    resDF = pd.DataFrame(index=resultatFull.params.index)
    resDF["Coef"] = resultatFull.params
    resDF["Pvalues"] = resultatFull.pvalues
    #print(resultatFull.summary())
    print(f'Modèle actuel: Accuracy : {scoresFull.mean()}, AIC : {resultatFull.aic}')
    #Recalcul du modèle avec une variable en moins, et affichage de l'Accuracy et l'AIC
    #Boucle dans X par les colonnes
    for col in X.columns:
        newX = X.drop(columns=col) #On enlève la colonne en cours et on recalcule les modèles
        scoresNewX = calculSK(newX, Y)
        resultatNewX = calculSM(newX, Y)
        resDF.loc[col, "AIC"] = resultatNewX.aic
        resDF.loc[col, "Accur"] = scoresNewX.mean()
    #On affiche le résultat
    pd.set_option("display.precision", 4)
    print(tabulate(resDF, headers="keys", tablefmt="psql"))

    colToDelete = input("Quelle variable voulez vous supprimer ? (Pour arrêter entrez STOP) ")
    if colToDelete in X:
        X = X.drop(columns=colToDelete)

resultatFinal = calculSM(X, Y)
scoresFinal = calculSK(X, Y, cv=Y.value_counts().min())
print(f'Le modèle choisi est: {X.columns}')
print(f'sa performance est {scoresFinal.mean()} avec une standard deviation de {scoresFinal.std()}')
confusion = resultatFinal.pred_table() #Affiche la table de confusion.
print(confusion[0,0])
# True positive rate or sensitivity = TP/(TP+FN)
tpr = confusion[1,1]/(confusion[1,1]+confusion[1,0])
# True negative rate or specificity = TN/(FP+TN)
tnr = confusion[0,0]/(confusion[0,1]+confusion[0,0])
# False negative rate
fnr = 1 - tpr
# False positive rate
fpr = 1 - tnr
print(f"TPR = {tpr}, TNR = {tnr}, FNR = {fnr}, FPR = {fpr}")


