# FirstAi
IA de prédiction de taille de pétales de roses

Ce script implémente un petit réseau de neurones (MLP) from scratch avec NumPy pour faire de la classification binaire sur des fleurs. Après avoir normalisé les données, on lance l'entraînement sur 1000 époques où tout se joue dans la rétropropagation et la descente de gradient : à chaque itération, on calcule l'erreur en sortie qu'on propage en arrière (backprop) via la chain rule pour déterminer la responsabilité de chaque neurone, puis on utilise ces gradients pondérés par la dérivée de la sigmoïde pour updater les matrices de poids $W1$ et $W2$, ce qui permet de minimiser l'erreur globale et de faire converger le modèle pour qu'il prédise correctement la couleur de la fleur test.

Le réseau est très simple: deux neurones en entrée, trois dans la couche cachée et deux en sortie.
