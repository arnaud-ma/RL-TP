#import "@preview/lovelace:0.3.0": *
#import "@preview/subpar:0.2.2"

#set document(
  author: "Arnaud Mabit",
  title: "TP RL",
)

#set text(font: "New Computer Modern", lang: "fr")
#set page(numbering: "1 / 1")
#set heading(numbering: "1.1")



#let argmax = $op("argmax", limits: #true)$
#let argmin = $op("argmin", limits: #true)$

#title()

#outline()

Le code est disponible ici: https://github.com/arnaud-ma/RL-TP

= Environnement de code / configuration

L'utilisation de `uv` simplifie les choses:
- s'installe rapidement en une commande (https://docs.astral.sh/uv/#installation)
- `uv init --package --python 3.13` crée la structure du projet
- `uv add <package>` ajoute les dépendances. Le fichier `pyproject.toml` qui contient toutes les métadonnées du projet (y compris les dépendances) est automatiquement mis à jour.
- `uv run <script>` pour lancer un script python. Cette dernière commande s'occupe, quelle que soit la machine, de automatiquement:
  - installer la bonne version de Python (si besoin)
  - créer un environnement virtuel (si besoin)
  - activer l'environnement virtuel (si besoin)
  - installer les dépendances (si besoin)
  - lancer le script

Cela permet d'avoir la sécurité d'avoir un environnement par projet, tout en n'ayant pas besoin de le gérer soi-même du tout.
Il y a une alternative `pixi` qui rajoute par défaut une compatibilité avec `conda`, mais dans notre cas `uv` suffit puisqu'on utilise que des paquets venant de Python.

Pour charger les configurations, j'utilise  `pydantic`.
```python
class ConfigModel(BaseModel):
    env: str

    seed: PositiveInt
    device: str | int
    render_mode: Literal["human", "terminal", "both"] = "both"
    deterministic_config: bool = False

    max_length_train: PositiveInt
    max_length_test: PositiveInt

    epsilon_start: ZeroToOne
    epsilon_min: ZeroToOne
    epsilon_decay: ZeroToOne

		...
```
Cela permet de facilement valider le configurations, directement avoir une erreur si une valeur est incorrecte, documenter les paramètres, et avoir un vrai objet python à la place d'un dictionnaire (plus pratique à utiliser, auto-complétion, etc.).

= Algorithme DQN

== Algorithme

#pseudocode-list[
  + *Données* :
    + Un environnement de Markov avec états S, actions A, récompenses R
    + Un taux d'actualisation $gamma in [0, 1]$
    + Un taux d'exploration initial $epsilon in [0, 1]$
    + Un taux d'exploration minimum $epsilon_min in [0, epsilon]$
    + Un facteur de décroissance de l'exploration $epsilon_"decay" in (0, 1)$
    + Une capacité de mémoire $N$
    + Une taille de mini-lot $M$
    + Un nombre maximum d'étapes par épisode $T_max$
    + Une fonction de perte $l$ (ex: erreur quadratique)
  + un optimiseur de descente de gradient $O$ (ex: Adam, RMSProp)
    + Une fréquence d'optimisation $F$
    + Une fréquence de mise à jour du réseau cible $C$

  + *Résultat* : Les poids $theta$ du réseau de Q-valeurs appris.

  + *Initialisation* :
    + la mémoire de replay $D$ à capacité $N$
    + le réseau de Q-valeurs $Q_theta$ avec des poids aléatoires $theta$
    + le réseau cible $Q_(theta^(-))$ avec des poids $theta^(-) = theta$

  + *Boucle principale* :
    + *pour* chaque épisode *faire*
      + Initialise l'état initial $s_0$
      + *pour* $t = 0$ à $T_max$ et *tant que* $s_t$ n'est pas terminal *faire*
        + $a_t <- cases(
            a tilde cal(U)(A) "avec probabilité" epsilon,
            argmax_a Q_(θ)(s_t, a) "sinon"
          )$
        + $epsilon <- max(epsilon_min, epsilon * epsilon_"decay")$
        + exécute l'action $a_t$, observe la récompense $r_t$ et le nouvel état $s_(t+1)$
        + stocke la transition $(s_t, a_t, r_t, s_(t+1))$ dans la mémoire de replay $D$
        + *si* la taille de $D >= M$ *et* $t mod F = 0$ *alors*
          + échantillonne un mini-lot de $M$ transitions $(s_j, a_j, r_j, s_(j+1))$ de $D$ uniformément, sans remise
          + *pour* chaque transition $j$ du mini-lot *faire*
            + $V_j <- cases(
                0 "si" s_(j+1) "est terminal",
                max_(a') Q_(theta^(-))(s_(j+1), a') "sinon"
              )$
            + $y_j <- r_j + gamma V_j$
            + On note $l_j: theta |-> l(Q_(theta)(s_j, a_j), y_j)$
          + *fin pour*
          + On note $L: theta |-> 1/M sum_j l_(j)(theta)$
          + $theta <- "O.step"(theta, nabla_theta L(theta))$ (descente de gradient)
          + *si* $t mod C = 0$
            + $theta^(-) <- theta$ (copie des poids du réseau de Q-valeurs vers le réseau cible)
          + *fin si*
      + *fin pour* (action)
    + *fin pour* (épisode)
]

Ensuite pour évaluer la politique apprise:
#pseudocode-list[
  + *Données* :
    + Le réseau de Q-valeurs appris $Q_(theta)$
    + Un nombre maximum d'étapes $T_"max"$

  + *Résultat* : La récompense $R$.

  + Initialise l'état initial $s_0$
  + $R <- 0$
  + *pour* $t = 0$ à $T_max$ et *tant que* $s_t$ n'est pas terminal *faire*
    + $a_t <- argmax_a Q_(θ)(s_t, a)$
    + exécute l'action $a_t$, observe la récompense $r_t$ et le nouvel état $s_(t+1)$
    + $R <- R + r_t$
  + *fin pour*
]

- On peut faire une version sans réseau cible: il suffit de remplacer tous les $theta^(-)$ par $theta$ dans l'algorithme.

- On peut faire une version sans mémoire de replay: il suffit de remplacer l'échantillonnage de mini-lots par l'utilisation de la transition courante. Le mini-lot est alors de taille 1.

- Une version avec une mise à jour douce du réseau cible peut être faite en remplaçant la copie des poids par une interpolation:
  $
    theta^(-) <- tau theta + (1 - tau) theta^(-)
  $
  avec $tau in (0, 1]$ un petit paramètre (ex: 0.001), qui remplace la fréquence de mise à jour $C$.

- Une version avec prioritized experience replay peut être faite en modifiant la manière d'échantillonner les transitions dans la mémoire de replay, et en ajoutant des poids d'importance dans le calcul de la perte:
  + Lors de l'échantillonnage des transitions, utiliser une probabilité proportionnelle à l'erreur TD ($l(Q_(theta)(s_j, a_j), y_j)$).
  + Calculer des poids d'importance pour chaque transition échantillonnée.
  + Utiliser ces poids dans le calcul de la perte:
  $
    L(theta) = 1/M sum_j w_j l(Q_(theta)(s_j, a_j), y_j)
  $

- Une version Double DQN peut être faite en modifiant le calcul de $V_j$:
  $
    V_j <- cases(
      0 "si" s_(j+1) "est terminal",
      Q_(theta^(-))(s_(j+1), argmax_(a') Q_(theta)(s_(j+1), a')) "sinon"
    )
  $


== CartPole-v1

=== Sans réseau cible ni prioritized experience replay

#subpar.grid(
  figure(image("assets/cartpole/nothing/epsilon.svg"), caption: [
    $epsilon$ en fonction des épisodes d'entraînement.
  ]),
  figure(image("assets/cartpole/nothing/q_value.svg"), caption: [
    Q-valeur moyenne des états visités en fonction des épisodes d'entraînement.
  ]),

  figure(image("assets/cartpole/nothing/reward_train.svg"), caption: [
    Récompense obtenue lors des épisodes d'entraînement
  ]),
  figure(image("assets/cartpole/nothing/reward_test.svg"), caption: [
    Récompense moyenne obtenue de 100 épisodes de test tous les 50 épisodes d'entraînement.
  ]),

  columns: (1fr, 1fr),
  caption: [Résultats de l'entraînement du DQN sans réseau cible ni prioritized experience replay sur l'environnement CartPole-v1.],
  label: <nothing>,
)

Je ne remarque pas de grandes différences si on rajoute un réseau cible. L'apprentissage reste peu stable, et la variance entre différentes trajectoires d'entraînement est importante. Par exemple dans l'exemple ci-dessus, on a eu de la chance d'atteindre une bonne politique assez rapidement.

=== Prioritized experience replay

```py
ConfigDQN(
    env='CartPole-v1',
    seed=6,
    max_length_train=500,
    max_length_test=500,
    epsilon_start=0.9,
    epsilon_min=0.0001,
    epsilon_decay=0.9999,
    gamma=0.99,
    learning_rate=0.0003,
    batch_size=100,
    optimizer=<class 'torch.optim.adam.Adam'>,
    value_loss=<class 'torch.nn.modules.loss.SmoothL1Loss'>,
    train_freq=1,
    use_target_network=False,
    clip_grad_norm=10,
    hidden_layers=[200],
    hidden_layers_activation=<class 'torch.nn.modules.activation.Tanh'>,
    final_layer_activation=<class 'torch.nn.modules.linear.Identity'>,
    replay_memory_capacity=10000,
    replay_memory_min_size=500,
    prioritized_replay=True,
    nb_episodes=2000,
    freq_test=50,
    nb_tests=100,
)
```

#subpar.grid(
  figure(image("assets/cartpole/graph2/epsilon.svg"), caption: [
    $epsilon$ en fonction des épisodes d'entraînement.
  ]),
  figure(image("assets/cartpole/graph2/q_value.svg"), caption: [
    Q-valeur moyenne des états visités en fonction des épisodes d'entraînement.
  ]),

  figure(image("assets/cartpole/graph2/reward_train.svg"), caption: [
    Récompense obtenue lors des épisodes d'entraînement
  ]),
  figure(image("assets/cartpole/graph2/reward_test.svg"), caption: [
    Récompense moyenne obtenue de 100 épisodes de test tous les 50 épisodes d'entraînement.
  ]),

  columns: (1fr, 1fr),
  caption: [Résultats de l'entraînement du DQN sur l'environnement CartPole-v1.],
  label: <full>,
)

La Q-valeur se stabilise vers 100. Ce qui confirme la théorie:

$
  Q^*(s, a) & = EE[R_0 | s_0 = s, a_0 = a] \
            & = EE[sum_(i=0)^(T-1) gamma^i r_i | s_0 = s, a_0 = a] \
            & = sum_(i=0)^(T-1) gamma^i EE[r_i | s_0 = s, a_0 = a] \
$
Et si notre politique est bonne, quel que soit l'état $s$ et l'action $a$, on a une récompense proche de $1$ en moyenne à chaque étape, donc:
$
  Q^*(s, a) & approx sum_(i=0)^(T-1) gamma^i = (1 - gamma^T) / (1 - gamma) approx 1 / (1 - gamma) \
$

On remarque que la Q-valeur approximée par le réseau est souvent plus grande que cette valeur théorique. On peut faire un lien avec la loss lors de l'apprentissage: si la Q-valeur est sur-estimée, alors les cibles $y_j$ seront aussi sur-estimées, et donc la loss sera plus grande.



#figure(
  image("assets/cartpole/graph2/loss.svg"),
  caption: [Loss de l'entraînement du DQN sur l'environnement CartPole-v1.],
)

Je n'ai pas remarqué de grande différence avec ou sans prioritized experience replay sur cet environnement.

De même, on pourrait penser que le Double DQN permettrait de réduire cette sur-estimation, mais dans les faits je n'ai pas remarqué de différence.

=== Réseau cible et prioritized experience replay

// L'utilisation d'un réseau cible permet de stabiliser l'apprentissage en réduisant la corrélation entre les cibles et les prédictions. Cela évite que les mises à jour successives ne se propagent trop rapidement à travers le réseau, ce qui peut entraîner des oscillations ou une divergence de l'apprentissage.

#subpar.grid(
  figure(image("assets/cartpole/full/epsilon.svg"), caption: [
    $epsilon$ en fonction des épisodes d'entraînement.
  ]),
  figure(image("assets/cartpole/full/q_value.svg"), caption: [
    Q-valeur moyenne des états visités en fonction des épisodes d'entraînement.
  ]),

  figure(image("assets/cartpole/full/reward_train.svg"), caption: [
    Récompense obtenue lors des épisodes d'entraînement
  ]),
  figure(image("assets/cartpole/full/reward_test.svg"), caption: [
    Récompense moyenne obtenue de 100 épisodes de test tous les 50 épisodes d'entraînement.
  ]),

  columns: (1fr, 1fr),
  caption: [Résultats de l'entraînement du DQN avec réseau cible et prioritized experience replay sur l'environnement CartPole-v1.],
  label: <full>,
)

L'apprentissage est bien plus sample-efficient. Seulement 300 épisodes contre 16000 de celle juste au-dessus pour atteindre une bonne politique.

== LunarLander-v3

L'environnement Lunar étant plus complexe, on change:
- la taille du replay buffer à $100 000$ (si utilisé).
- la taille du réseau à deux couches (128, 128)
- la fonction d'activation des couches cachées à ReLU pour accélérer l'apprentissage
- on change l'optimiseur à AdamW, une variante d'Adam.
- mise à jour du réseau cible tous les 1000 événements d'entraînement au lieu de tous les 100.
- Diminution de la taille du batch à 64, pour gagner du temps de calcul.

Et bien sûr, une adaptation de $epsilon$ pour que l'exploration dure plus longtemps.

```py
ConfigDQN(
    env='LunarLander-v3',
    seed=6,
    nb_episodes=10000,
    freq_test=100,
    nb_tests=2,
    max_length_train=1000,
    max_length_test=1000,
    epsilon_start=0.9,
    epsilon_min=0.01,
    epsilon_decay=0.999998,
    gamma=0.99,
    learning_rate=0.0003,
    batch_size=64,
    optimizer=<class 'torch.optim.adamw.AdamW'>,
    value_loss=<class 'torch.nn.modules.loss.SmoothL1Loss'>,
    train_freq=4,
    use_target_network=False,
    target_update_freq=1000,
    target_soft_update=0.0,
    clip_grad_norm=0,
    hidden_layers=[128, 128],
    hidden_layers_activation=<class 'torch.nn.modules.activation.ReLU'>,
    final_layer_activation=<class 'torch.nn.modules.linear.Identity'>,
    dropout=0.0,
    replay_memory_capacity=0,
    replay_memory_min_size=1000,
    prioritized_replay=False,
)
```

=== Sans réseau cible ni prioritized experience replay

#subpar.grid(
  figure(image("assets/lunar/nothing/epsilon.svg"), caption: [
    $epsilon$ en fonction des épisodes d'entraînement.
  ]),
  figure(image("assets/lunar/nothing/q_value.svg"), caption: [
    Q-valeur moyenne des états visités en fonction des épisodes d'entraînement.
  ]),

  figure(image("assets/lunar/nothing/reward_train.svg"), caption: [
    Récompense obtenue lors des épisodes d'entraînement
  ]),
  figure(image("assets/lunar/nothing/reward_test.svg"), caption: [
    Récompense moyenne obtenue de 100 épisodes de test tous les 50 épisodes d'entraînement.
  ]),

  columns: (1fr, 1fr),
  caption: [Résultats de l'entraînement du DQN sans réseau cible ni prioritized experience replay sur l'environnement LunarLander-v3.],
  label: <nothing>,
)

== Avec réseau cible et prioritized experience replay

#subpar.grid(
  figure(image("assets/lunar/full/epsilon.svg"), caption: [
    $epsilon$ en fonction des épisodes d'entraînement.
  ]),
  figure(image("assets/lunar/full/q_value.svg"), caption: [
    Q-valeur moyenne des états visités en fonction des épisodes d'entraînement.
  ]),

  figure(image("assets/lunar/full/reward_train.svg"), caption: [
    Récompense obtenue lors des épisodes d'entraînement
  ]),
  figure(image("assets/lunar/full/reward_test.svg"), caption: [
    Récompense moyenne obtenue de 100 épisodes de test tous les 50 épisodes d'entraînement.
  ]),

  columns: (1fr, 1fr),
  caption: [Résultats de l'entraînement du DQN avec réseau cible et prioritized experience replay sur l'environnement LunarLander-v3.],
  label: <full>,
)

Même avec le réseau cible et le prioritized experience replay, l'apprentissage la Q-valeur ne semble pas stable (même si bien meilleur que sans).

Testons avec le Double DQN.

== Double DQN

#subpar.grid(
  figure(image("assets/lunar/double/epsilon.svg"), caption: [
    $epsilon$ en fonction des épisodes d'entraînement.
  ]),
  figure(image("assets/lunar/double/q_value.svg"), caption: [
    Q-valeur moyenne des états visités en fonction des épisodes d'entraînement.
  ]),

  figure(image("assets/lunar/double/reward_train.svg"), caption: [
    Récompense obtenue lors des épisodes d'entraînement
  ]),
  figure(image("assets/lunar/double/reward_test.svg"), caption: [
    Récompense moyenne obtenue de 100 épisodes de test tous les 50 épisodes d'entraînement.
  ]),

  columns: (1fr, 1fr),
  caption: [Résultats de l'entraînement du Double DQN avec réseau cible et prioritized experience replay sur l'environnement LunarLander-v3.],
  label: <double>,
)

Le Double DQN permet de stabiliser l'apprentissage de la Q-valeur, et donc de la politique. Il faudrait avoir un temps d'entraînement plus long.
