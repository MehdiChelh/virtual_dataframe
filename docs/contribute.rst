.. index:: clone

Contribute
==========

Récupérez les sources

.. code-block:: bash

    $ git clone |giturl| virtual_dataframe

puis:

.. code-block:: bash

    $ cd virtual_dataframe
    $ make configure
    $ conda activate virtual_dataframe
    $ make docs

Principes d'organisation du projet
==================================
* Les traitements de data sciences utilisent une approche à l'aide de fichiers, indiqués
  dans les paramètres de la ligne de commande. Cela permet:

  - d'executer plusieurs règles en parallèle dans le ``Makefile`` utilisant le même script (``make -j 4 ...``)
  - d'ajuster les noms des répertoires aux environements d'exécution (local, Docker, SageMaker, etc)
  - d'avoir des tests unitaires isolant facilement chaque étape du workflow

* Les meta-paramètres des algorithmes doivent également pouvoir être valorisés via des paramètres de la ligne
  de commande. Cela permet d'utiliser des outils d'optimisations des meta-paramètres.

* Les metrics doivent être sauvegardées dans un fichier ``.json`` et pushé sur GIT. Cela permet
  d'avoir une trace des différents scénarios dans différents tags ou branches, pour les comparer.

* Un ``make validate`` est exécuter automatiquement avant un ``git push`` sur la branche ``master``.
  Il ne doit pas générer d'erreurs. Cela permet d'avoir un build incassable, en ne publiant
  que des commits validés.
  Il est possible de forcer tout de même le push avec ``FORCE=y git push``.
  Cela permet d'avoir l'équivalent d'une CI/CD en local. Bien entendu, cela peut etre supprimé
  si une plateforme d'intégration est disponible.

* La version du projet et déduite du tag courant GIT

* Pour les fichiers de données, le modèle propose
  d'utiliser :

  - git classic si les fichiers ne sont pas trop gros. L'utilisation de `daff <https://paulfitz.github.io/daff/>`_ permet alors
    de gérer plus facilement les évolutions des fichiers CSV. Les notebooks sont également
    nettoyés avant publication dans le repo GIT.
  - `git lfs <https://git-lfs.github.com/>`_ pour les gros fichiers,
    mais il faut avoir un serveur LFS disponible. C'est le cas pour Gitlab ou Github.

* Le modèle propose d'utiliser

  - un module pour ``virtual_dataframe``
  - des scripts Python dans ``scripts/``
  - des scripts Jupyter dans ``notebooks/``

* Le code du projet doit être présent dans le package ``virtual_dataframe``.
  Ils doivent être exécuté depuis le ``home`` du projet. Pour cela, une astuce consiste
  à forcer le ``home`` dans les templates Python de PyCharm. Ainsi, un ``run`` depuis un source
  fonctionne directement, sans manipulation.

* Pour gérer des scripts python autonome ou des notebooks,
  il est proposé d'utiliser des sous-répertoires correspondant
  au différentes phases de projet (e.g. ``scripts/phase1/`` ou ``notebooks/phase1/``).
  Dans ces répertoires, les scripts seront exécutés dans
  l'ordre lexicographique pour validation.

* Il est possible de convertir les notebooks en scripts, via ``make nb-convert``
* Le `typing <https://realpython.com/python-type-checking/>`_ est recommandé, afin d'améliorer la qualité du code
  et sa documentation. Vous pouvez vérifier cela avec ``make typing``, ou ajouter automatiquement une partie du typing
  à votre code avec ``make add-typing``.
* La documentation est générée en ``html`` et ``latexpdf`` dans le répertoire ``build/``. Tous les autres format
  de Sphinx sont possible, via un ``make build/epub`` par exemple.
* La distribution du package est conforme aux usages sous Python, avec un package dédié avec les sources
  et un package WHL.

Truc et astuces
===============
Quelques astuces disponibles dans le projet.

Les test
--------
Les tests sont divisés en deux groupes : ``unit-test`` et ``functional-test``.
Il est possible d'exécuter l'un des groups à la fois (``make (unit|functional)-test``) ou
l'ensemble (``make test``).

Les tests sont parallélisés lors de leurs executions. Cela permet de bénéficier des architectures
avec plusieurs coeurs CPU. Pour désactiver temporairement cette fonctionnalité, il suffit
d'indiquer un nombre de coeur réduit à utiliser. Par exemple : ``NPROC=1 make test``

Vérifier le build
-----------------
Pour vérifier que le ``Makefile`` est correct, vous pouvez vider l'environement conda avec ``make clean-venv``
puis lancer votre règle. Elle doit fonctionner directement et doit même pouvoir être exécuté deux fois
de suite, sans rejouer le traitement deux fois. Par exemple :


.. code-block:: bash

    $ make validate
    $ make validate

Déverminer le Makefile
----------------------
Il est possible de connaitre la valeur calculée d'une variable dans le ``Makefile``. Pour cela,
utilisez ``make dump-MA_VARIABLE``.

Pour comprendre les règles de dépendances justifiant un build, utilisez ``make --debug -n``.

Convertir un notebook
---------------------
Il est possible de convertir un notebook en script, puis de lui ajouter un typage.

.. code-block:: bash

    make nb-convert add-typing

Gestion des règles ne produisant pas de fichiers
------------------------------------------------
Le code génère des fichiers  ``.make-<rule>`` pour les règles ne produisant pas
de fichier, comme ``test`` ou ``validate`` par exemple. Vous pouvez ajoutez ces
fichiers à GIT pour mémoriser la date de la dernière exécution. Ainsi, les
autres développeurs n'ont pas besoin de les ré-executer si ce n'est pas nécessaire.


Recommandations
===============
* Utilisez un CHANGELOG basé sur `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
* Utilisez un format de version conforme à `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.
* Utiliser une approche `Develop/master branch <https://nvie.com/posts/a-successful-git-branching-model/>`_.
* Faite toujours un ``make validate`` avant de commiter le code
