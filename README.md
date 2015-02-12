afa_cup_learning
================

**WORK IN PROGRESS**: if you clone this, it's incomplete and won't work.

Just because I wanted to win a bet, some machine learning about the afa world cup :)

This work is based (but has important modifications) on my previous work on the world cup predictions: https://github.com/fisadev/world_cup_learning

You can see the main code and results (including some neat graphs) here: http://nbviewer.ipython.org/github/fisadev/afa_cup_learning/blob/master/learn.ipynb

To run this on your machine, you will need to install the requirements in the ``requirements.txt`` file using pip (inside a virtualenv if you can, as usual, to avoid sudo), but also these extra packages not via pip, but apt or the tool you use in your system for compiled packages:

* numpy
* scipy

Under ubuntu you would run:

    sudo apt-get install python-numpy python-scipy
    
Then, inside a virtualenv with access to your system site-packages:

    pip install -r requirements.txt
    
(Or, if you don't want to use virtualenv, the same but with sudo)
