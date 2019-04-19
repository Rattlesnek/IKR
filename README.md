# IKR
World's best classifier!


## Usage
```
usage: main.py [-h] [-t] [-p] [-m MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           train model and save it
  -p, --predict         load model and predict
  -m MODEL, --model MODEL
                        specify model for training / predicting
```


## Poznamky

Stale musite zavolat `main.py` s `-m MODEL` a urcit ci chcete trenovat alebo predikovat pomocou `-t` alebo `-p`

`MODEL` moze byt bud:
* `VGG`
* `VGG+SVM`
* `SVM`
* ...
* doplnte dalsie modely ktore mate


V `main.py` su dve funkcie:
* `training()`
* `prediction()`

Do oboch doplnte zavolanie funkcie z vasho modulu, ktora je na trenovanie / predikovanie.
Inspirujte sa tym ako je zavolane `vtf.execute_training()` a `vtf.execute_prediction()`.

`vtf.execute_training()` ma ako parametre cestu k modelu kde sa ulozi novy natrenovany model.
`vtf.execute_prediction()` ma ako parametre cestu k modelu ktory sa nacita a nasledne sa pomocou neho predukuje.


