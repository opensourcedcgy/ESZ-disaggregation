#### Dishwasher
Training data: trace, uk_dale, refit
- exp261:
    - parent exp260
    - F1 = 0.88, MAE = 10.76

- exp254:
    - parent  exp253:
    - valid loss = 0.00350, train loss = 0.00144
    - F1 = 0.91, MAE = 13.67 

#### Washing machine
Training data: trace, uk_dale, DRED
- exp249:
    - parent exp248
    - F1 = 0.9, MAE = 4.28

- exp221:
    - parent: exp220
    - MAE = 13.53

- exp199:
    - train from scratch
    - F1 = 0.672, MAE = 25.61 

#### Refrigerator
Training data: uk_dale, trace, DRED
- exp120
    - parent exp117
    - MAE = 15.94


- exp070
    - parent exp068
    - MAE = 26.768961351
    - Lower than exp068 but performs better on unseen REDD

- exp068:
    - parent exp067
    - MAE = 18.57 

#### Dryer:
Training data: peccan dataport + trace + refit
Issue: this type of dryer is quite close to washing machine, shall we merge these two types as one?
- exp641:
    - parent exp639
    - valid RMSE = 0.025
    - rae house 1: F1 = 0.816, MAE = 39.605
    - rae house 2: F1 = 0.886, MAE = 46.33

- exp639:
    - parent exp636
    - valid RMSE = 0.027
    - rae house 1: F1 = 0.811
    - rae house 2: F1 = 0.902

- exp636:
    - train from scratch
    - rae house 1: F1 = 0.813
    - rae house 2: F1 = 0.90


