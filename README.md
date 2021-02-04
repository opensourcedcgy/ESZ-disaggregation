# ESZ-Disaggregation
Discovergy, führender Komplettanbieter von Smart Metering-Lösungen, stattet im Rahmen des Pilotprogramms „Einsparzähler“ Kunden mit intelligenten Stromzählern (Smart Metern) aus. Damit wird der Energieverbrauch der Kunden sekundengenau, für jede einzelne Phase und in Echtzeit erfasst, analysiert und visualisiert. Über das Discovergy Web-Portal und die App haben die Einsparzähler-Kunden ihren Energieverbrauch somit jederzeit im Blick. Sie können einzelne Verbraucher erkennen, sich mit anderen Nutzern vergleichen und erhalten personalisierte Energiespartipps. Ermöglicht wird dies insbesondere durch die Zuordnung von Verbrauchskurven auf einzelne Geräte(-gruppen), die sogenannte automatische Geräteerkennung mittels NILM (non-intrusive load monitoring).
 
Ziel ist es, den Pilotkunden bei der Einsparung von möglichst viel Strom zu helfen und Einsparpotentiale durch Verbrauchsinformationen und Handlungsempfehlungen zu erschließen.

Die NILM-Algorithmen werden hier als Open Source Code veröffentlicht. Die Funktionsbeschreibung ist im Quellcode enthalten.

Das Projekt wird vom Bundesamt für Wirtschaft und Ausfuhrkontrolle im Auftrag des Bundesministeriums für Wirtschaft und Energie gefördert.

Weitere Informationen: www.einsparzaehler.de


### Project structure
- ESZ-disaggregation
    - classifiers: All classifiers' file
    - dataset: Test datasets
    - models: All TF model files in "assets, variables, saved_model.pb"
    - serve: Run files to get results
    - train_history: Training history
    - utils: Utilities used for event detection and so on
    - config.py: Some configurations
    - LICENSE
    - README.md
    - requirements.txt
     

#### How to run
Run the test() in the serve.run_disaggregation.py to get all the disaggregation results.
Tested with Python 3.6.

#### Challenges
##### Detection or Disaggregation?
- One challenge in disaggregation (NILM) during our work is to get higher detection rate (such as higher F1-Score) ,
while also get higher accurate energy estimation (such as MAE, RAE, MSE) at the same time. It's all about time and how we compromise
between two targets.
##### Real world?
- Another challenge is the differences between different datasets as well as the differences between the real world.
As we all know the deep learning network normally performs better on scaled data like values [0,1] or [-1,1]. But how can we scale the values that is unknown? 

#### Traning History

Some training histories of different appliances are also added here. May help you to get a better performance.

- We have tried many different kinds of network basic structures from DNN, RNN, LSTM, GRU to CNN, we finally find out CNN is the most convenient one.
    - CNN achieves almost same or even better performance as expected from RNN, LSTM.
    - CNN can be trained within shorter time as in comparison to RNN, LSTM.
    - CNN networks can be optimised easily with many optimisation methods.

- We are still testing some more complicated network structurs such as GAN, VAE and RL.
    - As right now the results from GAN give us quite promising performance, but the big issue with GAN is that we can't figure out stable training procedure.
    - VAE is also quite nice to have and to be used as feature engineering.
    - RL is still on the way. We need to figure out a nice and stable reward feature. 


### Thanks
Hereby also many thanks to all the helps from all over the world during our works.
We've got a lot of ideas and pre-works from them.

- NILMTK https://github.com/nilmtk/nilmtk
- EU NILM WORKSHOP http://www.nilm.eu/
- Tensorflow



