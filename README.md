# ESZ-Disaggregation
Discovergy, führender Komplettanbieter von Smart Metering-Lösungen, stattet im Rahmen des Pilotprogramms „Einsparzähler“ Kunden mit intelligenten Stromzählern (Smart Metern) aus. Damit wird der Energieverbrauch der Kunden sekundengenau, für jede einzelne Phase und in Echtzeit erfasst, analysiert und visualisiert. Über das Discovergy Web-Portal und die App haben die Einsparzähler-Kunden ihren Energieverbrauch somit jederzeit im Blick. Sie können einzelne Verbraucher erkennen, sich mit anderen Nutzern vergleichen und erhalten personalisierte Energiespartipps. Ermöglicht wird dies insbesondere durch die Zuordnung von Verbrauchskurven auf einzelne Geräte(-gruppen), die sogenannte automatische Geräteerkennung mittels NILM (non-intrusive load monitoring), KI und Big Data Technologie.
 
Ziel ist es, den Pilotkunden bei der Einsparung von möglichst viel Strom zu helfen und Einsparpotentiale durch Verbrauchsinformationen und Handlungsempfehlungen zu erschließen.

Die NILM-Algorithmen werden hier als Open Source Code veröffentlicht. Die Funktionsbeschreibung ist im Quellcode enthalten.

Das Projekt wird vom Bundesamt für Wirtschaft und Ausfuhrkontrolle im Auftrag des Bundesministeriums für Wirtschaft und Energie gefördert.

Weitere Informationen: www.einsparzaehler.de


Discovergy, the leading full-stack provider of smart metering solutions, supports customers with intelligent electricity meters (smart meters) as part of the “Einsparzähler” pilot project. This means that the customers' energy consumption is recorded, analyzed and visualized for each individual phase (up to 6 phases in bi-directional meters) in real time. Via the Discovergy web portal and the app, savings meter customers can keep an eye on their energy consumption at all times. They can recognize individual consumers, compare themselves with other users and receive personalized tips on how to save energy. This is made possible in particular by the assignment of consumption curves to individual devices (groups), the so-called automatic device detection and disaggregation using NILM (non-intrusive load monitoring), AI algorithms and Big Data Technologies.

The aim is to help pilot customers save as much electricity as possible and to tap potential savings through consumption information and recommendations for action.

The NILM algorithms are published here as open source code. The functional description is contained in the source code.

The project is funded by the Federal Office for Economic Affairs and Export Control on behalf of the Federal Ministry for Economic Affairs and Energy Germany.


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



