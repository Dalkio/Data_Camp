# Data Camp

Project I worked on at École Polytechnique.
The purpose is to verificate anti-cancer drugs before their administration.
That is, the goal here is to check that they contain the good chemotherapeutic agent with the good dosage.
We can see it as a pipeline of a classification and a regression problems, that leads us to two objectives:
* classification: predict which molecule it corresponds to given the spectrum.
* regression: predict the concentration of a molecule. The prediction should not depend on the vial or the solute group. The error metric is the mean absolute relative error.
