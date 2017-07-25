# Tino's analysis workflow

## used Monte Carlo
Everything so far — if not mentioned otherwise — was done on the ASTRI mini-array;
exclusively on the innermost 3×3 grid.

![ASTRI mini array](https://cloud.githubusercontent.com/assets/18286015/20310332/d497f1b8-ab4b-11e6-84fe-ee9ccb901623.png)

For this array, both Gammas and Protons have been simulated with an E^-2 energy spectrum.
For Gammas in an energy range from 0.1 TeV to 330 TeV  as a point source and
for Protons in an energy range from 0.1 TeV to 600 TeV as a diffuse flux 6° around the
pointing direction.

## Reconstruction Workflow
The reconstruction is done by the `FitGammaHillas` class that has already been merged into
ctapipe. <br  />
Only events with at least 2 telescopes are considered.<br  />

### Cleaning
The cleaning is done either with wavelets or with tailcuts:

#### wavelets
The wavelet cleaning algorithm expects a 2D array of a rectangular image. Therefore, the
non-rectangular image of the ASTRI cameras needs to be cut down for now.<br  />
The wavelet cleaning is applied as provided by Jérémie.

The wavelet cleaning leaves few isolated pixels or islands that can have a negative
impact on the Hillas parametrisation. Remove these islands and only leave the biggest
patch of connected pixels using a method provided by `scipy.ndimage` and implemented by
Fabio.

#### tailcuts
Uses the standard 2-step threshold implementation of ctapipe with the two values
at 5 and 10.<br  />

### edge rejection
Previously only applied to wavelets, now to both:<br  />
If any pixel in the outermost pixel layer records a charge (after cleaning) of more than
a fifth of the maximum pixel's charge (rough by-eye optimisation), the whole image is
rejected.

### Hillas Parametrisation
The cleaned images are given to the Hillas parametrisation implemented in ctapipe.


### Reconstruction
The reconstruction uses three of the provided Hillas parameters:
The position of the image core (x and y) and the tilt of the Hillas ellipsis (`ψ`).
From these parameters (position `c` and tilt `ψ`), a second position, `b`, on the camera
can be calculated. Both these points should lie on the shower axis. <br  />

![shower_reco_camera_frame](https://cloud.githubusercontent.com/assets/18286015/21807318/75c52498-d73e-11e6-868a-c1a7d4413a49.png)

#### GreatCircle
The two positions on the camera correspond to two directions in the sky. These two
directions define a plane (or `GreatCircle`) in the horizontal frame.
Since the shower on the image passes through the two points `c` and `b` on the camera,
the shower should also lie in the plane of the `GreatCircle`. <br  />
The distribution of the angle between the shower and the `GreatCircle` can be seen in the following figure:


#### Direction Reconstruction
Two cameras will see the shower from different directions. The `GreatCircle` defined by
their image will in most cases not be parallel, though the shower direction should lie
in both planes. Therefore, the shower direction should be the orientation of the cross section of two planes. <br  />
If more than two telescopes observe the shower, every unique pair of telescopes provides
an estimator of the cross section. All direction estimators get summed (while normalised
to a length of 1) with a weight provided by:
* the cosine of the angle between the two planes that were crossed
* the total size of each cleaned image of the two cameras used for the two `GreatCircle`
* the ratios of the Hillas length and width from the two cameras.

![shower_reco_horizontal_frame](https://cloud.githubusercontent.com/assets/18286015/21807321/780f10a6-d73e-11e6-8e8b-1f616c31c9fd.png)

#### Shower Core Reconstruction
For the impact position, the normal vector of the `trace` on the ground of each
`GreatCircle` is defined as the the `GreatCircle`'s normal vector with the z component set
to zero. Again, for ever `GreatCircle` the shower should lie in the circle and hit the
ground where the `GreatCircle` crosses the ground, too. That means: The shower's impact
position is somewhere on the trace. Since this is true for all traces, the shower core
position `(x, y)` ought to lie where all the traces cross. This is can be expressed with
an equation system in matrix form:
![full_equation_system](https://cloud.githubusercontent.com/assets/18286015/25378070/9071d94a-29a9-11e7-9026-370dabd31625.png)

or <br  />
<img src="https://cloud.githubusercontent.com/assets/18286015/25378068/9070c08c-29a9-11e7-88ed-32f5b41e7a1f.png" height=50px>,

where <img src="https://cloud.githubusercontent.com/assets/18286015/25378071/90723066-29a9-11e7-90ee-33db7ea3c356.png" height=25px>
is the normal vector of the trace of telescope i and
<img src="https://cloud.githubusercontent.com/assets/18286015/25378072/90792bb4-29a9-11e7-957c-c6e70ff7d681.png" height=25px>
is the position of telescope i. <br  />

Since we do not live in a perfect world and there probably is no point `(x, y)` that
fulfils this equation system, it is solved by the method of least linear square:

![rchisq_eq](https://cloud.githubusercontent.com/assets/18286015/25378074/909b69c2-29a9-11e7-8036-271fd7adcbda.png)


<img src="https://cloud.githubusercontent.com/assets/18286015/25378073/9082ee56-29a9-11e7-9772-2372cbebaa14.png" height=25px> minimises the
squared difference of
<img src="https://cloud.githubusercontent.com/assets/18286015/25378069/907195b6-29a9-11e7-8582-2004977288e0.png" height=25px>;
with weights like the parameters in last two points from the enumeration above.


## Discrimination

For the discrimination, the images get cleaned and parametrised and the event
reconstructed as above.

### learning
Consecutively, a RandomForestClassifier is trained on the separate images with
the following features:
* distance of the shower impact to the telescope,
* signal on the telescope,
* total signal on all selected telescopes,
* number of selected telescopes,
* Hillas parameters:
    * width and length
    * Skewness,
    * Kurtosis,
    * Asymmetry


### predicting
For the prediction which class (gamma/proton) an event belongs to, all selected telescopes
(with the parameters as in the training) are given to the classifier.<br  />
_Note: for the learning, the impact distance to the MC shower core is used, for the
prediction, the reconstructed shower core)_<br  />
The event is selected as a gamma-shower when more 4 (for now) telescopes have participated
in the classification and more than 75 % of the telescopes were classified as gammas.


## Sensitivity
To calculate the sensitivity, the (energy-dependent) *effective Area* has to be
determined. For this, the number of selected events is to be divided by the number of
simulated events:
```
simulated = generated * N_reuse * N_files
efficiency = selected / simulated
```
with `generated` the number generated showers per file (5000). Each shower is used
`N_reuse` times with the impact position randomised within the `generation_area`.
`N_reuse` is 10 for gamma events and 20 for proton events. `N_files` is the number of
files used here, 9 for the gamma channel, 51 for protons. The `effective_area` then is the
product of the `generation_area` and the `efficiency`:
```
generation_area = π * radius²
effective_area = efficiency * generation_area
```

with `radius` being 1000 m for gammas and 2000 m for protons.<br  />

The number of expected events from any given source can be determined by multiplying
the source's (non-differential) flux with an assumed observation duration and  the
`effective_area` of the detection-reconstruction-discrimination chain.

### significance
The gamma events have been simulated coming from a point-source while the proton events
were simulated as a diffuse flux. To calculate the significance, an on- and an off-region
can be defined around the direction point-source. With the numbers of events in both
regions, a significance can be calculated according to equation (17) of Li & Ma (1983):
```
'''
Non   - Number of on counts
Noff  - Number of off counts
alpha - Ratio of on-to-off exposure
'''

alpha1 = alpha + 1.0
sum    = Non + Noff
arg1   = Non / sum
arg2   = Noff / sum
term1  = Non  * np.log((alpha1/alpha)*arg1)
term2  = Noff * np.log(alpha1*arg2)
sigma  = np.sqrt(2.0 * (term1 + term2))
```

A sensitivity, binned in energy, can be defined as the signal flux needed to claim
* 5 sigma significance
* within 50 hours of observation
* with at least 10 events per energy bin and
* a background contamination of less than 5 % per energy bin.
