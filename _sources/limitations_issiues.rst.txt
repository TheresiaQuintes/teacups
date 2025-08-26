=============================
Limitations and known issiues
=============================


Limitations
-----------
Appropriate relaxation times and time traces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You should set :math:`T_{\text{relax}} > 10^{-8}` o.r. :math:`k < 10^8`. Otherwise the spectrum will get very broad. In order to see relaxation effects choose :math:`t \approx 2\cdot T_{\text{relax}}`

Relaxation models
^^^^^^^^^^^^^^^^^
Be aware that TEACUPS at the current state can only simulate models with transitions between the energy levels of the spin system. It is not possible to calculate changes of the system (e.g. conformational changes/ chemical reactions), which would change the energy levels. In order to do so, you would have to add a second spin system with changed parameters.

Secular approximation
^^^^^^^^^^^^^^^^^^^^^
Be aware that the secular approximation is used. For systems with large anisotropies the signal may become inaccurate.


Known issiues
-------------
RuntimeError on Windows system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the skript does not stop running and shows repeatedly a ``RuntimeError`` you may have forgotten to add ``if __name__ == "__main__"`` to your skript (See section: *Getting started*)

Bends in the spectrum for long times
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you see bends in the spectrum along the time axis for long times :math:`t` you can remove the mistake by raising the number of magnetic field points in ``Exp.B_z``.

Hyperfine coupling in coupled systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The hyperfine coupling does not work for coupled systems (i.e. radical pairs and triplet doublet pairs) until now. This issiue shall be removed in future versions.

