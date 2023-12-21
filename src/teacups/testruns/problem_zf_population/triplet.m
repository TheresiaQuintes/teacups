clear all

Sys.S = 1;
Sys.g = 2.003;
Sys.D = [700, -100];
Sys.lwpp = 1;
Sys.initState = {[0.3 0.2 0.1], 'xyz'};

Exp.Range = [320, 380];
Exp.nPoints = 600;
Exp.mwFreq = 9.75;
Exp.Harmonic = 0;

sim = pepper(Sys, Exp);
field = linspace(Exp.Range(1), Exp.Range(2), Exp.nPoints);
sim = normalize(sim);
plot(field, sim)

A = [field; sim];
writematrix(A, "M_tri.txt", "Delimiter", "tab")