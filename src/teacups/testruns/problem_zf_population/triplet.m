clear all

Sys.S = 1;
Sys.g = 2.003;
Sys.D = [-700, 100];
Sys.lwpp = 1;
Sys.initState = {[0.3 0.2 0.1], 'zerofield'};

Exp.Range = [320, 380];
Exp.nPoints = 600;
Exp.mwFreq = 9.75;
Exp.Harmonic = 0;

sim = pepper(Sys, Exp);

Sys.initState = {[0.1, 0.2, 0.3], 'eigen'};

sim_2 = mypepper(Sys, Exp);
field = linspace(Exp.Range(1), Exp.Range(2), Exp.nPoints);
sim = sim/max(abs(sim));
sim_2 = sim_2/max(abs(sim_2));
plot(field, sim)
hold on
plot(field, sim_2)
hold off


A = [field; sim];
writematrix(A, "M_tri.txt", "Delimiter", "tab")